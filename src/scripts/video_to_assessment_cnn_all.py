import os
import json
import argparse
import cv2
import numpy as np
import mediapipe as mp
import torch
import torch.nn as nn

from vc_core import safe_name, segment_reps_from_sequence

# -----------------------------
# Torch load compatibility (PyTorch 2.6+)
# -----------------------------
def torch_load_compat(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)
    except Exception:
        # only do this if you trust your checkpoint file
        return torch.load(path, map_location=device, weights_only=False)

# -----------------------------
# CNN model (same as training)
# -----------------------------
class CNNSubjectRegressor(nn.Module):
    def __init__(self, in_feats, n_aspects):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_feats, 64, 5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, 5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.attn = nn.Linear(128, 1)
        self.head = nn.Sequential(
            nn.Linear(128, n_aspects),
            nn.Sigmoid()
        )

    def forward(self, x, mask):
        B, R, T, F = x.shape
        x = x.view(B * R, T, F).transpose(1, 2)      # (B*R, F, T)
        emb = self.encoder(x).squeeze(-1)            # (B*R, 128)
        emb = emb.view(B, R, -1)                     # (B, R, 128)

        scores = self.attn(emb).squeeze(-1)          # (B, R)
        scores = scores.masked_fill(mask == 0, -1e9)
        w = torch.softmax(scores, dim=1)

        subj = (emb * w.unsqueeze(-1)).sum(dim=1)    # (B, 128)
        return self.head(subj)                       # (B, A)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--exercise", required=True)
    ap.add_argument("--view", required=True, choices=["front", "side"])
    ap.add_argument("--out", default="assessment.json")

    # either give model path OR models_dir
    ap.add_argument("--model", default=None)
    ap.add_argument("--models_dir", default=None)

    # optional overrides (leave empty to use per-exercise defaults from vc_core)
    ap.add_argument("--win", type=int, default=None)
    ap.add_argument("--min_dist", type=int, default=None)
    ap.add_argument("--thresh_pct", type=float, default=None)

    ap.add_argument("--stride", type=int, default=1, help="Process every Nth frame (speed).")
    ap.add_argument("--max_frames", type=int, default=None, help="Stop after N processed frames (debug).")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -----------------------------
    # Resolve model path
    # -----------------------------
    if args.model:
        model_path = args.model
    else:
        if not args.models_dir:
            raise ValueError("Provide either --model or --models_dir")
        model_path = os.path.join(args.models_dir, f"{safe_name(args.exercise)}_best.pt")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    ckpt = torch_load_compat(model_path, device)
    aspect_cols = ckpt["aspect_cols"]
    feature_dim = int(ckpt.get("feature_dim", 9))
    if feature_dim != 9:
        raise RuntimeError(f"Expected feature_dim=9, but checkpoint has {feature_dim}")

    model = CNNSubjectRegressor(in_feats=9, n_aspects=len(aspect_cols)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # -----------------------------
    # MediaPipe Pose extraction -> lm_seq_xyz
    # -----------------------------
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(model_complexity=1)
    cap = cv2.VideoCapture(args.video)

    lm_seq_xyz = []
    last_good = None
    processed = 0
    frame_i = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_i += 1
        if args.stride > 1 and (frame_i % args.stride != 0):
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)

        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark
            arr = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)  # (33,3)
            last_good = arr
            lm_seq_xyz.append(arr)
        else:
            # keep sequence continuity (important for rep detection)
            if last_good is not None:
                lm_seq_xyz.append(last_good.copy())

        processed += 1
        if args.max_frames is not None and processed >= args.max_frames:
            break

    cap.release()
    pose.close()

    if len(lm_seq_xyz) < 50:
        raise RuntimeError("Video too short or pose detection failed (too few frames with pose).")

    # -----------------------------
    # Rep segmentation (exercise+view specific from vc_core)
    # -----------------------------
    reps, peaks, signal, used = segment_reps_from_sequence(
        exercise=args.exercise,
        view=args.view,
        lm_seq_xyz=lm_seq_xyz,
        win=args.win,
        min_dist=args.min_dist,
        thresh_pct=args.thresh_pct
    )

    R = int(reps.shape[0])
    if R == 0:
        raise RuntimeError(
            "No reps detected. Try adding --debug, or override with --min_dist / --win."
        )

    # -----------------------------
    # CNN inference (0–10 scale)
    # -----------------------------
    x = torch.from_numpy(reps).unsqueeze(0).float()      # (1, R, T, 9)
    mask = torch.ones(1, R, dtype=torch.float32)         # all reps present

    with torch.no_grad():
        pred01 = model(x.to(device), mask.to(device)).cpu().numpy()[0]  # (A,)

    scores_0_10 = (pred01 * 10.0).tolist()

    result = {
        "exercise": args.exercise,
        "view": args.view,
        "model_path": model_path,
        "n_frames_used": int(len(lm_seq_xyz)),
        "n_reps": int(R),
        "scores_0_10": {aspect_cols[i]: round(scores_0_10[i], 2) for i in range(len(aspect_cols))}
    }

    if args.debug:
        result["debug"] = {
            "peaks": [int(p) for p in peaks],
            "used_params": used,
            "signal_preview": {
                "min": float(np.min(signal)),
                "max": float(np.max(signal)),
                "mean": float(np.mean(signal)),
            }
        }

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    print("\n✅ Assessment complete")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
