# vc_core.py
import os
import re
import numpy as np
import pandas as pd
from difflib import SequenceMatcher

# =========================================================
# Small utils
# =========================================================
def safe_name(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", str(s)).strip("_")

def norm_col(c):
    return re.sub(r"[^a-z0-9]+", "_", str(c).lower()).strip("_")

def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

def normalize_vid(v):
    """Converts 'V03', 'v3', '03', 3 -> 3"""
    if pd.isna(v):
        return None
    if isinstance(v, (int, np.integer)):
        return int(v)
    digits = re.findall(r"\d+", str(v))
    if not digits:
        return None
    return int(digits[0])

def infer_volunteer_col(df):
    for c in df.columns:
        nc = norm_col(c)
        if any(k in nc for k in ["volunteer", "subject", "participant", "person", "id"]):
            return c
    return None

# =========================================================
# Annotation loader (0.75*C2 + 0.25*C1)
# Works even if columns are like:
#   "Aspect (C1)" + "Aspect (C2)"   OR
#   "C1 Aspect" + "C2 Aspect"       OR
#   already single aspect columns
# =========================================================
def load_weighted_annotation_for_exercise(exercise, annotation_dir, min_sim=0.60):
    ex_norm = norm_col(exercise)

    best_file, best_score = None, 0.0
    for f in os.listdir(annotation_dir):
        if not f.lower().endswith(".xlsx"):
            continue
        name = re.sub(r"^\d+\)\s*", "", f)  # remove "10) "
        name_norm = norm_col(name.replace(".xlsx", ""))
        sc = similarity(ex_norm, name_norm)
        if sc > best_score:
            best_score = sc
            best_file = f

    if best_file is None or best_score < min_sim:
        raise FileNotFoundError(f"No annotation file matches '{exercise}' (best similarity={best_score:.2f})")

    path = os.path.join(annotation_dir, best_file)
    df = pd.read_excel(path)

    vol_col = infer_volunteer_col(df)
    if vol_col is None:
        raise ValueError(f"Volunteer/Subject column not found in {best_file}")

    # numeric cols excluding volunteer
    num_cols = []
    for c in df.columns:
        if c == vol_col:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().any():
            num_cols.append(c)

    if not num_cols:
        raise ValueError(f"No numeric columns found in {best_file}")

    # group cols into aspects with possible C1/C2 pairs
    # detect coach tag
    def coach_tag(col):
        nc = norm_col(col)
        if "c1" in nc or "coach1" in nc or "coach_1" in nc:
            return "c1"
        if "c2" in nc or "coach2" in nc or "coach_2" in nc:
            return "c2"
        return None

    # base aspect name (remove coach markers)
    def base_aspect(col):
        nc = norm_col(col)
        nc = re.sub(r"(^|_)c1(_|$)", "_", nc)
        nc = re.sub(r"(^|_)c2(_|$)", "_", nc)
        nc = nc.replace("coach1", "").replace("coach2", "")
        nc = re.sub(r"_+", "_", nc).strip("_")
        return nc

    groups = {}
    singles = []
    for c in num_cols:
        tag = coach_tag(c)
        base = base_aspect(c)
        if tag is None:
            singles.append(c)
        else:
            groups.setdefault(base, {})[tag] = c

    # Final aspect columns (human-readable)
    aspect_cols = []

    # If sheet already has single aspect columns (no coach split)
    # we just use them directly (no weighting).
    if len(groups) == 0:
        aspect_cols = singles
        labels = {}
        for _, row in df.iterrows():
            vid = normalize_vid(row[vol_col])
            if vid is None:
                continue
            labels[int(vid)] = row[aspect_cols].to_numpy(np.float32)
        return labels, aspect_cols, best_file, best_score

    # Otherwise compute weighted per base aspect
    # Make stable aspect names from base keys
    base_keys = sorted(groups.keys())
    aspect_cols = base_keys  # these will be used as names in outputs

    labels = {}
    for _, row in df.iterrows():
        vid = normalize_vid(row[vol_col])
        if vid is None:
            continue

        vals = []
        for base in base_keys:
            cols = groups[base]
            c1 = pd.to_numeric(row.get(cols.get("c1", None), np.nan), errors="coerce")
            c2 = pd.to_numeric(row.get(cols.get("c2", None), np.nan), errors="coerce")

            if pd.isna(c1) and pd.isna(c2):
                v = np.nan
            elif pd.isna(c1):
                v = float(c2)
            elif pd.isna(c2):
                v = float(c1)
            else:
                v = 0.25 * float(c1) + 0.75 * float(c2)

            vals.append(v)

        arr = np.array(vals, dtype=np.float32)
        if np.isnan(arr).any():
            # skip rows missing aspect grades
            continue
        labels[int(vid)] = arr

    return labels, aspect_cols, best_file, best_score


# =========================================================
# Feature extraction (EXACT match to training lateral script)
# IMPORTANT: angles are in RADIANS (np.arccos), not degrees
# =========================================================
# MediaPipe pose landmark indices (fixed)
LS, RS = 11, 12
LE, RE = 13, 14
LW, RW = 15, 16
LH, RH = 23, 24
LK, RK = 25, 26
LA, RA = 27, 28
LHEEL, RHEEL = 29, 30

def extract_9_features(lm_xyz):
    def angle(a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        ba = a - b
        bc = c - b
        cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        return np.arccos(np.clip(cos, -1.0, 1.0))  # radians

    LS_p, RS_p = lm_xyz[LS], lm_xyz[RS]
    LE_p, RE_p = lm_xyz[LE], lm_xyz[RE]
    LW_p, RW_p = lm_xyz[LW], lm_xyz[RW]
    LH_p, RH_p = lm_xyz[LH], lm_xyz[RH]

    feats = [
        angle(LE_p, LS_p, LH_p),      # left shoulder-hip angle
        angle(RE_p, RS_p, RH_p),      # right shoulder-hip angle
        angle(LW_p, LE_p, LS_p),      # left elbow flex angle
        angle(RW_p, RE_p, RS_p),      # right elbow flex angle
        LW_p[1] - LS_p[1],            # left wrist relative y
        RW_p[1] - RS_p[1],            # right wrist relative y
        abs(LS_p[1] - RS_p[1]),       # shoulder symmetry
        abs(LW_p[1] - RW_p[1]),       # wrist symmetry
        abs(LE_p[1] - RE_p[1]),       # elbow symmetry
    ]
    return np.array(feats, dtype=np.float32)

def smooth(x, k=7):
    k = int(max(1, k))
    if k == 1:
        return x
    return np.convolve(x, np.ones(k, dtype=np.float32) / k, mode="same")

def detect_peaks(signal, mode="max", thresh_pct=75, min_dist=20):
    s = smooth(np.asarray(signal, dtype=np.float32), k=7)
    if len(s) < (2 * min_dist + 5):
        return []

    if mode == "max":
        thr = np.percentile(s, thresh_pct)
        cond = lambda i: (s[i] > thr and s[i] > s[i-1] and s[i] > s[i+1])
    else:
        thr = np.percentile(s, 100 - thresh_pct)
        cond = lambda i: (s[i] < thr and s[i] < s[i-1] and s[i] < s[i+1])

    peaks = []
    for i in range(min_dist, len(s) - min_dist):
        if cond(i):
            if not peaks or (i - peaks[-1] > min_dist):
                peaks.append(i)
    return peaks

# =========================================================
# Exercise+view specific rep signals
# (rep segmentation differs, but FEATURES stay identical)
# =========================================================
def _mean2(a, b):
    return 0.5 * (a + b)

# =========================================================
# Threshold UP->DOWN->UP (or DOWN->UP->DOWN) segmentation
# The user-requested logic:
#   pick a threshold level; whenever signal is below it => "down",
#   whenever above it => "up"; count a rep when we observe:
#     UP -> DOWN -> UP  (for exercises where UP is "high")
#   or
#     UP -> DOWN -> UP  with inverted comparison (UP is "low")
#
# This avoids "rep duration" heuristics and is robust to noisy peaks.
# =========================================================
def detect_reps_threshold_updownup(
    signal,
    up_is_high: bool = True,
    p_low: float = 20.0,
    p_high: float = 80.0,
    thr: float = None,
    smooth_k: int = 9,
    debounce: int = 3,
    extreme_mode: str = "min",
):
    """
    Returns:
      peaks: list[int] rep centers (frame indices) at the extreme inside each cycle
      debug: dict
    Notes:
      - 'up_is_high' defines whether "UP" corresponds to signal >= threshold.
      - threshold is computed from percentiles unless 'thr' is provided.
      - 'debounce' requires a state to persist for N frames before committing a transition.
      - 'extreme_mode' chooses the rep center inside the cycle:
          'min' => argmin between (enter_down .. return_up)
          'max' => argmax between (enter_down .. return_up)
    """
    s = smooth(np.asarray(signal, dtype=np.float32), k=int(max(1, smooth_k))).astype(np.float32)
    n = len(s)
    if n < 10:
        return [], {"method": "threshold", "reason": "too_short", "n": int(n)}

    # robust threshold from percentiles
    lo = float(np.percentile(s, p_low))
    hi = float(np.percentile(s, p_high))
    if thr is None:
        thr = 0.5 * (lo + hi)
    thr = float(thr)

    # define boolean "UP"
    if up_is_high:
        is_up = s >= thr
    else:
        is_up = s <= thr
    is_down = ~is_up

    # --- debounce: compress frames into stable states ---
    # state: 1 = up, 0 = down
    state = np.full(n, -1, dtype=np.int8)
    cur = 1 if is_up[0] else 0
    cnt = 1
    state[0] = cur
    for i in range(1, n):
        new = 1 if is_up[i] else 0
        if new == cur:
            cnt += 1
        else:
            # require new to persist for debounce frames
            cnt = 1
            # lookahead: if next frames confirm, flip
            j = i
            ok = True
            for k in range(1, debounce):
                if j + k >= n:
                    ok = False
                    break
                if (1 if is_up[j + k] else 0) != new:
                    ok = False
                    break
            if ok:
                cur = new
        state[i] = cur

    # indices where debounced state changes
    changes = np.where(state[1:] != state[:-1])[0] + 1
    # build segments of constant state
    seg_starts = np.concatenate([[0], changes])
    seg_ends = np.concatenate([changes - 1, [n - 1]])
    seg_states = state[seg_starts]  # 1=up,0=down

    peaks = []
    cycles = 0

    # We want: UP -> DOWN -> UP (full rep)
    i = 0
    while i + 2 < len(seg_states):
        if seg_states[i] == 1 and seg_states[i + 1] == 0 and seg_states[i + 2] == 1:
            # full cycle
            up1_start, up1_end = int(seg_starts[i]), int(seg_ends[i])
            down_start, down_end = int(seg_starts[i + 1]), int(seg_ends[i + 1])
            up2_start, up2_end = int(seg_starts[i + 2]), int(seg_ends[i + 2])

            # choose center at extreme inside [up1_end .. up2_start]
            a = up1_end
            b = up2_start
            if b <= a:
                i += 1
                continue
            seg = s[a:b + 1]
            if extreme_mode == "max":
                pk = int(a + np.argmax(seg))
            else:
                pk = int(a + np.argmin(seg))
            peaks.append(pk)
            cycles += 1
            i += 2  # move to the second UP as new start
        else:
            i += 1

    debug = {
        "method": "threshold",
        "thr": thr,
        "p_low": float(p_low),
        "p_high": float(p_high),
        "lo": lo,
        "hi": hi,
        "up_is_high": bool(up_is_high),
        "smooth_k": int(smooth_k),
        "debounce": int(debounce),
        "extreme_mode": str(extreme_mode),
        "n_peaks": int(len(peaks)),
        "signal_stats": {"min": float(np.min(s)), "max": float(np.max(s)), "mean": float(np.mean(s))},
    }
    return peaks, debug


def _pick_side_by_motion(sig_L, sig_R):
    # choose the side with higher std (more motion), helps side-view occlusion
    sL = np.std(sig_L)
    sR = np.std(sig_R)
    return sig_L if sL >= sR else sig_R

def build_rep_signal(lm_arr, exercise, view):
    """
    lm_arr: (N,33,3)
    Returns: 1D signal (N,)
    """
    ex = norm_col(exercise)

    # common y series
    yLS = lm_arr[:, LS, 1]; yRS = lm_arr[:, RS, 1]
    yLW = lm_arr[:, LW, 1]; yRW = lm_arr[:, RW, 1]
    yLH = lm_arr[:, LH, 1]; yRH = lm_arr[:, RH, 1]
    yLA = lm_arr[:, LA, 1]; yRA = lm_arr[:, RA, 1]
    yLHEEL = lm_arr[:, LHEEL, 1]; yRHEEL = lm_arr[:, RHEEL, 1]

    shoulder_y = _mean2(yLS, yRS)
    hip_y = _mean2(yLH, yRH)

    # elbow angles over time (radians) from landmarks
    def elbow_angle_series(side="R"):
        if side == "R":
            a = lm_arr[:, RW, :]; b = lm_arr[:, RE, :]; c = lm_arr[:, RS, :]
        else:
            a = lm_arr[:, LW, :]; b = lm_arr[:, LE, :]; c = lm_arr[:, LS, :]
        ba = a - b
        bc = c - b
        cos = np.sum(ba * bc, axis=1) / (np.linalg.norm(ba, axis=1) * np.linalg.norm(bc, axis=1) + 1e-6)
        return np.arccos(np.clip(cos, -1.0, 1.0))

    angL = elbow_angle_series("L")
    angR = elbow_angle_series("R")

    # ---------- ARM RAISES / PRESS ----------
    if ex in ["lateral_raises", "standing_dumbbell_front_raises", "dumbbell_shoulder_press"]:
        relL = shoulder_y - yLW
        relR = shoulder_y - yRW

        if view == "front":
            return _mean2(relL, relR)
        return _pick_side_by_motion(relL, relR)

    # ---------- CURLS / EXTENSIONS / BENCH / ROWS (elbow flex) ----------
    if ex in ["hummer_curls", "ez_bar_curls", "seated_biceps_curls",
              "overhead_triceps_extension", "triceps_kickbacks",
              "inclined_dumbbell_bench_press", "rows"]:
        # curl peak = minimum angle -> use negative so we can detect MAX peaks
        sigL = -angL
        sigR = -angR
        if view == "front":
            return _mean2(sigL, sigR)
        return _pick_side_by_motion(sigL, sigR)

    # ---------- SHRUGS ----------
    if ex == "shrugs":
        # shrug up => shoulders go up => y smaller => use negative
        return -shoulder_y

    # ---------- SQUATS / SPLIT SQUAT / DEADLIFT ----------
    if ex in ["weighted_squats", "weighted_sqauts", "bulgarian_split_squat", "deadlift"]:
        # bottom position => hips lower => y larger => detect MAX peaks
        return hip_y

    # ---------- CALF RAISES ----------
    if ex == "calf_raises":
        # heel up => heel y smaller => use negative to detect max at top
        heel = _mean2(yLHEEL, yRHEEL)
        ankle = _mean2(yLA, yRA)
        sig = -(0.7 * heel + 0.3 * ankle)
        return sig

    # fallback (generic): use hip_y
    return hip_y

# parameters per exercise/view
# parameters per exercise/view
# - method: "peaks" or "threshold"
# - For "peaks": uses detect_peaks(mode, thresh_pct, min_dist)
# - For "threshold": uses UP->DOWN->UP logic on a per-exercise rep-signal

# =========================================================
# Alternating arms rep detection (for exercises like Standing Dumbbell Front Raises)
# =========================================================
def _arm_raise_rel_signals(lm_arr):
    """Returns two 1D signals (L,R) for arm raise height relative to shoulders.
    Higher value => wrist is higher relative to shoulder line (since y is downwards).
    """
    yLS = lm_arr[:, LS, 1]; yRS = lm_arr[:, RS, 1]
    yLW = lm_arr[:, LW, 1]; yRW = lm_arr[:, RW, 1]
    shoulder_y = _mean2(yLS, yRS)
    relL = shoulder_y - yLW
    relR = shoulder_y - yRW
    return relL.astype(np.float32), relR.astype(np.float32)

def detect_peaks_k(signal, mode="max", thresh_pct=75, min_dist=20, smooth_k=7):
    """Peak detector like detect_peaks(), but with configurable smoothing."""
    s = smooth(np.asarray(signal, dtype=np.float32), k=int(smooth_k))
    if len(s) < (2 * int(min_dist) + 5):
        return []
    if mode == "max":
        thr = np.percentile(s, float(thresh_pct))
        cond = lambda i: (s[i] > thr and s[i] > s[i-1] and s[i] > s[i+1])
    else:
        thr = np.percentile(s, 100.0 - float(thresh_pct))
        cond = lambda i: (s[i] < thr and s[i] < s[i-1] and s[i] < s[i+1])

    peaks = []
    md = int(min_dist)
    for i in range(md, len(s) - md):
        if cond(i):
            # enforce separation by keeping the stronger peak if too close
            if peaks and (i - peaks[-1] < md):
                prev = peaks[-1]
                if (mode == "max" and s[i] > s[prev]) or (mode != "max" and s[i] < s[prev]):
                    peaks[-1] = i
            else:
                peaks.append(i)
    return peaks

def detect_reps_alt_arms(relL, relR, thresh_pct=70, min_dist=18, smooth_k=7, enforce_alt=True):
    """Detect reps for alternating-arm movements.
    We detect peaks on each arm signal separately then merge them.
    - enforce_alt=True will avoid counting two consecutive peaks from the same arm (softly).
    Returns: peaks (list[int]), combined_signal (1D), debug dict
    """
    relL = np.asarray(relL, dtype=np.float32)
    relR = np.asarray(relR, dtype=np.float32)

    pkL = detect_peaks_k(relL, mode="max", thresh_pct=thresh_pct, min_dist=min_dist, smooth_k=smooth_k)
    pkR = detect_peaks_k(relR, mode="max", thresh_pct=thresh_pct, min_dist=min_dist, smooth_k=smooth_k)

    events = []
    for p in pkL:
        events.append((int(p), "L", float(relL[p])))
    for p in pkR:
        events.append((int(p), "R", float(relR[p])))

    events.sort(key=lambda x: x[0])

    # merge close events (global min_dist)
    merged = []
    md = int(min_dist)
    for e in events:
        if not merged:
            merged.append(e)
            continue
        if e[0] - merged[-1][0] < md:
            # keep the stronger peak
            if e[2] > merged[-1][2]:
                merged[-1] = e
        else:
            merged.append(e)

    # optional: softly enforce alternation (avoid L,L,L or R,R,R due to jitter)
    if enforce_alt and merged:
        alt = [merged[0]]
        for e in merged[1:]:
            if e[1] == alt[-1][1]:
                # same arm again: keep only if clearly stronger
                if e[2] > alt[-1][2]:
                    alt[-1] = e
            else:
                alt.append(e)
        merged = alt

    peaks = [e[0] for e in merged]

    # combined signal for preview/filtering
    comb = np.maximum(smooth(relL, k=int(smooth_k)), smooth(relR, k=int(smooth_k))).astype(np.float32)

    dbg = {
        "n_L": int(len(pkL)),
        "n_R": int(len(pkR)),
        "n_merged": int(len(peaks)),
        "enforce_alt": bool(enforce_alt),
    }
    return peaks, comb, dbg


EXERCISE_PARAMS = {
    # UPPER BODY (most work fine with peaks)
    "lateral_raises": {
        "front": dict(method="peaks", win=25, min_dist=20, thresh_pct=75, mode="max"),
        "side":  dict(method="peaks", win=25, min_dist=20, thresh_pct=75, mode="max"),
    },
    "standing_dumbbell_front_raises": {
        # Alternating front raises: one arm at a time.
        # We detect peaks per-arm and merge (more reliable than averaging both arms).
        "front": dict(method="alt_arms", win=25, min_dist=18, thresh_pct=70, mode="max", smooth_k=7, enforce_alt=True),
        "side":  dict(method="alt_arms", win=25, min_dist=18, thresh_pct=70, mode="max", smooth_k=7, enforce_alt=False),
    },
    "dumbbell_shoulder_press": {
        "front": dict(method="peaks", win=25, min_dist=25, thresh_pct=75, mode="max"),
        "side":  dict(method="peaks", win=25, min_dist=25, thresh_pct=75, mode="max"),
    },
    "hummer_curls": {
        "front": dict(method="peaks", win=25, min_dist=20, thresh_pct=70, mode="max"),
        "side":  dict(method="peaks", win=25, min_dist=20, thresh_pct=70, mode="max"),
    },
    "ez_bar_curls": {
        "front": dict(method="peaks", win=25, min_dist=20, thresh_pct=70, mode="max"),
        "side":  dict(method="peaks", win=25, min_dist=20, thresh_pct=70, mode="max"),
    },
    "seated_biceps_curls": {
        "front": dict(method="peaks", win=25, min_dist=20, thresh_pct=70, mode="max"),
        "side":  dict(method="peaks", win=25, min_dist=20, thresh_pct=70, mode="max"),
    },
    "overhead_triceps_extension": {
        "front": dict(method="peaks", win=25, min_dist=22, thresh_pct=70, mode="max"),
        "side":  dict(method="peaks", win=25, min_dist=22, thresh_pct=70, mode="max"),
    },
    "triceps_kickbacks": {
        "front": dict(method="peaks", win=25, min_dist=22, thresh_pct=70, mode="max"),
        "side":  dict(method="peaks", win=25, min_dist=22, thresh_pct=70, mode="max"),
    },
    "inclined_dumbbell_bench_press": {
        "front": dict(method="peaks", win=30, min_dist=25, thresh_pct=70, mode="max"),
        "side":  dict(method="peaks", win=30, min_dist=25, thresh_pct=70, mode="max"),
    },
    "rows": {
        "front": dict(method="peaks", win=25, min_dist=22, thresh_pct=70, mode="max"),
        "side":  dict(method="peaks", win=25, min_dist=22, thresh_pct=70, mode="max"),
    },
    "shrugs": {
        "front": dict(method="peaks", win=20, min_dist=18, thresh_pct=75, mode="max"),
        "side":  dict(method="peaks", win=20, min_dist=18, thresh_pct=75, mode="max"),
    },

    # LOWER BODY
    # Squats are the most common failure case for peak-based counting (partial reps, noise, edges).
    # Use threshold UP->DOWN->UP on knee angle (up is high angle).
    "weighted_squats": {
        "front": dict(method="threshold", win=38, smooth_k=11, debounce=3,
                      up_is_high=True, p_low=20, p_high=80, extreme_mode="min",
                      pad_edges=True, max_reps=40, q_drop_pct=15),
        "side":  dict(method="threshold", win=38, smooth_k=11, debounce=4,
                      up_is_high=True, p_low=20, p_high=80, extreme_mode="min",
                      pad_edges=True, max_reps=40, q_drop_pct=15),
    },
"weighted_sqauts": {
        "front": dict(method="threshold", win=38, smooth_k=11, debounce=3,
                      up_is_high=True, p_low=20, p_high=80, extreme_mode="min",
                      pad_edges=True, max_reps=40, q_drop_pct=15),
        "side":  dict(method="threshold", win=38, smooth_k=11, debounce=4,
                      up_is_high=True, p_low=20, p_high=80, extreme_mode="min",
                      pad_edges=True, max_reps=40, q_drop_pct=15),
    },
    "bulgarian_split_squat": {
        # often works with peaks; switch to threshold if needed
        "front": dict(method="peaks", win=30, min_dist=28, thresh_pct=75, mode="max"),
        "side":  dict(method="peaks", win=30, min_dist=28, thresh_pct=75, mode="max"),
    },
    "deadlift": {
        "front": dict(method="peaks", win=35, min_dist=35, thresh_pct=75, mode="max"),
        "side":  dict(method="peaks", win=35, min_dist=35, thresh_pct=75, mode="max"),
    },
    "calf_raises": {
        "front": dict(method="peaks", win=20, min_dist=18, thresh_pct=75, mode="max"),
        "side":  dict(method="peaks", win=20, min_dist=18, thresh_pct=75, mode="max"),
    },
}

def _params_for(exercise, view, win_override=None, min_dist_override=None):
    ex = norm_col(exercise)
    v = "front" if view == "front" else "side"
    base = EXERCISE_PARAMS.get(ex, {}).get(v, dict(win=25, min_dist=20, thresh_pct=75, mode="max"))
    p = dict(base)
    if win_override is not None:
        p["win"] = int(win_override)
    if min_dist_override is not None:
        p["min_dist"] = int(min_dist_override)
    return p

# =========================================================
# Main segmentation entry (used by video_to_assessment_cnn_all.py)
# Returns:
#   reps:  (R, 2*win, 9)
#   peaks: list[int]
#   signal: 1D array
# =========================================================
# =========================================================
# Main segmentation entry (used by video_to_assessment_cnn_all.py)
# Returns:
#   reps:  (R, 2*win, 9)
#   peaks: list[int]
#   signal: 1D array
#   used_params: dict (debug + exact params used)
# =========================================================
def _extract_window(feats: np.ndarray, center: int, win: int, pad_edges: bool):
    """
    feats: (N, F)
    Returns window (2*win, F), padding edges if needed.
    """
    N = feats.shape[0]
    a = int(center) - int(win)
    b = int(center) + int(win)  # exclusive
    if not pad_edges:
        if a < 0 or b > N:
            return None
        return feats[a:b]

    # pad with edge values to always return a valid window
    out = np.empty((2 * win, feats.shape[1]), dtype=np.float32)
    for i in range(2 * win):
        t = a + i
        if t < 0:
            out[i] = feats[0]
        elif t >= N:
            out[i] = feats[-1]
        else:
            out[i] = feats[t]
    return out

def _quality_score(signal: np.ndarray, center: int, win: int):
    """Local amplitude inside the window around 'center'."""
    a = max(0, int(center) - int(win))
    b = min(len(signal), int(center) + int(win))
    seg = signal[a:b]
    if len(seg) < 3:
        return 0.0
    return float(np.max(seg) - np.min(seg))

def _filter_peaks(peaks, signal, win, max_reps=40, q_drop_pct=0):
    """
    Keep chronological order, but optionally drop low-amplitude peaks and cap max reps.
    This helps when the segmenter produces a couple of false reps (won't affect grading badly).
    """
    peaks = list(map(int, peaks))
    if len(peaks) == 0:
        return [], {"n_raw": 0}

    qs = np.array([_quality_score(signal, p, win) for p in peaks], dtype=np.float32)
    order = np.argsort(-qs)  # high quality first

    keep = order
    if q_drop_pct and len(peaks) >= 5:
        k = int(np.ceil(len(peaks) * (1.0 - q_drop_pct / 100.0)))
        k = max(1, k)
        keep = keep[:k]

    if max_reps and len(keep) > max_reps:
        keep = keep[:max_reps]

    kept = sorted([peaks[i] for i in keep])
    dbg = {
        "n_raw": int(len(peaks)),
        "n_kept": int(len(kept)),
        "q_min": float(np.min(qs)) if len(qs) else None,
        "q_max": float(np.max(qs)) if len(qs) else None,
        "q_mean": float(np.mean(qs)) if len(qs) else None,
        "q_drop_pct": int(q_drop_pct),
        "max_reps": int(max_reps) if max_reps else None,
    }
    return kept, dbg

def segment_reps_from_sequence(
    exercise,
    view,
    lm_seq_xyz,
    win=25,
    min_dist=20,
    thresh_pct=None,
    **kwargs,
):
    """
    Compatible signature (older callers may pass thresh_pct etc.).
    """
    lm_arr = np.asarray(lm_seq_xyz, dtype=np.float32)  # (N,33,3)
    if lm_arr.ndim != 3 or lm_arr.shape[1] < 33 or lm_arr.shape[2] < 3:
        raise ValueError("lm_seq_xyz must be list/array of frames, each frame is 33 landmarks of (x,y,z).")

    feats = np.array([extract_9_features(lm_arr[i]) for i in range(lm_arr.shape[0])], dtype=np.float32)
    params = _params_for(exercise, view, win_override=win, min_dist_override=min_dist)

    # allow CLI override for thresh_pct (peaks mode)
    if thresh_pct is not None:
        params["thresh_pct"] = int(thresh_pct)

    signal = build_rep_signal(lm_arr, exercise, view).astype(np.float32)

    used = {"exercise": str(exercise), "view": str(view)}
    method = params.get("method", "peaks")
    used["method"] = method

    if method == "threshold":
        # try a few relaxed thresholds if needed
        p_low = float(params.get("p_low", 20))
        p_high = float(params.get("p_high", 80))
        smooth_k = int(params.get("smooth_k", 9))
        debounce = int(params.get("debounce", 3))
        up_is_high = bool(params.get("up_is_high", True))
        extreme_mode = str(params.get("extreme_mode", "min"))
        pad_edges = bool(params.get("pad_edges", True))
        max_reps = int(params.get("max_reps", 40))
        q_drop_pct = int(params.get("q_drop_pct", 0))

        # small relax sweep (keeps your logic, but avoids missing reps when range is tight)
        relax = params.get("relax", [(p_low, p_high), (max(5, p_low-5), min(95, p_high+5))])

        debug_passes = []
        peaks = []
        chosen = None
        for pl, ph in relax:
            pk, dbg = detect_reps_threshold_updownup(
                signal,
                up_is_high=up_is_high,
                p_low=float(pl),
                p_high=float(ph),
                smooth_k=smooth_k,
                debounce=debounce,
                extreme_mode=extreme_mode,
            )
            debug_passes.append({"p_low": float(pl), "p_high": float(ph), "n_peaks": int(len(pk)), "dbg": dbg})
            if len(pk) >= 1:
                peaks = pk
                chosen = (pl, ph)
                break
        used["threshold_debug_passes"] = debug_passes
        if chosen is not None:
            used["p_low_used"] = float(chosen[0])
            used["p_high_used"] = float(chosen[1])

        # filter/cap to reduce the effect of a couple false reps
        w = int(params.get("win", win))
        peaks, qdbg = _filter_peaks(peaks, signal, w, max_reps=max_reps, q_drop_pct=q_drop_pct)
        used["quality_filter"] = qdbg
        used["pad_edges"] = pad_edges
        used["win"] = int(w)

    elif method == "alt_arms":
        # alternating arms: detect peaks per-arm and merge
        w = int(params.get("win", win))
        tp = int(params.get("thresh_pct", 70))
        md = int(params.get("min_dist", min_dist))
        smooth_k = int(params.get("smooth_k", 7))
        enforce_alt = bool(params.get("enforce_alt", True))

        relL, relR = _arm_raise_rel_signals(lm_arr)
        peaks, sig_alt, dbg = detect_reps_alt_arms(
            relL, relR, thresh_pct=tp, min_dist=md, smooth_k=smooth_k, enforce_alt=enforce_alt
        )

        # use combined signal for filtering & debug preview
        signal = sig_alt.astype(np.float32)

        pad_edges = bool(params.get("pad_edges", False))
        max_reps = int(params.get("max_reps", 60))
        q_drop_pct = int(params.get("q_drop_pct", 0))
        peaks, qdbg = _filter_peaks(peaks, signal, w, max_reps=max_reps, q_drop_pct=q_drop_pct)

        used.update({
            "win": int(w),
            "mode": "max",
            "thresh_pct": int(tp),
            "min_dist": int(md),
            "smooth_k": int(smooth_k),
            "enforce_alt": bool(enforce_alt),
        })
        used["alt_arms_debug"] = dbg
        used["quality_filter"] = qdbg
        used["pad_edges"] = pad_edges

    else:
        # classic peak method
        w = int(params.get("win", win))
        mode = params.get("mode", "max")
        tp = int(params.get("thresh_pct", 75))
        md = int(params.get("min_dist", min_dist))
        peaks = detect_peaks(signal, mode=mode, thresh_pct=tp, min_dist=md)

        pad_edges = bool(params.get("pad_edges", False))
        max_reps = int(params.get("max_reps", 40))
        q_drop_pct = int(params.get("q_drop_pct", 0))
        peaks, qdbg = _filter_peaks(peaks, signal, w, max_reps=max_reps, q_drop_pct=q_drop_pct)

        used.update({"win": int(w), "mode": str(mode), "thresh_pct": int(tp), "min_dist": int(md)})
        used["quality_filter"] = qdbg
        used["pad_edges"] = pad_edges

    # build rep windows
    w = int(used["win"])
    reps = []
    kept_peaks = []
    for p in peaks:
        window = _extract_window(feats, p, w, pad_edges=bool(used.get("pad_edges", False)))
        if window is not None and window.shape[0] == 2 * w:
            reps.append(window)
            kept_peaks.append(int(p))

    reps = np.asarray(reps, dtype=np.float32) if len(reps) else np.zeros((0, 2*w, feats.shape[1]), dtype=np.float32)

    used["peaks"] = kept_peaks
    used["n_reps"] = int(reps.shape[0])
    used["signal_stats"] = {"min": float(np.min(signal)), "max": float(np.max(signal)), "mean": float(np.mean(signal))}
    return reps, kept_peaks, signal, used
