# Backend API Contract — AI Virtual Coach

> **Purpose:** This document defines the exact API contract that the Flutter frontend
> expects from the FastAPI backend. Hand this to the backend coding agent before
> implementation begins.
>
> **Generated:** February 8, 2026 — based on the finalized Flutter frontend code.

---

## Architecture Overview

```
┌──────────────────────┐         POST /api/session/analyze         ┌──────────────────────┐
│   Flutter Frontend   │ ──────────────────────────────────────▶   │   FastAPI Backend     │
│   (Android phone)    │   JSON body (application/json)            │   (Python + uvicorn)  │
│                      │ ◀──────────────────────────────────────   │                       │
│                      │   200 JSON  or  4xx/5xx JSON error        │                       │
└──────────────────────┘                                           └──────────────────────┘
```

- **Communication:** Standard HTTP REST (no WebSocket, no streaming, no auth)
- **Data format:** JSON with `Content-Type: application/json`
- **Key naming convention:** `snake_case` (Pydantic default — matches Flutter's `@JsonKey` annotations)
- **Processing model:** Synchronous — the backend processes the request and returns the result in the same HTTP response
- **Client timeout:** 120 seconds — the Flutter app will abort and show an error if the backend takes longer

---

## Endpoint

### `POST /api/session/analyze`

Receives a full exercise session's pose data, runs ML inference, and returns
an assessment with scores and coaching feedback.

---

## Request

### Headers

| Header         | Value              |
| -------------- | ------------------ |
| `Content-Type` | `application/json` |

### Body Schema

```json
{
  "exercise_view": "front" | "side",
  "pose_sequence": [
    // Array of frames (one per detected frame during the session)
    [
      // Each frame contains exactly 33 landmarks (MediaPipe Pose)
      [x, y, z, visibility],   // Landmark 0  — nose
      [x, y, z, visibility],   // Landmark 1  — left eye (inner)
      [x, y, z, visibility],   // Landmark 2  — left eye
      // ... (33 landmarks total per frame)
      [x, y, z, visibility]    // Landmark 32 — right foot index
    ],
    // ... more frames
  ],
  "metadata": {
    "fps": 24.57,
    "frame_count": 734,
    "device": "mobile"
  }
}
```

### Field Details

| Field | Type | Required | Description |
| ----- | ---- | -------- | ----------- |
| `exercise_view` | `string` | ✅ | Camera angle: `"front"` or `"side"`. Indicates which side the user was filmed from. |
| `pose_sequence` | `float[][][]` | ✅ | 3D array: `frames × 33 landmarks × 4 values`. See landmark format below. |
| `metadata.fps` | `float` | ✅ | Actual detection FPS computed from timestamps (not a constant 30). Typically 15–30 depending on device. |
| `metadata.frame_count` | `int` | ✅ | Total number of frames where pose detection succeeded. Should equal `len(pose_sequence)`. |
| `metadata.device` | `string` | ✅ | Always `"mobile"` for Phase 1. |

### Landmark Format — 4 Values per Landmark

Each landmark is an array of **4 floats**: `[x, y, z, visibility]`

| Index | Field | Range | Description |
| ----- | ----- | ----- | ----------- |
| 0 | `x` | 0.0 – 1.0 | Horizontal position, normalized to image width |
| 1 | `y` | 0.0 – 1.0 | Vertical position, normalized to image height |
| 2 | `z` | ~-1.0 – 1.0 | Depth relative to hip midpoint (smaller = closer to camera) |
| 3 | `visibility` | 0.0 – 1.0 | Confidence that the landmark is visible (not occluded). Values below ~0.3 are unreliable. |

> **Note:** The `visibility` field is included intentionally — the backend can use it
> to filter out low-confidence landmarks before analysis, improving accuracy.

### MediaPipe Pose Landmark Indices (33 total)

| Index | Landmark | Index | Landmark |
| ----- | -------- | ----- | -------- |
| 0 | Nose | 17 | Left pinky |
| 1 | Left eye (inner) | 18 | Right pinky |
| 2 | Left eye | 19 | Left index |
| 3 | Left eye (outer) | 20 | Right index |
| 4 | Right eye (inner) | 21 | Left thumb |
| 5 | Right eye | 22 | Right thumb |
| 6 | Right eye (outer) | 23 | Left hip |
| 7 | Left ear | 24 | Right hip |
| 8 | Right ear | 25 | Left knee |
| 9 | Mouth (left) | 26 | Right knee |
| 10 | Mouth (right) | 27 | Left ankle |
| 11 | Left shoulder | 28 | Right ankle |
| 12 | Right shoulder | 29 | Left heel |
| 13 | Left elbow | 30 | Right heel |
| 14 | Right elbow | 31 | Left foot index |
| 15 | Left wrist | 32 | Right foot index |
| 16 | Right wrist | | |

---

## Response — Success (200 OK)

```json
{
  "exercise": "squat",
  "reps_detected": 12,
  "scores": {
    "depth": 8.5,
    "back_angle": 7.0,
    "knees_in": 9.0,
    "tempo": 6.5,
    "stability": 8.0
  },
  "overall_score": 7.8,
  "feedback": [
    "Great depth on most reps!",
    "Watch your back angle, try to keep it more upright.",
    "Good knee stability throughout the movement."
  ]
}
```

### Field Details

| Field | Type | Description |
| ----- | ---- | ----------- |
| `exercise` | `string` | Name of the detected exercise (e.g. `"squat"`, `"push_up"`, `"lunge"`). The frontend displays this in uppercase. |
| `reps_detected` | `int` | Number of valid repetitions detected in the session. |
| `scores` | `dict[string, float]` | Key-value map of aspect names to scores. **Scores are on a 0–10 scale.** The frontend multiplies by 10 for display (0–100). Aspect names use `snake_case` — displayed as uppercase with spaces (e.g. `"back_angle"` → `"BACK ANGLE"`). |
| `overall_score` | `float` | Overall form quality score, **0–10 scale**. The frontend multiplies by 10 for display. |
| `feedback` | `list[string]` | Array of human-readable coaching tips. Displayed as a bulleted list. Can be empty if form is perfect. |

### Score Scale Convention

| API Value | Frontend Display | Meaning |
| --------- | ---------------- | ------- |
| 8.5–10.0 | 85–100 (green) | Excellent form |
| 7.0–8.4 | 70–84 (teal) | Good form |
| 5.0–6.9 | 50–69 (orange) | Needs improvement |
| 0.0–4.9 | 0–49 (red) | Poor form |

---

## Response — Error (non-200)

When the backend cannot process the session, return an appropriate HTTP status
code with a JSON error body:

```json
{
  "error_code": "NO_REPS_DETECTED",
  "message": "No valid repetitions were detected. Please try again with a clearer view."
}
```

### Field Details

| Field | Type | Description |
| ----- | ---- | ----------- |
| `error_code` | `string` | Machine-readable error identifier (used for logging/debugging). |
| `message` | `string` | Human-readable error message. **Displayed verbatim to the user** on the error screen. Write clear, helpful messages. |

### Suggested Error Codes

| HTTP Status | `error_code` | `message` (example) |
| ----------- | ------------ | ------------------- |
| 400 | `INVALID_REQUEST` | "Invalid request format. Please update the app and try again." |
| 400 | `NO_POSE_DATA` | "No pose data was received. Please try recording again." |
| 400 | `INSUFFICIENT_FRAMES` | "Too few frames to analyze. Please record for at least 10 seconds." |
| 422 | `NO_REPS_DETECTED` | "No valid repetitions were detected. Please try again with a clearer view." |
| 422 | `UNRECOGNIZED_EXERCISE` | "Could not identify the exercise. Please ensure your full body is visible." |
| 500 | `ANALYSIS_FAILED` | "An internal error occurred during analysis. Please try again." |

> **Important:** If the backend returns a non-200 status but the body is not valid JSON
> or doesn't match the `{error_code, message}` schema, the Flutter app falls back to
> displaying: *"Server error ({status_code}). Please try again."*

---

## Payload Size Estimates

The frontend logs the JSON payload size before sending. Expected sizes with
**4 values per landmark** (x, y, z, visibility):

| Session Duration | Frames (~24 fps) | Data Points | Est. JSON Size |
| ---------------- | ----------------- | ----------- | -------------- |
| 15 seconds | ~360 | 47,520 | ~550 KB |
| 30 seconds | ~720 | 95,040 | ~1.1 MB |
| 60 seconds | ~1,440 | 190,080 | ~2.2 MB |
| 120 seconds | ~2,880 | 380,160 | ~4.4 MB |

> FPS varies by device (15–30). The `metadata.fps` field carries the actual value.

---

## Recommended FastAPI Implementation

### Pydantic Models

```python
from pydantic import BaseModel

class SessionMetadata(BaseModel):
    fps: float
    frame_count: int
    device: str

class SessionRequest(BaseModel):
    exercise_view: str                          # "front" or "side"
    pose_sequence: list[list[list[float]]]      # frames × 33 × 4
    metadata: SessionMetadata

class SessionResponse(BaseModel):
    exercise: str
    reps_detected: int
    scores: dict[str, float]                    # aspect_name → 0.0–10.0
    overall_score: float                        # 0.0–10.0
    feedback: list[str]

class ErrorResponse(BaseModel):
    error_code: str
    message: str
```

### Endpoint Skeleton

```python
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="AI Virtual Coach API")

# CORS — needed if you ever test via Flutter Web or browser tools
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/session/analyze", response_model=SessionResponse)
async def analyze_session(request: SessionRequest):
    """
    Analyze a pose session and return exercise assessment.

    - Validate pose_sequence shape (each frame should have 33 landmarks × 4 values)
    - Detect exercise type from movement patterns
    - Count repetitions
    - Score form quality across multiple aspects (0–10 scale)
    - Generate coaching feedback strings
    - Return results synchronously (target < 30s processing time)
    """

    # Validate input
    for i, frame in enumerate(request.pose_sequence):
        if len(frame) != 33:
            return JSONResponse(
                status_code=400,
                content={
                    "error_code": "INVALID_REQUEST",
                    "message": f"Frame {i} has {len(frame)} landmarks, expected 33."
                }
            )
        for j, landmark in enumerate(frame):
            if len(landmark) != 4:
                return JSONResponse(
                    status_code=400,
                    content={
                        "error_code": "INVALID_REQUEST",
                        "message": f"Frame {i}, landmark {j} has {len(landmark)} values, expected 4 [x, y, z, visibility]."
                    }
                )

    if len(request.pose_sequence) < 30:  # ~1 second of data
        return JSONResponse(
            status_code=400,
            content={
                "error_code": "INSUFFICIENT_FRAMES",
                "message": "Too few frames to analyze. Please record for at least 10 seconds."
            }
        )

    # ── ML Processing goes here ──
    # 1. Convert pose_sequence to numpy array: shape (N, 33, 4)
    # 2. Optionally filter landmarks where visibility < 0.5
    # 3. Detect exercise type
    # 4. Count repetitions
    # 5. Score each aspect
    # 6. Generate feedback

    # Return results
    return SessionResponse(
        exercise="squat",
        reps_detected=12,
        scores={"depth": 8.5, "back_angle": 7.0, "knees_in": 9.0, "tempo": 6.5, "stability": 8.0},
        overall_score=7.8,
        feedback=["Great depth on most reps!", "Watch your back angle."]
    )
```

### Server Configuration

```bash
# Run with:
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# --host 0.0.0.0  → accessible from other devices on the local network
# --port 8000     → matches Flutter's _baseUrl
# --reload        → auto-restart on code changes (dev only)
```

Key settings:
- **Request body limit:** Default is fine (Starlette allows up to 10 MB by default, our max payload is ~4.4 MB)
- **Timeout:** Keep processing under 60 seconds; the Flutter client times out at 120 seconds
- **Concurrency:** Use `--limit-concurrency 5` if running on a constrained machine, to prevent overload from concurrent sessions

---

## Network Setup for Physical Android Device

Since the Flutter app runs on a physical Android phone and the FastAPI server
runs on a development machine:

1. Both devices must be on the **same Wi-Fi network**
2. Find the dev machine's local IP: run `ipconfig` on Windows → look for `IPv4 Address` (e.g. `192.168.1.42`)
3. Update the Flutter app's `_baseUrl` constant in `lib/services/api_service.dart`:
   ```dart
   const String _baseUrl = 'http://192.168.1.42:8000';
   ```
4. Make sure Windows Firewall allows inbound connections on port 8000
5. Start the FastAPI server with `--host 0.0.0.0` (not `--host 127.0.0.1`)

---

## Testing the Contract

### Quick smoke test with curl

```bash
# Minimal valid request (2 frames, 33 landmarks × 4 values each)
curl -X POST http://localhost:8000/api/session/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "exercise_view": "front",
    "pose_sequence": [
      [[0.5,0.5,0.0,0.99],[0.5,0.5,0.0,0.99],[0.5,0.5,0.0,0.99],[0.5,0.5,0.0,0.99],[0.5,0.5,0.0,0.99],[0.5,0.5,0.0,0.99],[0.5,0.5,0.0,0.99],[0.5,0.5,0.0,0.99],[0.5,0.5,0.0,0.99],[0.5,0.5,0.0,0.99],[0.5,0.5,0.0,0.99],[0.5,0.5,0.0,0.99],[0.5,0.5,0.0,0.99],[0.5,0.5,0.0,0.99],[0.5,0.5,0.0,0.99],[0.5,0.5,0.0,0.99],[0.5,0.5,0.0,0.99],[0.5,0.5,0.0,0.99],[0.5,0.5,0.0,0.99],[0.5,0.5,0.0,0.99],[0.5,0.5,0.0,0.99],[0.5,0.5,0.0,0.99],[0.5,0.5,0.0,0.99],[0.5,0.5,0.0,0.99],[0.5,0.5,0.0,0.99],[0.5,0.5,0.0,0.99],[0.5,0.5,0.0,0.99],[0.5,0.5,0.0,0.99],[0.5,0.5,0.0,0.99],[0.5,0.5,0.0,0.99],[0.5,0.5,0.0,0.99],[0.5,0.5,0.0,0.99],[0.5,0.5,0.0,0.99]]
    ],
    "metadata": {"fps": 24.5, "frame_count": 1, "device": "mobile"}
  }'
```

---

## Summary of What the Frontend Sends & Expects

| Aspect | Detail |
| ------ | ------ |
| **Method** | `POST` |
| **URL** | `http://{host}:8000/api/session/analyze` |
| **Content-Type** | `application/json` |
| **Auth** | None (Phase 1) |
| **Request body** | `SessionRequest` JSON — see schema above |
| **Landmarks per frame** | 33 (MediaPipe Pose), each with **4 values** `[x, y, z, visibility]` |
| **Success response** | 200 + `SessionResponse` JSON |
| **Error response** | 4xx/5xx + `ErrorResponse` JSON (`error_code` + `message`) |
| **Client timeout** | 120 seconds |
| **Max expected payload** | ~4.4 MB (2-minute session at 24 fps) |
