"""
FastAPI entry point for the AI Virtual Coach backend.

Endpoint:
    POST /api/session/analyze
        Receives a full exercise session's pose data from the Flutter mobile
        app, runs the 4-stage ML pipeline, and returns assessment + coaching
        feedback.

Run:
    cd <project_root>
    uvicorn src.pipelines.main:app --host 0.0.0.0 --port 8000 --reload
"""

import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so ``src.*`` imports work when running
# with ``uvicorn src.pipelines.main:app`` from the project root.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Suppress noisy TF / MediaPipe logs before any TF import
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from src.pipelines.config import (
    EXERCISE_MAP,
    EXERCISE_NAME_TO_ID,
    RECOGNITION_CONFIDENCE_THRESHOLD,
)
from src.pipelines.preprocessing import preprocess_pose_sequence
from src.pipelines.recognition import recognize_exercise
from src.pipelines.assessment import assess_reps, aggregate_scores
from src.pipelines.utils import generate_fallback_feedback, preload_models

logger = logging.getLogger("ai_virtual_coach")
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")


# ============================================================================
# Pydantic request / response models — match backend_api_contract.md
# ============================================================================

class SessionMetadata(BaseModel):
    fps: float
    frame_count: int
    device: str


class SessionRequest(BaseModel):
    exercise_view: str = Field(..., description="Camera angle: 'front' or 'side'")
    pose_sequence: list[list[list[float]]] = Field(
        ..., description="frames × 33 landmarks × 4 values (x, y, z, visibility)"
    )
    metadata: SessionMetadata


class SessionResponse(BaseModel):
    exercise: str
    reps_detected: int
    scores: dict[str, float] = Field(
        ..., description="Aspect name → score (0-10 scale)"
    )
    overall_score: float
    feedback: list[str]


class ErrorResponse(BaseModel):
    error_code: str
    message: str


# ============================================================================
# App lifecycle — preload models on startup
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load recognition models at startup."""
    logger.info("Starting AI Virtual Coach backend …")
    preload_models()
    logger.info("Models loaded — server is ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="AI Virtual Coach API",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — needed for Flutter Web or browser-based testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Health-check
# ============================================================================

@app.get("/health")
async def health():
    return {"status": "ok"}


# ============================================================================
# Main inference endpoint
# ============================================================================

@app.post(
    "/api/session/analyze",
    response_model=SessionResponse,
    responses={
        400: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
def analyze_session(request: SessionRequest):
    """Full 4-stage inference pipeline: pose → features → recognition →
    assessment → coaching feedback.

    NOTE: This is a **sync** endpoint on purpose. FastAPI runs it in a
    threadpool so that the heavy TensorFlow ``model.predict()`` calls do
    not block the asyncio event loop (which caused request timeouts).
    """
    t0 = time.time()
    view = request.exercise_view

    # ── STAGE 1: Pose Preprocessing ──────────────────────────────────────
    try:
        features_50, lm_xyz = preprocess_pose_sequence(
            pose_sequence=request.pose_sequence,
            view=view,
        )
    except ValueError as exc:
        err_msg = str(exc)
        if "Too few" in err_msg or "at least" in err_msg:
            code = "INSUFFICIENT_FRAMES"
        elif "pose_sequence" in err_msg or "landmark" in err_msg:
            code = "INVALID_REQUEST"
        else:
            code = "NO_POSE_DATA"
        return JSONResponse(
            status_code=400,
            content={"error_code": code, "message": err_msg},
        )
    except Exception as exc:
        logger.exception("Stage 1 failed")
        return JSONResponse(
            status_code=500,
            content={
                "error_code": "ANALYSIS_FAILED",
                "message": f"Preprocessing error: {exc}",
            },
        )

    # ── STAGE 2: Exercise Recognition ────────────────────────────────────
    logger.info("Starting Stage 2: Exercise Recognition …")
    try:
        exercise_id, exercise_name, confidence = recognize_exercise(
            features_50=features_50,
            view=view,
        )
    except FileNotFoundError as exc:
        return JSONResponse(
            status_code=500,
            content={
                "error_code": "ANALYSIS_FAILED",
                "message": f"Recognition model not found: {exc}",
            },
        )
    except Exception as exc:
        logger.exception("Stage 2 failed")
        return JSONResponse(
            status_code=500,
            content={
                "error_code": "ANALYSIS_FAILED",
                "message": f"Recognition error: {exc}",
            },
        )

    if confidence < RECOGNITION_CONFIDENCE_THRESHOLD:
        return JSONResponse(
            status_code=422,
            content={
                "error_code": "UNRECOGNIZED_EXERCISE",
                "message": (
                    f"Could not identify the exercise (confidence {confidence:.0%}). "
                    "Please ensure your full body is visible."
                ),
            },
        )

    # ── STAGE 3: Rep Segmentation & Assessment ───────────────────────────
    logger.info("Starting Stage 3: Assessment (exercise='%s', id=%d) …", exercise_name, exercise_id)
    try:
        per_rep_scores, _debug = assess_reps(
            lm_seq_xyz=lm_xyz,
            exercise_name=exercise_name,
            exercise_id=exercise_id,
            view=view,
        )
    except RuntimeError as exc:
        err_msg = str(exc)
        if "No valid repetitions" in err_msg or "No reps" in err_msg:
            return JSONResponse(
                status_code=422,
                content={
                    "error_code": "NO_REPS_DETECTED",
                    "message": (
                        "No valid repetitions were detected. "
                        "Please try again with a clearer view."
                    ),
                },
            )
        return JSONResponse(
            status_code=500,
            content={"error_code": "ANALYSIS_FAILED", "message": err_msg},
        )
    except FileNotFoundError as exc:
        return JSONResponse(
            status_code=500,
            content={
                "error_code": "ANALYSIS_FAILED",
                "message": f"Assessment model not found: {exc}",
            },
        )
    except Exception as exc:
        logger.exception("Stage 3 failed")
        return JSONResponse(
            status_code=500,
            content={
                "error_code": "ANALYSIS_FAILED",
                "message": f"Assessment error: {exc}",
            },
        )

    aspect_means, overall_score = aggregate_scores(per_rep_scores)
    logger.info("Stage 3 complete: %d reps, overall=%.2f", len(per_rep_scores), overall_score)

    # ── STAGE 4: Coaching Agent ──────────────────────────────────────────
    feedback_lines: list[str] = []
    try:
        from src.agents import CoachingAgent

        agent = CoachingAgent()
        coaching_response = agent.generate_feedback(
            exercise_id=exercise_id,
            exercise_name=exercise_name,
            rep_scores=per_rep_scores,
            recognition_confidence=confidence,
            view_type=view,
        )

        # Use the agent's aggregated scores / overall if available
        overall_score = round(coaching_response.overall_score, 2)
        aspect_means = {
            k: round(v, 2) for k, v in coaching_response.aggregated_scores.items()
        }

        # Split the LLM feedback summary into individual lines
        raw_feedback = coaching_response.feedback_summary or ""
        feedback_lines = [
            line.strip() for line in raw_feedback.split("\n") if line.strip()
        ]

        # Append any warnings
        if coaching_response.warnings:
            feedback_lines.extend(coaching_response.warnings)

    except Exception as exc:
        logger.warning("Stage 4 (coaching agent) failed: %s — using fallback.", exc)
        feedback_lines = generate_fallback_feedback(
            per_rep_scores, aspect_means, overall_score,
        )

    elapsed = time.time() - t0
    logger.info(
        "Pipeline complete in %.2fs: exercise='%s' reps=%d overall=%.1f",
        elapsed, exercise_name, len(per_rep_scores), overall_score,
    )

    return SessionResponse(
        exercise=exercise_name,
        reps_detected=len(per_rep_scores),
        scores=aspect_means,
        overall_score=overall_score,
        feedback=feedback_lines,
    )
