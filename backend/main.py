"""
backend/main.py — FastAPI application for the deepfake detector.

Run from the repo root:
    uvicorn backend.main:app --reload --port 8000

Or from the backend/ directory:
    uvicorn main:app --reload --port 8000

Endpoints:
    POST /analyze   — accepts multipart video, returns AnalysisResponse JSON
    GET  /health    — liveness check
"""

from __future__ import annotations

import logging
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from models import AnalysisResponse
from detector import analyze_video

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pre-trained model path (optional — falls back gracefully if absent)
# ---------------------------------------------------------------------------
# When running `uvicorn main:app` from backend/, the repo root is one level up.
_BACKEND_DIR = Path(__file__).parent.resolve()
_REPO_ROOT = _BACKEND_DIR.parent
MODEL_PKL = _REPO_ROOT / "data" / "ensemble_model.pkl"


# ---------------------------------------------------------------------------
# Lifespan: load the model bundle at startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load data/ensemble_model.pkl at startup if it exists.

    The pickle is expected to be a dict: {"model": LogisticRegression, "scaler": StandardScaler}
    produced by running ensemble.py and saving the result with joblib.dump().

    If the file is absent or load fails, the API falls back to equal-weights
    scoring (no trained model required).
    """
    try:
        import joblib  # optional dependency

        if MODEL_PKL.exists():
            bundle = joblib.load(MODEL_PKL)
            app.state.model = bundle.get("model")
            app.state.scaler = bundle.get("scaler")
            logger.info("Loaded ensemble model from %s", MODEL_PKL)
        else:
            app.state.model = None
            app.state.scaler = None
            logger.info(
                "No ensemble_model.pkl found at %s — using equal-weights scoring", MODEL_PKL
            )
    except ImportError:
        app.state.model = None
        app.state.scaler = None
        logger.info("joblib not installed — using equal-weights scoring")
    except Exception as exc:
        app.state.model = None
        app.state.scaler = None
        logger.warning("Model load failed (%s) — using equal-weights scoring", exc)

    yield  # application runs here

    # Cleanup (nothing to release for sklearn models)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Deepfake Detector API",
    description=(
        "Analyze a video file for deepfake signals using handcrafted features "
        "(JPEG artifact score, FFT spectral slope, Laplacian texture sharpness)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Allow requests from the Vite dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/analyze", response_model=AnalysisResponse, summary="Analyze a video for deepfakes")
async def analyze(
    video: UploadFile = File(..., description="Video file (mp4, avi, mov, webm, etc.)"),
    n_frames: int = Form(
        default=12,
        ge=1,
        le=60,
        description="Number of frames to sample from the video (1–60, default 12)",
    ),
):
    """
    Analyze a video for deepfake signals.

    Samples n_frames evenly across the video, detects faces in each frame,
    computes artifact/FFT/texture scores per face, and returns a verdict.

    The response includes per-frame probabilities and base64-encoded face crops.
    """
    original_name = video.filename or "upload.mp4"
    suffix = Path(original_name).suffix or ".mp4"

    # Windows safety: NamedTemporaryFile with delete=False so cv2 can open it
    # (Windows holds the file open while the context manager is active,
    #  preventing cv2.VideoCapture from opening it).
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = tmp.name
            content = await video.read()
            if not content:
                raise HTTPException(status_code=422, detail="Uploaded file is empty.")
            tmp.write(content)
        # File is now closed; cv2 can open it

        result = analyze_video(
            video_path=tmp_path,
            model=app.state.model,
            scaler=app.state.scaler,
            n_frames=n_frames,
        )
        result["video_name"] = original_name
        return result

    except HTTPException:
        raise
    except ValueError as exc:
        # No faces detected, or other user-correctable error
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except IOError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Could not open video file: {exc}",
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected error during analysis of %s", original_name)
        raise HTTPException(
            status_code=500, detail=f"Analysis failed: {exc}"
        ) from exc
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError as exc:
                logger.warning("Could not delete temp file %s: %s", tmp_path, exc)


@app.get("/health", summary="Liveness check")
async def health():
    """Returns API status and whether a pre-trained model is loaded."""
    return {
        "status": "ok",
        "model_loaded": getattr(app.state, "model", None) is not None,
    }
