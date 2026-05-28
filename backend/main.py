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
CNN_FALLBACK_AUC_GATE = 0.65


def _resolve_cnn_fallback_state(
    handcrafted_auc: float,
    force_cnn_fallback: bool = False,
) -> tuple[bool, str]:
    """
    Resolve fallback activation from gate + optional force override.

    Contract: handcrafted AUC < 0.65 -> cnn_fallback.
    """
    if force_cnn_fallback:
        return True, "forced"
    if handcrafted_auc < CNN_FALLBACK_AUC_GATE:
        return True, "gated_auc"
    return False, "off"


# ---------------------------------------------------------------------------
# Lifespan: load the model bundle at startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load data/ensemble_model.pkl at startup if it exists.

    The pickle is a dict from ensemble.py: model, scaler, threshold,
    threshold_mode, feature_names.

    If the file is absent or load fails, the API falls back to equal-weights
    scoring (no trained model required).
    """
    try:
        import joblib  # optional dependency

        if MODEL_PKL.exists():
            bundle = joblib.load(MODEL_PKL)
            app.state.model = bundle.get("model")
            app.state.scaler = bundle.get("scaler")
            app.state.threshold = float(bundle.get("threshold", 0.5))
            app.state.uncertain_band = float(bundle.get("uncertain_band", 0.1))
            app.state.threshold_policy = bundle.get("threshold_policy", {})
            app.state.threshold_mode = str(
                bundle.get(
                    "threshold_mode",
                    app.state.threshold_policy.get("mode", "balanced_accuracy")
                    if isinstance(app.state.threshold_policy, dict)
                    else "balanced_accuracy",
                )
            )
            logger.info(
                "Loaded ensemble model from %s (threshold=%.4f, uncertain_band=%.4f, mode=%s)",
                MODEL_PKL,
                app.state.threshold,
                app.state.uncertain_band,
                app.state.threshold_mode,
            )
        else:
            app.state.model = None
            app.state.scaler = None
            app.state.threshold = 0.5
            app.state.uncertain_band = 0.1
            app.state.threshold_policy = {}
            app.state.threshold_mode = "default_0.5"
            logger.info(
                "No ensemble_model.pkl found at %s — using equal-weights scoring", MODEL_PKL
            )
    except ImportError:
        app.state.model = None
        app.state.scaler = None
        app.state.threshold = 0.5
        app.state.uncertain_band = 0.1
        app.state.threshold_policy = {}
        app.state.threshold_mode = "default_0.5"
        logger.info("joblib not installed — using equal-weights scoring")
    except Exception as exc:
        app.state.model = None
        app.state.scaler = None
        app.state.threshold = 0.5
        app.state.uncertain_band = 0.1
        app.state.threshold_policy = {}
        app.state.threshold_mode = "default_0.5"
        logger.warning("Model load failed (%s) — using equal-weights scoring", exc)

    handcrafted_auc = float(os.getenv("HANDCRAFTED_VALIDATION_AUC", "1.0"))
    force_cnn_fallback = os.getenv("FORCE_CNN_FALLBACK", "0").lower() in {"1", "true", "yes", "on"}
    cnn_fallback_active, cnn_fallback_reason = _resolve_cnn_fallback_state(
        handcrafted_auc=handcrafted_auc,
        force_cnn_fallback=force_cnn_fallback,
    )
    app.state.handcrafted_auc = handcrafted_auc
    app.state.force_cnn_fallback = force_cnn_fallback
    app.state.cnn_fallback_active = cnn_fallback_active
    app.state.cnn_fallback_reason = cnn_fallback_reason

    # Optional CNN inference extraction hook. Keep None if unavailable.
    cnn_infer = None
    try:
        from cnn_detector import load_cnn_infer  # type: ignore

        cnn_infer = load_cnn_infer()
    except Exception as exc:
        logger.info("CNN fallback inference unavailable: %s", exc)
    app.state.cnn_infer = cnn_infer

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

# Allow requests from configured origins (defaults to Vite dev server).
_cors_env = os.getenv("CORS_ORIGINS", "").strip()
if _cors_env:
    _cors_origins = [origin.strip() for origin in _cors_env.split(",") if origin.strip()]
else:
    _cors_origins = ["http://localhost:5173"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
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
            threshold=getattr(app.state, "threshold", 0.5),
            uncertain_band=getattr(app.state, "uncertain_band", 0.1),
            cnn_fallback_active=bool(getattr(app.state, "cnn_fallback_active", False)),
            cnn_infer=getattr(app.state, "cnn_infer", None),
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
    threshold = float(getattr(app.state, "threshold", 0.5))
    uncertain_band = float(getattr(app.state, "uncertain_band", 0.1))
    verdict_hi = min(1.0, threshold + uncertain_band)
    verdict_lo = max(0.0, threshold - uncertain_band)
    return {
        "status": "ok",
        "model_loaded": getattr(app.state, "model", None) is not None,
        "model_used_states": [
            "ensemble_learned",
            "equal_weights",
            "cnn_fallback",
            "cnn_fallback_degraded",
        ],
        "threshold": threshold,
        "threshold_mode": getattr(app.state, "threshold_mode", "default_0.5"),
        "threshold_policy": getattr(app.state, "threshold_policy", {}),
        "uncertain_band": uncertain_band,
        "verdict_hi": verdict_hi,
        "verdict_lo": verdict_lo,
        "handcrafted_auc": float(getattr(app.state, "handcrafted_auc", 1.0)),
        "cnn_fallback_auc_gate": CNN_FALLBACK_AUC_GATE,
        "cnn_fallback_active": bool(getattr(app.state, "cnn_fallback_active", False)),
        "cnn_fallback_reason": getattr(app.state, "cnn_fallback_reason", "off"),
    }
