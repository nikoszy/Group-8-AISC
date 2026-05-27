"""
backend/main.py — FastAPI application for the deepfake detector.

Run from the repo root:
    uvicorn backend.main:app --reload --port 8000

Or from the backend/ directory:
    uvicorn main:app --reload --port 8000

Endpoints:
    POST /analyze          — accepts multipart video, returns AnalysisResponse JSON
    GET  /health           — liveness check (includes active model info)
    GET  /models           — full model registry (sorted by F1)
    POST /models/reload    — dev-only: re-read registry + reload active model
"""

from __future__ import annotations

import logging
import os
import pickle
import sys
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# Support both invocation forms:
#   uvicorn backend.main:app   (from repo root  — relative import works)
#   uvicorn main:app           (from backend/   — absolute import works)
try:
    from .models import AnalysisResponse
    from .detector import analyze_video
except ImportError:
    from models import AnalysisResponse   # type: ignore[no-redef]
    from detector import analyze_video    # type: ignore[no-redef]

# Ensure repo root is on sys.path for model_registry import
_BACKEND_DIR = Path(__file__).parent.resolve()
_REPO_ROOT = _BACKEND_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_REGISTRY_PATH = _REPO_ROOT / "artifacts" / "model_registry.json"
_LEGACY_PKL    = _REPO_ROOT / "data" / "ensemble_model.pkl"
_MRL_CKPT_PATH = _REPO_ROOT / "data" / "best_model.pth"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_model_from_path(artifact_path: str) -> tuple:
    """
    Load a model bundle from a .pkl file.
    Returns (model, scaler) or (None, None) on failure.
    """
    p = Path(artifact_path)
    if not p.is_absolute():
        p = _REPO_ROOT / p
    if not p.exists():
        logger.warning("Artifact not found: %s", p)
        return None, None
    try:
        with open(p, "rb") as fh:
            bundle = pickle.load(fh)
        return bundle.get("model"), bundle.get("scaler")
    except Exception as exc:
        logger.warning("Failed to load model bundle from %s: %s", p, exc)
        return None, None


def _load_mrl_model():
    """
    Try to load the MRL MobileNetV2 checkpoint.

    Returns (model, img_size, idx_to_label, device) or (None, 84, {}, None).
    """
    try:
        from src.mrl.inference import load_model, resolve_device  # type: ignore[import]
        device = resolve_device(None)
        model, img_size, idx_to_label = load_model(_MRL_CKPT_PATH, device=device)
        logger.info("MRL model loaded from %s (img_size=%d)", _MRL_CKPT_PATH, img_size)
        return model, img_size, idx_to_label, device
    except FileNotFoundError:
        logger.info(
            "MRL checkpoint not found at %s — ear_score will be 0.5", _MRL_CKPT_PATH
        )
    except ImportError as exc:
        logger.info("MRL import failed (%s) — ear_score will be 0.5", exc)
    except Exception as exc:
        logger.warning("MRL model load failed: %s — ear_score will be 0.5", exc)
    return None, 84, {}, None


def _build_app_state_from_registry() -> dict:
    """
    Read the model registry and return app_state dict with active model info.

    Behaviour:
    - No registry file → fall back silently to legacy data/ensemble_model.pkl
    - Registry exists, no active model → raise RuntimeError (operator error)
    - Registry exists, active model found → load that model bundle
    """
    try:
        from src.model_registry import ModelRegistry  # type: ignore[import]
    except ImportError:
        logger.warning("ModelRegistry not importable — falling back to legacy pkl")
        return _legacy_fallback_state()

    registry = ModelRegistry(registry_path=_REGISTRY_PATH)

    if not _REGISTRY_PATH.exists():
        logger.info(
            "No model registry found at %s — falling back to %s",
            _REGISTRY_PATH, _LEGACY_PKL,
        )
        return _legacy_fallback_state()

    active = registry.get_active()
    if active is None:
        # Registry exists but no model is active — this is a developer error.
        raise RuntimeError(
            f"Model registry exists at {_REGISTRY_PATH} but NO active model is set.\n"
            "Fix: run  python ensemble.py  to train and register a model.\n"
            "The registry will automatically mark the best-F1 model as active."
        )

    model, scaler = _load_model_from_path(active["artifact_path"])
    if model is None:
        logger.warning(
            "Active model artifact missing (%s) — falling back to legacy pkl",
            active["artifact_path"],
        )
        return _legacy_fallback_state()

    logger.info(
        "Loaded active model from registry: %s  (type=%s, F1=%s)",
        active["model_id"],
        active["model_type"],
        active["metrics"].get("f1"),
    )
    return {
        "model":              model,
        "scaler":             scaler,
        "active_model_id":   active["model_id"],
        "active_model_type": active["model_type"],
        "active_model_f1":   active["metrics"].get("f1"),
        "registry":          registry,
    }


def _legacy_fallback_state() -> dict:
    """Fall back to data/ensemble_model.pkl (no registry)."""
    model, scaler = _load_model_from_path(str(_LEGACY_PKL))
    if model is not None:
        logger.info("Loaded legacy model bundle from %s", _LEGACY_PKL)
    else:
        logger.info("No model bundle found — using equal-weights scoring")
    return {
        "model":              model,
        "scaler":             scaler,
        "active_model_id":   "legacy_ensemble" if model is not None else "equal_weights",
        "active_model_type": "lr"              if model is not None else "equal_weights",
        "active_model_f1":   None,
        "registry":          None,
    }


# ---------------------------------------------------------------------------
# Lifespan: load models at startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load the active model (from registry or legacy path) and the optional
    MRL blink-detection model at startup.

    Fails loudly if the registry exists but has no active model — this
    indicates a developer forgot to run ensemble.py after setting up the
    registry system.
    """
    # 1. Load LR / stacked model from registry (or legacy fallback)
    state = _build_app_state_from_registry()
    app.state.model              = state["model"]
    app.state.scaler             = state["scaler"]
    app.state.active_model_id   = state["active_model_id"]
    app.state.active_model_type = state["active_model_type"]
    app.state.active_model_f1   = state["active_model_f1"]
    app.state.registry          = state["registry"]

    # 2. Load MRL model (soft dependency — falls back to ear_score=0.5)
    mrl_model, mrl_img_size, mrl_idx_to_label, mrl_device = _load_mrl_model()
    app.state.mrl_model         = mrl_model
    app.state.mrl_img_size      = mrl_img_size
    app.state.mrl_idx_to_label  = mrl_idx_to_label
    app.state.mrl_device        = mrl_device

    yield  # application runs here


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Deepfake Detector API",
    description=(
        "Analyze a video file for deepfake signals using handcrafted features "
        "(JPEG artifact score, FFT spectral slope, Laplacian texture sharpness, "
        "MRL blink-rate ear_score), automatically serving the best-F1 model from "
        "the model registry."
    ),
    version="2.0.0",
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
    request: Request,
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
    computes artifact/FFT/texture/blink scores per face, and returns a verdict.

    The response includes per-frame probabilities, base64-encoded face crops,
    and the registry model_id + model_f1 of the model that produced the result.
    """
    original_name = video.filename or "upload.mp4"
    suffix = Path(original_name).suffix or ".mp4"

    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = tmp.name
            content = await video.read()
            if not content:
                raise HTTPException(status_code=422, detail="Uploaded file is empty.")
            tmp.write(content)

        app_state = {
            "active_model_id":   getattr(request.app.state, "active_model_id",   "unknown"),
            "active_model_type": getattr(request.app.state, "active_model_type", "equal_weights"),
            "active_model_f1":   getattr(request.app.state, "active_model_f1",   None),
        }

        result = analyze_video(
            video_path=tmp_path,
            model=request.app.state.model,
            scaler=request.app.state.scaler,
            n_frames=n_frames,
            app_state=app_state,
            mrl_model=getattr(request.app.state, "mrl_model",        None),
            mrl_img_size=getattr(request.app.state, "mrl_img_size",  84),
            mrl_idx_to_label=getattr(request.app.state, "mrl_idx_to_label", {}),
            mrl_device=getattr(request.app.state, "mrl_device",      None),
        )
        result["video_name"] = original_name
        return result

    except HTTPException:
        raise
    except ValueError as exc:
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
async def health(request: Request):
    """Returns API status, active model info, and whether MRL is loaded."""
    return {
        "status":            "ok",
        "model_loaded":      getattr(request.app.state, "model", None) is not None,
        "active_model_id":   getattr(request.app.state, "active_model_id",   "unknown"),
        "active_model_type": getattr(request.app.state, "active_model_type", "equal_weights"),
        "active_model_f1":   getattr(request.app.state, "active_model_f1",   None),
        "mrl_loaded":        getattr(request.app.state, "mrl_model",         None) is not None,
    }


@app.get("/models", summary="List all registered models")
async def list_models(request: Request):
    """
    Return the full model registry, sorted by F1 descending.

    Use this to see which models have been trained, their metrics, and which
    one is currently serving predictions.
    """
    registry = getattr(request.app.state, "registry", None)
    if registry is None:
        # Try to load fresh in case registry was created after startup
        try:
            from src.model_registry import ModelRegistry  # type: ignore[import]
            registry = ModelRegistry(registry_path=_REGISTRY_PATH)
        except Exception:
            return {"models": [], "active_model_id": None,
                    "note": "Registry not available"}

    all_models = registry.list_all(metric="f1")
    active = registry.get_active()
    return {
        "models":          all_models,
        "active_model_id": active["model_id"] if active else None,
        "total":           len(all_models),
    }


@app.post("/models/reload", summary="[Dev] Reload active model from registry")
async def reload_models(request: Request):
    """
    Dev-only: re-read the model registry and reload the active model without
    restarting the server. Useful during iterative training on demo day.

    WARNING: This replaces the in-memory model; any in-flight requests will
    use the old model until they complete.
    """
    try:
        state = _build_app_state_from_registry()
        request.app.state.model              = state["model"]
        request.app.state.scaler             = state["scaler"]
        request.app.state.active_model_id   = state["active_model_id"]
        request.app.state.active_model_type = state["active_model_type"]
        request.app.state.active_model_f1   = state["active_model_f1"]
        request.app.state.registry          = state["registry"]

        logger.info(
            "Hot-reloaded model: %s (F1=%s)",
            state["active_model_id"], state["active_model_f1"]
        )
        return {
            "reloaded":          True,
            "active_model_id":   state["active_model_id"],
            "active_model_type": state["active_model_type"],
            "active_model_f1":   state["active_model_f1"],
        }
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Reload failed")
        raise HTTPException(status_code=500, detail=f"Reload failed: {exc}") from exc
