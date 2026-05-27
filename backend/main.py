"""
backend/main.py — FastAPI application for the deepfake detector.

Run from the repo root:
    uvicorn backend.main:app --reload --port 8000

Or from the backend/ directory:
    uvicorn main:app --reload --port 8000

Endpoints:
    POST /predict          — contract-shape: file=, returns 5-tier verdict
    POST /analyze          — richer shape: video=, returns AnalysisResponse JSON
    GET  /health           — liveness check (includes CNN + active model info)
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

import numpy as np
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
_REGISTRY_PATH  = _REPO_ROOT / "artifacts" / "model_registry.json"
_LEGACY_PKL     = _REPO_ROOT / "data" / "ensemble_model.pkl"
_MRL_CKPT_PATH  = _REPO_ROOT / "data" / "best_model.pth"
_CNN_CKPT_PATH  = _REPO_ROOT / "data" / "cnn_model.pth"
_STACK_PKL_PATH = _REPO_ROOT / "data" / "stacking_bundle.pkl"

# 5-tier verdict thresholds matching the API contract
_VERDICT_BANDS = [
    (0.00, 0.20, "VERY LIKELY REAL"),
    (0.20, 0.40, "LIKELY REAL"),
    (0.40, 0.60, "UNCERTAIN"),
    (0.60, 0.80, "LIKELY FAKE"),
    (0.80, 1.01, "VERY LIKELY FAKE"),
]


def _five_tier_verdict(prob: float) -> str:
    for lo, hi, label in _VERDICT_BANDS:
        if lo <= prob < hi:
            return label
    return "VERY LIKELY FAKE"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_artifact(artifact_path: str) -> Path:
    p = Path(artifact_path)
    return p if p.is_absolute() else _REPO_ROOT / p


def _load_model_from_path(artifact_path: str) -> tuple:
    """
    Load a model bundle from a .pkl file.
    Returns (model, scaler) or (None, None) on failure.
    """
    p = _resolve_artifact(artifact_path)
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


def _load_alpha_from_stacking_artifact(artifact_path: str, fallback: float = 0.65) -> float:
    """
    Read the alpha blend weight stored inside a stacking bundle .pkl.
    Returns fallback if the file is absent or the key is missing.
    """
    p = _resolve_artifact(artifact_path)
    if not p.exists():
        return fallback
    try:
        with open(p, "rb") as fh:
            bundle = pickle.load(fh)
        return float(bundle.get("alpha", fallback))
    except Exception as exc:
        logger.warning("Could not read alpha from stacking artifact %s: %s", p, exc)
        return fallback


def _load_cnn_model():
    """
    Load EfficientNet-B0 from the checkpoint.

    Strategy:
    - If torch is NOT installed or checkpoint does NOT exist → soft skip (CNN disabled).
    - If both prerequisites ARE present → must succeed; raises RuntimeError on failure
      so the server refuses to start with a silently broken CNN.

    Returns the loaded model (eval mode) or None.
    Logs exactly one clear line: 'CNN loaded successfully …' or the failure reason.
    """
    try:
        from src.cnn_runner import _check_torch, _build_architecture  # noqa: E402
    except ImportError as exc:
        logger.warning("CNN disabled — cnn_runner not importable: %s", exc)
        return None

    torch_ok = _check_torch()
    ckpt_exists = _CNN_CKPT_PATH.exists()

    if not torch_ok:
        logger.warning("CNN disabled — PyTorch not installed in this environment")
        return None
    if not ckpt_exists:
        logger.info(
            "CNN disabled — checkpoint not found at %s  "
            "(train with python cnn_detector.py)",
            _CNN_CKPT_PATH,
        )
        return None

    # Both prerequisites met — failure is a startup error
    try:
        import torch
        arch = _build_architecture()
        state = torch.load(str(_CNN_CKPT_PATH), map_location="cpu", weights_only=True)
        arch.load_state_dict(state)
        arch.eval()
        logger.info(
            "CNN loaded successfully from %s on device cpu", _CNN_CKPT_PATH
        )
        return arch
    except Exception as exc:
        logger.critical(
            "CNN load FAILED — checkpoint exists at %s but could not be loaded: %s",
            _CNN_CKPT_PATH, exc, exc_info=True,
        )
        raise RuntimeError(
            f"CNN checkpoint exists at {_CNN_CKPT_PATH} but failed to load.\n"
            f"Likely cause: architecture mismatch or corrupt file.\n"
            f"Error: {exc}"
        ) from exc


def _load_cnn_alpha(fallback: float = 0.65) -> float:
    """
    Return the CNN blend weight from data/stacking_bundle.pkl.

    Falls back to `fallback` if the bundle is absent or alpha_reliable=False.
    """
    if not _STACK_PKL_PATH.exists():
        logger.info(
            "Stacking bundle not found at %s — using fallback CNN alpha=%.2f",
            _STACK_PKL_PATH, fallback,
        )
        return fallback
    try:
        with open(_STACK_PKL_PATH, "rb") as fh:
            sb = pickle.load(fh)
        if sb.get("alpha_reliable", False):
            alpha = float(sb["alpha"])
            logger.info(
                "CNN alpha from stacking bundle: %.2f  "
                "(combined AUC=%.4f)",
                alpha, sb.get("combined_auc", float("nan")),
            )
            return alpha
        logger.info(
            "Stacking bundle present but alpha_reliable=False — "
            "using fallback CNN alpha=%.2f", fallback,
        )
        return fallback
    except Exception as exc:
        logger.warning("Could not load stacking bundle (%s) — using alpha=%.2f", exc, fallback)
        return fallback


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
    cnn_alpha_override: float | None = None

    if model is None:
        if active.get("model_type") == "stacked":
            # Stacking bundles store CNN/LR blend weights, not a sklearn model.
            # Load the LR base from the best comparable registry entry instead.
            best_lr = registry.get_best(metric="f1")
            if best_lr and best_lr["model_id"] != active["model_id"]:
                model, scaler = _load_model_from_path(best_lr["artifact_path"])
                lr_source = best_lr["model_id"]
            else:
                model, scaler = _load_model_from_path(str(_LEGACY_PKL))
                lr_source = "legacy_ensemble"

            cnn_alpha_override = _load_alpha_from_stacking_artifact(active["artifact_path"])
            logger.info(
                "Stacked model active: LR base loaded from '%s', CNN alpha=%.2f",
                lr_source, cnn_alpha_override,
            )
            if model is None:
                logger.warning("Stacked model: LR base also unavailable — equal-weights fallback")
        else:
            logger.warning(
                "Active model artifact missing (%s) — falling back to legacy pkl",
                active["artifact_path"],
            )
            return _legacy_fallback_state()

    if model is not None:
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
        "cnn_alpha":         cnn_alpha_override,  # None → use _load_cnn_alpha() default
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
        "cnn_alpha":         None,  # use _load_cnn_alpha() default
    }


# ---------------------------------------------------------------------------
# Lifespan: load models at startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load all models at startup.

    Order:
    1. LR / stacked model from registry (or legacy fallback).
    2. CNN (EfficientNet-B0) — hard failure if checkpoint + torch present but load fails.
    3. MRL blink-detection model (soft dependency).

    Fails loudly if:
    - The registry exists but has no active model (run ensemble.py to fix).
    - The CNN checkpoint + torch are both present but the model fails to load
      (indicates a corrupt file or architecture mismatch).
    """
    # 1. Load LR / stacked model from registry (or legacy fallback)
    state = _build_app_state_from_registry()
    app.state.model              = state["model"]
    app.state.scaler             = state["scaler"]
    app.state.active_model_id   = state["active_model_id"]
    app.state.active_model_type = state["active_model_type"]
    app.state.active_model_f1   = state["active_model_f1"]
    app.state.registry          = state["registry"]

    # 2. Load CNN (raises RuntimeError on unexpected failure)
    cnn_model = _load_cnn_model()
    app.state.cnn_model = cnn_model
    # Use alpha from stacking bundle embedded in registry entry if present,
    # otherwise fall back to data/stacking_bundle.pkl or hardcoded 0.65.
    app.state.cnn_alpha = (
        state["cnn_alpha"] if state.get("cnn_alpha") is not None
        else _load_cnn_alpha()
    )

    # 3. Load MRL model (soft dependency — falls back to ear_score=0.5)
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
        "Analyze a video file for deepfake signals. "
        "Uses handcrafted features (JPEG artifact, FFT, Laplacian texture, MRL blink), "
        "EfficientNet-B0 CNN, and the best-F1 model from the registry."
    ),
    version="2.1.0",
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
# Shared upload helper
# ---------------------------------------------------------------------------

async def _save_upload(upload: UploadFile) -> tuple[str, str]:
    """Write UploadFile to a temp file; return (tmp_path, original_name)."""
    original_name = upload.filename or "upload.mp4"
    suffix = Path(original_name).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = tmp.name
        content = await upload.read()
        if not content:
            raise HTTPException(status_code=422, detail="Uploaded file is empty.")
        tmp.write(content)
    return tmp_path, original_name


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/predict", summary="Detect deepfakes (contract shape)")
async def predict(
    request: Request,
    file: UploadFile = File(..., description="Video file (.mp4, .avi, .mov, .webm)"),
    frames: int = Form(
        default=16,
        ge=1,
        le=60,
        description="Frames to sample (1–60, default 16)",
    ),
    min_quality: float = Form(
        default=0.20,
        ge=0.0,
        le=1.0,
        description="Minimum face quality threshold (0–1, default 0.20)",
    ),
):
    """
    Analyze a video for deepfake signals.

    Returns the API-contract-defined JSON shape with a 5-tier verdict,
    per-module scores, and a per-frame breakdown.
    """
    tmp_path: str | None = None
    try:
        tmp_path, original_name = await _save_upload(file)

        app_state = {
            "active_model_id":   getattr(request.app.state, "active_model_id",   "unknown"),
            "active_model_type": getattr(request.app.state, "active_model_type", "equal_weights"),
            "active_model_f1":   getattr(request.app.state, "active_model_f1",   None),
        }

        result = analyze_video(
            video_path=tmp_path,
            model=request.app.state.model,
            scaler=request.app.state.scaler,
            n_frames=frames,
            app_state=app_state,
            mrl_model=getattr(request.app.state, "mrl_model",        None),
            mrl_img_size=getattr(request.app.state, "mrl_img_size",  84),
            mrl_idx_to_label=getattr(request.app.state, "mrl_idx_to_label", {}),
            mrl_device=getattr(request.app.state, "mrl_device",      None),
            cnn_model=getattr(request.app.state, "cnn_model",        None),
            cnn_alpha=getattr(request.app.state, "cnn_alpha",        0.65),
        )

        # Map to contract shape
        combined_score = result["quality_weighted_prob_fake"]
        verdict = _five_tier_verdict(combined_score)

        detected = [f for f in result["frames"] if f["face_detected"]]
        per_frame = [
            {
                "frame":   f["frame_index"],
                "cnn":     f.get("cnn_prob"),
                "lr":      f.get("lr_prob"),
                "quality": f.get("laplacian_score"),
            }
            for f in detected
        ]

        degraded_reasons = []
        if not result["cnn_active"]:
            degraded_reasons.append("CNN not active — LR-only scoring")
        if result["warnings"]:
            degraded_reasons.extend(result["warnings"])

        return {
            "verdict":        verdict,
            "confidence":     result["confidence"],
            "combined_score": combined_score,
            "frame_count":    result["frames_sampled"],
            "face_frames":    result["frames_analyzed"],
            "module_scores":  result["module_scores"],
            "per_frame":      per_frame,
            "model_id":       result["model_id"],
            "cnn_active":     result["cnn_active"],
            "degraded_reason": "; ".join(degraded_reasons) if degraded_reasons else None,
        }

    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except IOError as exc:
        raise HTTPException(status_code=422, detail=f"Could not open video: {exc}") from exc
    except Exception as exc:
        logger.exception("Unexpected error during /predict of %s", file.filename)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError as exc:
                logger.warning("Could not delete temp file %s: %s", tmp_path, exc)


@app.post("/analyze", response_model=AnalysisResponse, summary="Analyze a video (rich shape)")
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
    Analyze a video for deepfake signals (richer response than /predict).

    Returns the full AnalysisResponse with per-frame face crops and all
    intermediate scores. Used by the Streamlit/React frontend.
    """
    tmp_path: str | None = None
    try:
        tmp_path, original_name = await _save_upload(video)

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
            cnn_model=getattr(request.app.state, "cnn_model",        None),
            cnn_alpha=getattr(request.app.state, "cnn_alpha",        0.65),
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
        logger.exception("Unexpected error during analysis of %s", video.filename)
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
    """Returns API status, active model info, CNN state, and MRL state."""
    return {
        "status":            "ok",
        "model_loaded":      getattr(request.app.state, "model", None) is not None,
        "active_model_id":   getattr(request.app.state, "active_model_id",   "unknown"),
        "active_model_type": getattr(request.app.state, "active_model_type", "equal_weights"),
        "active_model_f1":   getattr(request.app.state, "active_model_f1",   None),
        "cnn_loaded":        getattr(request.app.state, "cnn_model",         None) is not None,
        "cnn_alpha":         getattr(request.app.state, "cnn_alpha",         0.65),
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
        request.app.state.cnn_alpha         = (
            state["cnn_alpha"] if state.get("cnn_alpha") is not None
            else _load_cnn_alpha()
        )

        logger.info(
            "Hot-reloaded model: %s (F1=%s, cnn_alpha=%.2f)",
            state["active_model_id"], state["active_model_f1"],
            request.app.state.cnn_alpha,
        )
        return {
            "reloaded":          True,
            "active_model_id":   state["active_model_id"],
            "active_model_type": state["active_model_type"],
            "active_model_f1":   state["active_model_f1"],
            "cnn_alpha":         request.app.state.cnn_alpha,
        }
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Reload failed")
        raise HTTPException(status_code=500, detail=f"Reload failed: {exc}") from exc
