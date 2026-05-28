from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    os.chdir(root)
    os.environ["MANIFEST_PATH"] = "data/experiments/faceswap/manifest.csv"
    os.environ["FEATURES_CSV"] = "data/experiments/faceswap/module3_features.csv"
    os.environ["VIDEO_EAR_CSV"] = "data/experiments/faceswap/video_ear_scores.csv"
    os.environ["MODEL_PKL_PATH"] = "data/experiments/faceswap/ensemble_model.pkl"
    os.environ["PLOTS_DIR"] = "data/experiments/faceswap/plots"
    os.environ["VIZ_DIR"] = "data/experiments/faceswap/visualizations"
    runpy.run_path("ensemble.py", run_name="__main__")


if __name__ == "__main__":
    main()
