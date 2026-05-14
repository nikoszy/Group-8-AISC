"""Download the MRL Eye Dataset from Kaggle into data/mrl/.

Uses the project's existing `download_data` helper, which wraps `kagglehub`.

Kaggle authentication:
  - Either place your `kaggle.json` at `%USERPROFILE%/.kaggle/kaggle.json`
    (Windows) or `~/.kaggle/kaggle.json` (Linux/macOS), or
  - Set the env vars KAGGLE_USERNAME and KAGGLE_KEY.

Dataset: https://www.kaggle.com/datasets/akashshingha850/mrl-eye-dataset
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from download_data import download_data  # noqa: E402


DATASET_NAME = "akashshingha850/mrl-eye-dataset"
OUT_DIR = str(PROJECT_ROOT / "data" / "mrl")


def main():
    download_data(name=DATASET_NAME, out_dir=OUT_DIR)


if __name__ == "__main__":
    main()
