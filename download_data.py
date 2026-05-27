import os
import shutil
import kagglehub


def download_data(name, out_dir="./data", is_competition=False, unzip=True):
    os.makedirs(out_dir, exist_ok=True)

    if is_competition:
        # kagglehub doesn't support competitions directly — use kaggle package for those
        raise NotImplementedError("Use the 'kaggle' package for competition downloads.")
    else:
        # kagglehub auto-authenticates via ~/.kaggle/kaggle.json or env vars
        cached_path = kagglehub.dataset_download(name)
        
        # Copy from cache to your desired out_dir
        shutil.copytree(cached_path, out_dir, dirs_exist_ok=True)
        print(f"Dataset downloaded to {out_dir}")