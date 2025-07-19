#!/usr/bin/env python3
"""
Download pre-sampled `af_data_new` dataset (cat/dog/tiger 450 each + human_like_animal 30)
from Hugging Face Hub.

Usage (default repo id is placeholder, replace with your own):
    python scripts/download_af_data_new.py \
        --repo yourname/facial_traits_af_data_new \
        --out-dir data/af_data_new
"""
from pathlib import Path
import argparse
from huggingface_hub import snapshot_download

def main():
    parser = argparse.ArgumentParser(description="Download af_data_new dataset from HF Hub")
    parser.add_argument("--repo", default="yourname/facial_traits_af_data_new",
                        help="Hugging Face Hub dataset repo id")
    parser.add_argument("--out-dir", default="data/af_data_new",
                        help="Local directory to place the dataset")
    parser.add_argument("--no-symlinks", action="store_true",
                        help="Disable symlinks (Windows recommended)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading dataset '{args.repo}' to {out_dir} …")
    snapshot_download(
        repo_id=args.repo,
        repo_type="dataset",
        local_dir=str(out_dir),
        local_dir_use_symlinks=not args.no_symlinks,
    )
    print("Download complete.")

if __name__ == "__main__":
    main() 