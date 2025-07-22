#!/usr/bin/env python3
"""
Download pre-sampled `af_data_new` dataset from Hugging Face Hub.
Contains: cat/dog/tiger (450 each) + human_like_animal (30 images)

Usage:
    python scripts/download_af_data_new.py
"""
from pathlib import Path
from huggingface_hub import snapshot_download
import tarfile
import os
import shutil

def extract_tar_gz(tar_path, extract_dir):
    """ Extract the tar.gz file """
    print(f"Unzipping: {tar_path}")
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(extract_dir)
    print("Unzipped!")

def move_data_to_correct_location(out_dir):
    """Move expanded data to the correct location"""
    nested_dir = out_dir / "af_data_new"
    if nested_dir.exists():
        print(f"Moving data from {nested_dir} to {out_dir}")
        
        # Move each subdirectory
        for subdir in ['cat', 'dog', 'tiger', 'human_like_animal']:
            src = nested_dir / subdir
            dst = out_dir / subdir
            
            if src.exists():
                if dst.exists():
                    shutil.rmtree(dst)  # Delete existing directory
                shutil.move(str(src), str(dst))
                print(f"  Moved {subdir}/")
        
        # Delete the empty nested_dir
        try:
            nested_dir.rmdir()
            print(f"  Removed empty directory: {nested_dir}")
        except OSError:
            print(f"  Warning: Could not remove {nested_dir} (may not be empty)")

def main():
    # Fixed configuration for reproducibility
    repo_id = "futa-06/A_Computational_Classification_of_Human_Facial_Traits"  # ← Replace with your HF repo ID
    out_dir = Path("data/af_data_new")
    
    print(f"Downloading af_data_new dataset from '{repo_id}'...")
    print(f"Target directory: {out_dir}")
    
    # Create output directory
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Download dataset
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(out_dir),
        local_dir_use_symlinks=False,  # Windows compatibility
    )
    
    print("Download complete!")
    
    # Unzip the compressed file
    tar_gz_path = out_dir / "af_data_new.tar.gz"
    if tar_gz_path.exists():
        print(f"\nUnzipping the compressed file...")
        extract_tar_gz(tar_gz_path, out_dir)
        
        # Move the data to the correct location
        move_data_to_correct_location(out_dir)
        
        # Delete the compressed file
        tar_gz_path.unlink()
        print("Compressed file removed")
    else:
        print(f"Compressed file not found: {tar_gz_path}")
    
    # Verify downloaded files
    print(f"\nVerifying downloaded files...")
    total_files = 0
    for subdir in ['cat', 'dog', 'tiger', 'human_like_animal']:
        subdir_path = out_dir / subdir
        if subdir_path.exists():
            files = list(subdir_path.glob("*.jpg")) + list(subdir_path.glob("*.png"))
            print(f"  {subdir}: {len(files)} images")
            total_files += len(files)
        else:
            print(f"  {subdir}: directory not found")
    
    print(f"Total: {total_files} images downloaded")
    
    print(f"\nDataset structure:")
    print(f"  {out_dir}/")
    print(f"  ├── cat/    (450 images)")
    print(f"  ├── dog/    (450 images)")
    print(f"  ├── tiger/  (450 images)")
    print(f"  └── human_like_animal/ (30 images)")
    print(f"\nReady to run: python main_simple.py")

if __name__ == "__main__":
    main() 