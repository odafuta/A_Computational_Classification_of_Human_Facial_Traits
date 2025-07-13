#!/usr/bin/env python3
"""
Kaggle Animal Faces Dataset Download Script
URL: https://www.kaggle.com/datasets/andrewmvd/animal-faces

This script downloads the Animal Faces dataset from Kaggle and organizes it
into cat, dog, and wild categories for classification.

Prerequisites:
1. Install kaggle: pip install kaggle
2. Set up Kaggle API credentials:
   - Go to https://www.kaggle.com/settings
   - Create new API token
   - Place kaggle.json in ~/.kaggle/ (Linux/Mac) or %USERPROFILE%\.kaggle\ (Windows)
   - Set permissions: chmod 600 ~/.kaggle/kaggle.json
"""

import os
import kaggle
from pathlib import Path
import shutil
import zipfile

def setup_kaggle_api():
    """Setup Kaggle API credentials"""
    kaggle_dir = Path.home() / '.kaggle'
    if not kaggle_dir.exists():
        kaggle_dir.mkdir(parents=True)
        print(f"Created kaggle directory at {kaggle_dir}")
        print("Please place your kaggle.json file in this directory")
        print("Download it from https://www.kaggle.com/settings")
        return False
    
    kaggle_json = kaggle_dir / 'kaggle.json'
    if not kaggle_json.exists():
        print(f"kaggle.json not found at {kaggle_json}")
        print("Please download it from https://www.kaggle.com/settings")
        return False
    
    # Set proper permissions on Unix systems
    if os.name != 'nt':
        os.chmod(kaggle_json, 0o600)
    
    return True

def download_dataset():
    """Download the animal faces dataset from Kaggle"""
    if not setup_kaggle_api():
        return False
    
    try:
        # Download the dataset
        dataset_name = 'andrewmvd/animal-faces'
        download_path = '../data/kaggle_raw'
        
        print(f"Downloading {dataset_name}...")
        kaggle.api.dataset_download_files(dataset_name, path=download_path, unzip=True)
        
        print("Dataset downloaded successfully!")
        return True
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return False

def organize_dataset():
    """Organize the downloaded dataset into cat, dog, wild categories"""
    kaggle_data_path = Path('../data/kaggle_raw')
    new_data_path = Path('../data/af_data_new')
    
    if not kaggle_data_path.exists():
        print("Kaggle data directory not found. Please download the dataset first.")
        return False
    
    # Create mapping for animal categories
    # Based on the dataset structure, we need to identify which animals go to which category
    category_mapping = {
        'cat': ['cat'],
        'dog': ['dog'],
        'wild': ['wild']  # This might include other wild animals in the dataset
    }
    
    # Process each category
    for category, animals in category_mapping.items():
        category_path = new_data_path / category
        category_path.mkdir(parents=True, exist_ok=True)
        
        for animal in animals:
            animal_path = kaggle_data_path / animal
            if animal_path.exists():
                # Copy all images from the animal directory
                for image_file in animal_path.glob('*'):
                    if image_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                        shutil.copy2(image_file, category_path / image_file.name)
                        print(f"Copied {image_file.name} to {category}")
    
    print("Dataset organization completed!")
    return True

def main():
    """Main function to download and organize the dataset"""
    print("=== Kaggle Animal Faces Dataset Downloader ===")
    print()
    
    # Check if dataset already exists
    if Path('../data/af_data_new/cat').exists() and any(Path('../data/af_data_new/cat').glob('*')):
        response = input("Dataset already exists. Re-download? (y/n): ")
        if response.lower() != 'y':
            print("Skipping download.")
            return
    
    # Download dataset
    if download_dataset():
        # Organize dataset
        organize_dataset()
        
        print("\n=== Dataset Setup Complete ===")
        print("Dataset structure:")
        print("data/af_data_new/")
        print("├── cat/")
        print("├── dog/")
        print("├── wild/")
        print("└── human_like_animal/")
        print()
        print("Note: The human_like_animal directory will be populated with AI-generated images.")
    else:
        print("Failed to download dataset. Please check your Kaggle API setup.")

if __name__ == "__main__":
    main() 