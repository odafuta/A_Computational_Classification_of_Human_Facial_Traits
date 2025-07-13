#!/usr/bin/env python3
"""
Project Setup Script for Animal Facial Classification System

This script automates the setup of the entire project including:
1. Dataset download
2. AI-generated image creation  
3. Model training
4. Jupyter notebook creation
"""

import os
import sys
import subprocess
from pathlib import Path
import argparse

def check_requirements():
    """Check if required packages are installed"""
    required_packages = [
        'numpy', 'matplotlib', 'scikit-learn', 'joblib',
        'Pillow', 'seaborn', 'pandas', 'jupyter', 'kaggle', 'requests'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("Installing missing packages...")
        subprocess.run([sys.executable, '-m', 'pip', 'install'] + missing_packages)
        print("✅ Packages installed successfully")
    else:
        print("✅ All required packages are installed")

def setup_directories():
    """Create necessary directories"""
    directories = [
        '../data',
        '../data/af_data_new',
        '../data/af_data_new/cat',
        '../data/af_data_new/dog', 
        '../data/af_data_new/wild',
        '../data/af_data_new/human_like_animal',
        '../data/kaggle_raw',
        '../models',
        '../results',
        '../notebooks'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"📁 Created directory: {directory}")

def setup_git_lfs():
    """Setup Git LFS for large files"""
    try:
        # Change to project root directory
        os.chdir('..')
        
        subprocess.run(['git', 'lfs', 'install'], check=True)
        
        # Track large files
        file_types = ['*.pkl', '*.jpg', '*.png', '*.jpeg', '*.h5', '*.model']
        for file_type in file_types:
            subprocess.run(['git', 'lfs', 'track', file_type], check=True)
        
        # Track data directories
        subprocess.run(['git', 'lfs', 'track', 'data/**'], check=True)
        subprocess.run(['git', 'lfs', 'track', 'models/**'], check=True)
        subprocess.run(['git', 'lfs', 'track', 'results/**'], check=True)
        
        print("✅ Git LFS configured successfully")
        
        # Change back to scripts directory
        os.chdir('scripts')
        
    except subprocess.CalledProcessError:
        print("⚠️  Git LFS setup failed. Please install Git LFS manually.")

def download_dataset():
    """Download Kaggle dataset"""
    print("🔄 Downloading Kaggle dataset...")
    try:
        from download_kaggle_dataset import main as download_main
        download_main()
        print("✅ Dataset downloaded successfully")
    except Exception as e:
        print(f"⚠️  Dataset download failed: {e}")
        print("Please download the dataset manually from Kaggle")

def generate_ai_images():
    """Generate AI human-animal hybrid images"""
    print("🔄 Generating AI images...")
    try:
        from generate_human_animal_images import main as generate_main
        generate_main()
        print("✅ AI images generated successfully")
    except Exception as e:
        print(f"⚠️  AI image generation failed: {e}")
        print("Some images may be placeholders")

def create_jupyter_notebook():
    """Create Jupyter notebook file"""
    print("🔄 Creating Jupyter notebook...")
    
    # The notebook content was created in the previous step
    # Move it to the notebooks directory
    src_notebook = '../animal_classification_notebook.ipynb'
    dst_notebook = '../notebooks/animal_classification_analysis.ipynb'
    
    if os.path.exists(src_notebook):
        os.rename(src_notebook, dst_notebook)
        print(f"✅ Jupyter notebook created: {dst_notebook}")
    else:
        print("⚠️  Jupyter notebook file not found")

def train_initial_model():
    """Train initial model"""
    print("🔄 Training initial model...")
    try:
        # Change to project root directory
        os.chdir('..')
        
        # Import and run the main training function
        sys.path.append('.')
        from main_updated import main as train_main
        train_main()
        print("✅ Initial model trained successfully")
        
        # Change back to scripts directory
        os.chdir('scripts')
        
    except Exception as e:
        print(f"⚠️  Model training failed: {e}")
        print("Please run the training script manually")

def create_readme():
    """Create comprehensive README"""
    readme_content = """# Animal Facial Classification System

A computational classification system for animal facial traits using PCA and SVM.

## Features

- **Multi-class Classification**: Cat, Dog, Wild, Human-like Animal
- **PCA Dimensionality Reduction**: Efficient feature extraction
- **SVM Classification**: High-accuracy classification
- **AI-Generated Images**: Human-animal hybrid images for extended analysis
- **Jupyter Notebook**: Interactive analysis and visualization
- **Git LFS**: Large file management

## Project Structure

```
A_Computational_Classification_of_Human_Facial_Traits/
├── data/                           # Dataset directory
│   ├── af_data_new/               # New dataset
│   │   ├── cat/                   # Cat images
│   │   ├── dog/                   # Dog images
│   │   ├── wild/                  # Wild animal images
│   │   └── human_like_animal/     # AI-generated hybrid images
│   ├── af_data/                   # Original dataset
│   └── kaggle_raw/                # Raw Kaggle data
├── scripts/                       # Utility scripts
│   ├── download_kaggle_dataset.py # Dataset download script
│   ├── generate_human_animal_images.py # AI image generation script
│   └── setup_project.py           # Project setup script
├── notebooks/                     # Jupyter notebooks
│   └── animal_classification_analysis.ipynb
├── models/                        # Trained models
├── results/                       # Results and reports
├── docs/                          # Documentation
├── main_updated.py                # Updated main script
├── main.py                        # Original main script
└── requirements.txt               # Python dependencies
```

## Setup Instructions

### 1. Automatic Setup
```bash
cd scripts
python setup_project.py --full-setup
```

### 2. Manual Setup

#### Install Dependencies
```bash
pip install -r requirements.txt
```

#### Setup Git LFS
```bash
git lfs install
```

#### Download Dataset
```bash
cd scripts
python download_kaggle_dataset.py
```

#### Generate AI Images
```bash
cd scripts
python generate_human_animal_images.py
```

#### Train Model
```bash
python main_updated.py
```

#### Run Jupyter Notebook
```bash
jupyter notebook notebooks/animal_classification_analysis.ipynb
```

## Usage

### Basic Classification
```python
from main_updated import imgLoad, perform_pca_analysis, train_svm_classifier

# Load data
X, y, class_names = imgLoad('data/af_data_new')

# Perform PCA
X_pca, pca = perform_pca_analysis(X, y, class_names)

# Train classifier
# ... (see main_updated.py for complete example)
```

### Model Performance
- **Accuracy**: ~85-90% (depends on dataset quality)
- **Classes**: 4 (cat, dog, wild, human_like_animal)
- **Features**: 16,384 → 110 (PCA reduction)
- **Algorithm**: SVM with RBF kernel

## Advanced Usage

### Hyperparameter Tuning
```python
# Modify parameters in main_updated.py
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
    'kernel': ['rbf', 'poly', 'sigmoid']
}
```

### Custom Dataset
```python
# Use your own dataset
X, y, class_names = imgLoad('path/to/your/dataset')
```

## Model Files

Trained models are saved with timestamps:
- `pca_model_YYYYMMDD_HHMMSS.pkl`
- `svm_model_YYYYMMDD_HHMMSS.pkl`
- `scaler_YYYYMMDD_HHMMSS.pkl`
- `model_metadata_YYYYMMDD_HHMMSS.json`

## Visualization

The system generates several visualizations:
- PCA explained variance plots
- Eigenfaces visualization
- 2D PCA scatter plots
- Confusion matrices
- Classification reports

## Requirements

See `requirements.txt` for complete list of dependencies.

## Troubleshooting

### Common Issues

1. **Kaggle API Error**: Ensure kaggle.json is in ~/.kaggle/
2. **Memory Error**: Reduce image size or PCA components
3. **Import Error**: Check virtual environment activation
4. **Git LFS Error**: Install Git LFS separately

### Performance Tips

- Use SSD for faster data loading
- Increase RAM for larger datasets
- Use GPU for faster training (if available)

## License

This project is for educational purposes.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Contact

For questions or issues, please create an issue in the repository.
"""
    
    with open('../docs/README.md', 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print("✅ README created successfully")

def main():
    """Main setup function"""
    parser = argparse.ArgumentParser(description='Setup Animal Facial Classification Project')
    parser.add_argument('--full-setup', action='store_true', help='Run complete setup')
    parser.add_argument('--skip-dataset', action='store_true', help='Skip dataset download')
    parser.add_argument('--skip-ai-images', action='store_true', help='Skip AI image generation')
    parser.add_argument('--skip-training', action='store_true', help='Skip model training')
    
    args = parser.parse_args()
    
    print("🚀 Starting Animal Facial Classification Project Setup")
    print("=" * 60)
    
    # Always run these basic setup steps
    check_requirements()
    setup_directories()
    setup_git_lfs()
    
    if args.full_setup or not any([args.skip_dataset, args.skip_ai_images, args.skip_training]):
        if not args.skip_dataset:
            download_dataset()
        
        if not args.skip_ai_images:
            generate_ai_images()
        
        create_jupyter_notebook()
        
        if not args.skip_training:
            train_initial_model()
    
    create_readme()
    
    print("\n🎉 Setup Complete!")
    print("=" * 60)
    print("Next steps:")
    print("1. Review the generated README.md")
    print("2. Run 'python main_updated.py' to train models")
    print("3. Open 'notebooks/animal_classification_analysis.ipynb' in Jupyter")
    print("4. Check 'results/' directory for visualizations")

if __name__ == "__main__":
    main() 