# A Computational Classification of Human Facial Traits
## Which Animal Are You? 🐱🐶🦊

**Assignment 2 - International Fusion Science Course**  
**Dr. Suyong Eum**  
**Osaka University**

---

## Overview

Machine learning system to classify human facial traits using animal face classifiers (PCA + SVM).

**Results**: 87.29% accuracy on animal classification, 36.67% accuracy on human face evaluation.

---

## Setup

### 1. Install Dependencies
```bash
# Create virtual environment
python -m venv facial_classification_env

# Activate (Windows)
facial_classification_env\Scripts\activate

# Activate (Linux/Mac)
source facial_classification_env/bin/activate

# Install packages
pip install -r requirements.txt
```

### 2. Download Dataset
```bash
# Install Kaggle API
pip install kaggle

# Configure credentials (place kaggle.json in ~/.kaggle/)
kaggle datasets download -d andrewmvd/animal-faces

# Extract
unzip animal-faces.zip -d data/kaggle_raw/
```

### 3. Organize Data
Manually organize images into:
```
data/af_data_new/
├── cat/          # Cat images
├── dog/          # Dog images  
├── wild/         # Wild animal images
└── human_like_animal/  # Human test images (30 images)
```

---

## Usage

### wang sann Version
```bash
# Activate environment
# Windows: facial_classification_env\Scripts\activate
# Linux/Mac: source facial_classification_env/bin/activate

# Run the simple version
python main.py
```

### odafuta Version
```bash
# Train new model
python main_updated.py

# Use existing model
python main_updated.py --use-existing
```

---

## Files

- `main.py` - Simple classification script
- `main_updated.py` - Advanced implementation with evaluation
- `requirements.txt` - Python dependencies
- `data/` - Dataset directory (not in git)
- `models/` - Trained models
- `results/` - Results and visualizations
- `docs/` - Technical reports

---

## Requirements

- Python 3.8+
- scikit-learn (PCA, SVM, metrics)
- OpenCV (image processing)
- NumPy (numerical computing)
- Matplotlib (visualization)
- Seaborn (statistical plots)
- Pandas (data manipulation)
- Joblib (model persistence)
- Kaggle API for dataset download
---

## Academic Info

- **Email**: suyong@ist.osaka-u.ac.jp
- **Subject**: "G[X]-assignment2" 
- **Deadline**: July 24 