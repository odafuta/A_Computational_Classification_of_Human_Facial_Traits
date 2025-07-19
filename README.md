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

# Install packages (incl. huggingface_hub)
pip install -r requirements.txt
```

### 2. Download Prepared Dataset (Hugging Face Hub)
A dataset of 450 x 3 classes + 30 `human_like_animal` images for training and evaluation is available.

```bash
# 1 行でダウンロード & 展開
title="Download af_data_new from HF Hub" && \
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="yourname/facial_traits_af_data_new",  # ← あなたの HF repo ID に置換
    repo_type="dataset",
    local_dir="data/af_data_new",
    local_dir_use_symlinks=False
)
PY
```

### 3. Organize Data

```
data/af_data_new/
├── cat/    (450 images)
├── dog/    (450 images)
├── tiger/  (450 images)
└── human_like_animal/ (30 images)
```

Then, you can run `python main_simple.py`.


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
- `data/` - Dataset directory (not in git except for data/af_data_new/human_like_animal)
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