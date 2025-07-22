# A Computational Classification of Human Facial Traits
## Which Animal Are You? 🐱🐶🦊

<<<<<<< HEAD
=======
**Assignment 2 - International Fusion Science Course**  
**Dr. Suyong Eum**  
**Osaka University**

>>>>>>> main
---

## Overview

Machine learning system to classify human facial traits using animal face classifiers (PCA + SVM).

**Results**: 87.29% accuracy on animal classification, 36.67% accuracy on human face evaluation.

---

## Setup
### 0. git clone
```bash
git clone https://github.com/odafuta/A_Computational_Classification_of_Human_Facial_Traits.git
```

### 1. Install Dependencies
```bash
# Create virtual environment
python -m venv facial_classification_env

# GitBash
source facial_classification_env/Scripts/activate

# Activate (Windows)
facial_classification_env\Scripts\activate

# Activate (Linux/Mac)
source facial_classification_env/bin/activate

# Install packages (take a few menutes) (incl. huggingface_hub)
pip install -r requirements.txt
```

### 2. Download Prepared Dataset (Hugging Face Hub)
A dataset of 450 x 3 classes + 30 `human_like_animal` images for training and evaluation is available.

```bash
# One-click download & unzip
python scripts/download_af_data_new.py
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

```bash
# (reccomend at farst execution) now latest model of us
python main_simple.py --model-dir models/20250719_161924

# Train new model
python main_simple.py

# Use existing model
python main_simple.py --use-existing
```

---

## Files

- `main.py` - PCA/SVC classification script
- `requirements.txt` - Python dependencies
- `data/` - Dataset directory (be made by executing the command above for downloading from my hugging face dataset)
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
- Hugging Face API for dataset download
- Kaggle API for dataset download
---
