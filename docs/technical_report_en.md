# Animal Face Feature Classification System – Technical Report (Model 20250719_161924)
## A Computational Classification of Human Facial Traits

**Date**: 19 July 2025  
**Script**: `main_simple.py`  
**Model ID**: `20250719_161924`

---

## 0. Overview
This project trains a 3-class animal face classifier (cat, dog, tiger) and applies it to human faces to estimate “which animal the face most resembles.”  
The latest model achieves **83.1 %** cross-validation, **81.8 %** validation, and **87.7 %** test accuracy.

---

## 1. Dataset & Pre-processing
- **Animal images**: 1 350 (train 70 % = 944, val 15 % = 203, test 15 % = 203)  
- **Human images**: 30 (AI-generated, labelled via filename)  
- **Pipeline**: RGB → Grayscale → resize 128×128 → normalise (0–1) → flatten (16 384-dim)

---

## 2. Model Configuration
```
StandardScaler → PCA(n_components=110, explained_variance=0.823) → SVC(kernel='rbf', C=10, γ='scale')
```
- PCA compresses dimensions **149×** (16 384 → 110).  
- Hyper-parameters optimised with 5-fold GridSearchCV.

### 2.1 Implementation Flow (pseudo-code)
```python
# condensed from main_simple.py

def main():
    setup_plotting()                    # 0) style
    args = parse_arguments()            # 1) CLI
    ts = now(); models_root, res_dir = create_directories(ts)

    redirect_stdout_to(res_dir/'experiment_log.txt')  # 2) logging

    # 3) load data
    X_animal, y_animal, _ = load_animal_data()
    X_human,  y_human, fns = load_human_data()

    # 4) load or train model
    if args.use_existing or args.model_dir:
        pipeline, _ = load_model_and_metadata(...)
    else:
        pipeline, m = train_model_with_validation(X_animal, y_animal, res_dir)
        save_model_and_metadata(pipeline, m, ts, models_root)

    # 5) visualisations
    visualize_pca_analysis(X_animal, y_animal, pipeline.named_steps['pca'], res_dir)
    if not args.skip_boundary:
        visualize_svm_decision_boundary(X_animal, y_animal, pipeline, res_dir)

    # 6) evaluation
    evaluate_animal_test_set(pipeline, X_animal, y_animal, res_dir)
    evaluate_human_faces(pipeline, X_human, y_human, fns, res_dir)

    print('=== Experiment Complete ===')


def create_model_pipeline():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('pca',    PCA(n_components=110)),
        ('svc',    SVC(kernel='rbf'))
    ])
```
This pseudo-code shows the full workflow: **data → training → evaluation → save & visualise**.

---

## 3. Metrics (from `models/20250719_161924/metadata.json`)
| Metric | Value |
|--------|-------|
| CV accuracy (mean) | **0.8305** |
| CV std-dev | 0.0195 |
| Validation accuracy | **0.8177** |
| Test accuracy | **0.8768** |
| PCA cumulative variance | 0.8230 |

### 3.1 Animal Test-set Classification Report
| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| cat   | 0.84 | 0.91 | 0.87 | 67 |
| dog   | 0.90 | 0.81 | 0.85 | 68 |
| tiger | 0.90 | 0.91 | 0.91 | 68 |
| **Overall** | – | – | **0.88** | 203 |

### 3.2 Human Image Classification Report (30 images)
| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| cat   | 0.12 | 0.10 | 0.11 | 10 |
| dog   | 0.35 | 0.70 | 0.47 | 10 |
| tiger | 1.00 | 0.20 | 0.33 | 10 |
| **Overall** | – | – | **0.33** | 30 |

---

## 4. Result Files (`results/20250719_161924/`)
| # | File | Description & how to interpret |
|---|------|-------------------------------|
| 1 | `experiment_log.txt` | Full console log. Reproduces hyper-parameter search & score progression. |
| 2 | `pca_analysis.png` | Left: cumulative variance, right: top-20 component variance. Confirms 110 components > 80 % variance. |
| 3 | `eigenfaces.png` | Top-6 principal components visualised as “eigenfaces” for intuitive feature understanding. |
| 4 | `pca_2d_visualization.png` | 2-D scatter (PC1 vs PC2) coloured by class. Visualises class separation. |
| 5 | `svm_decision_boundary.png` | SVM decision regions plotted in 2-D PCA space; shows boundary complexity & error zones. |
| 6 | `animal_confusion_matrix.png` | Confusion matrix for the animal test set. Identify mis-classification patterns. |
| 7 | `animal_classification_summary.png` | Bar chart of predicted class counts for the test set. Detect class imbalance/bias. |
| 8 | `animal_classification_report.txt` | Text report (precision / recall / F1 / support) for quantitative comparison. |
| 9 | `human_confusion_matrix.png` | Confusion matrix for 30 human images, revealing dominant predicted classes. |
|10 | `human_classification_summary.png` | Bar chart of predicted class distribution for human images (dog bias, etc.). |

---

## 5. Discussion
- **Improved Test Accuracy**: +0.9 pt versus previous model; validation ≤ test suggests reduced data-shift.
- **Remaining Confusion**: Mis-classification mainly between dog ↔ tiger; richer features may help.
- **Human Evaluation**: Strong bias toward dog class persists; domain adaptation required.

---

## 6. Future Work
1. **Wider hyper-parameter search** (C, γ) incl. Bayesian optimisation.  
2. **Data augmentation**: rotation, colour jitter, etc. to diversify training data.  
3. **Domain adaptation**: fine-tune with human faces to bridge animal→human gap.  
4. **Model benchmarking**: compare with RandomForest, CNN, etc. in accuracy & cost.

---
**Last updated**: 19 Jul 2025  
**Animal accuracy**: 88 %  
**Human accuracy**: 33 %  
**Dataset**: 1 350 animal images (3 classes)  
**Human evaluation**: 30 images (33 % accuracy)  
**Tech stack**: Python, scikit-learn, OpenCV, Kaggle API, AI image generation APIs  
**Key findings**: Bias towards dog-like features; difficulty detecting cat-like features 