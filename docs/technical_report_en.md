# A Computational Classification of Human Facial Traits Using Animal Face Classifiers
## Technical Report

**Date**: July 12, 2025
**Project**: Animal Facial Feature Classification System
**Last Execution**: July 12, 2025 03:07:11

---

## 1. Data Collection and Preprocessing Methods

### 1.1 Data Collection Methods

#### 1.1.1 Animal Image Dataset
- **Data Source**: Kaggle Animal Faces Dataset (`andrewmvd/animal-faces`)
- **Collection Method**: Automated download using Kaggle API
- **Data Structure**: 
  - Cat: 5,153 animal face images
  - Dog: 4,739 animal face images  
  - Wild: 4,738 wild animal face images
- **Total**: 14,630 animal images (for training)
- **Implementation**: Automated via `scripts/download_kaggle_dataset.py`

#### 1.1.2 Human-Animal Hybrid Images
- **Generation Method**: AI image generation services (Pollinations.ai API, Imagine4, ChatGPT)
- **Generated Count**: 30 images (10 images per animal type)
- **File Naming Convention**: `{animal_name}_human_{number}.jpg`
  - Example: `cat_human_01.jpg`, `dog_human_05.jpg`, `wild_human_03.jpg`
- **Generation Prompts**:
  ```
  Cat-like humans:
  - "Color image of a human face with a slight cat-like appearance."
  - "Color image of a human face with a slight cat-like appearance like anime."
  
  Dog-like humans:
  - "Color image of a human face with a slight dog-like appearance."
  - "Color image of a human face with a slight dog-like appearance like anime."
  
  Wild animal-like humans:
  - "Color image of a human face with a slight tiger-like appearance."
  - "Color image of a human face with a slight tiger-like appearance like anime."
  ```
- **Implementation**: Automated via `scripts/generate_human_animal_images.py`, manual generation via browser
- **Features**: Human face images with animal-like characteristics in both realistic and anime styles

### 1.2 Preprocessing Methods

#### 1.2.1 Image Preprocessing Pipeline
1. **Image Loading**: OpenCV (`cv2.imread()`)
2. **Color Space Conversion**: BGR → RGB → Grayscale
3. **Size Standardization**: Resize to 128×128 pixels (`cv2.resize()`)
4. **Cropping**: **Not implemented** - Only uniform resizing to 128×128
5. **Flattening**: Convert to 16,384-dimensional vector
6. **Normalization**: Normalize to 0-1 range (`pixel_value / 255.0`)

#### 1.2.2 Actual Preprocessing Parameters
- **Input Image Size**: 128×128 pixels
- **Feature Vector Length**: 16,384 dimensions
- **Normalization Range**: [0, 1]
- **Data Split**: Training 80% (11,704 images) / Test 20% (2,926 images)
- **Cropping**: None (uniform resizing only)

---

## 2. PCA (Principal Component Analysis) Implementation

### 2.1 Libraries and Implementation Details

#### 2.1.1 Libraries
- **scikit-learn**: `sklearn.decomposition.PCA`
- **Functionality**: Dimensionality reduction via principal component analysis

#### 2.1.2 Implementation Code Example
```python
from sklearn.decomposition import PCA

# Create PCA object
pca = PCA(n_components=110)
X_pca = pca.fit_transform(X)

# Calculate explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_.sum()
```

### 2.2 Hyperparameters and Their Impact

#### 2.2.1 Principal Component Selection
- **Selected Components**: 110 components
- **Decision Basis**: Target of retaining 80%+ information
- **Actual Explained Variance**: **80.89%**
- **Compression Ratio**: 16,384 → 110 (approximately 149× compression)

#### 2.2.2 Impact of Component Count
- **Too Few Components**: Significant information loss, reduced classification accuracy
- **Too Many Components**: Noise inclusion, overfitting risk
- **Optimal Range**: Components retaining 80-90% explained variance

### 2.3 PCA Results Visualization

#### 2.3.1 Eigenfaces
- **Generated Count**: Top 6 components visualized
- **Save Location**: `results/20250712_000423/eigenfaces.png`
- **Interpretation**: Each component represents different facial feature patterns
- **Characteristics**: Each eigenface emphasizes different facial features (contours, eyes, nose, mouth, etc.)

#### 2.3.2 Explained Variance Plot
- **Cumulative Explained Variance**: Progression of cumulative explained variance vs. component count
- **Individual Explained Variance**: Individual explained variance for top 20 components
- **Save Location**: `results/20250712_000423/pca_analysis.png`

---

## 3. SVM (Support Vector Machine) Implementation

### 3.1 Libraries Used

#### 3.1.1 Libraries
- **scikit-learn**: `sklearn.svm.SVC`
- **Grid Search**: `sklearn.model_selection.GridSearchCV`
- **Preprocessing**: `sklearn.preprocessing.StandardScaler`
- **Evaluation**: `sklearn.metrics` (various evaluation metrics)

#### 3.1.2 Implementation Code Example
```python
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

# Standardization (Important: SVM is sensitive to feature scaling)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_pca)

# SVM implementation (with probability estimation)
svm = SVC(probability=True)
grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
```

### 3.2 Hyperparameters and Their Impact

#### 3.2.1 Optimized Parameters
```python
# Actual optimal parameters
best_params = {
    'C': 10,           # Regularization parameter
    'gamma': 0.01,     # RBF kernel parameter
    'kernel': 'rbf',   # Radial basis function kernel
    'probability': True # Enable probability estimation
}
```

#### 3.2.2 Grid Search Range
```python
param_grid = {
    'C': [0.1, 1, 10, 100],  # Regularization parameter
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],  # RBF kernel parameter
    'kernel': ['rbf', 'poly', 'sigmoid']  # Kernel function
}
```

#### 3.2.3 Parameter Impact
- **C Value**: Regularization strength
  - High value: Complex decision boundary, overfitting risk
  - Low value: Simple decision boundary, improved generalization
- **Gamma Value**: RBF kernel influence range
  - High value: Local decision boundary, overfitting risk
  - Low value: Smooth decision boundary, improved generalization
- **Kernel Type**: 
  - RBF: Suitable for non-linear separation
  - Poly: Polynomial-based non-linear separation
  - Sigmoid: Neural network-like

---

## 4. Animal-like Human Face Images for Experiment 2

### 4.1 Collection and Labeling Methods

#### 4.1.1 AI-Generated Image Collection
- **Generation Tools**: Pollinations.ai API (free), Imagine4, ChatGPT
- **Generation Method**: Text prompt-based image generation
- **Success Rate**: Low (multiple generations required to select appropriate images)
- **Selection Criteria**: Realistic and anime-style variations, not too animal-like nor too human-like, moderate animal-like features

#### 4.1.2 Labeling Method
- **Approach**: Filename-based labeling from generation context
- **Naming Convention**: `{animal_name}_human_{number}.jpg`
- **3-Class Classification**: cat-like, dog-like, wild-like
- **Class Mapping**: `{'cat': 0, 'dog': 1, 'wild': 2}`

### 4.2 Specific Generation Examples and Distribution

#### 4.2.1 Generated Image Distribution
- **Cat-like**: 10 images (cat_human_01.jpg ~ cat_human_10.jpg)
- **Dog-like**: 10 images (dog_human_01.jpg ~ dog_human_10.jpg)
- **Wild-like**: 10 images (wild_human_01.jpg ~ wild_human_10.jpg)

#### 4.2.2 Detailed Generation Prompts
```
Cat-like humans:
- "Color image of a human face with a slight cat-like appearance."
- "Color image of a human face with a slight cat-like appearance like anime."

Dog-like humans:
- "Color image of a human face with a slight dog-like appearance."
- "Color image of a human face with a slight dog-like appearance like anime."

Wild animal-like humans:
- "Color image of a human face with a slight tiger-like appearance."
- "Color image of a human face with a slight tiger-like appearance like anime."
```

---

## 5. Experimental Results

### 5.1 Model Evaluation Results

#### 5.1.1 Actual Dataset Scale
```python
# Confirmed data counts
dataset_summary = {
    'total_animal_images': 14630,
    'total_human_images': 30,
    'train_size': 11704,
    'test_size': 2926,
    'image_size': (128, 128),
    'feature_dim_original': 16384,
    'feature_dim_pca': 110
}
```

#### 5.1.2 Actual Preprocessing Parameters
- **PCA Components**: 110 components (reduced from 16,384 to 110 dimensions)
- **Image Size**: 128×128 pixels
- **Normalization**: 0-1 range normalization
- **Grayscale Conversion**: Implemented
- **Explained Variance**: **80.89%** (110 components retain approximately 81% of total information)

### 5.2 PCA Visualization Results (**Completed**)

#### 5.2.1 Principal Component Analysis Results
```python
# Actual results
pca_results = {
    'original_dimensions': 16384,
    'reduced_dimensions': 110,
    'explained_variance_ratio': 0.8089,  # 80.89%
    'compression_ratio': 149.0  # Approximately 149× compression
}
```

#### 5.2.2 Eigenface Visualization (**Completed**)
- **Save Location**: `results/20250712_000423/eigenfaces.png`
- **Content**: Top 6 principal components displayed as images
- **Interpretation**: Each eigenface emphasizes different facial features (contours, eyes, nose, mouth, etc.)
- **Characteristics**: PCA eigenvectors visualized as grayscale images

#### 5.2.3 2D Scatter Plot (**Completed**)
- **Save Location**: `results/20250712_000423/pca_2d_visualization.png`
- **Content**: Distribution of each class in 1st and 2nd principal components
- **Color Coding**: Different colors for 3 classes (cat, dog, wild)
- **Observation**: Visualization of class separability and overlap regions

### 5.3 SVM Classification Results (**Visualization Excluded, Otherwise Completed**)

#### 5.3.1 Optimized Hyperparameters
```python
# Actual optimal parameters
best_params = {
    'C': 10,           # Regularization parameter
    'gamma': 0.01,     # RBF kernel parameter
    'kernel': 'rbf',   # Radial basis function kernel
    'probability': True # Enable probability estimation
}

# Cross-validation score: 86.61%
# Test accuracy: 87.29%
```

#### 5.3.2 Evaluation Metrics (**Completed**)
- **Confusion Matrix**: `results/20250712_000423/confusion_matrix.png`
- **Overall Accuracy**: **87.29%**
- **Cross-validation Accuracy**: **86.61%**

### 5.4 Experiment 1: Per-Class Accuracy for Animal Classification (**Completed**)

#### 5.4.1 Actual Classification Results
```python
# Detailed performance per class (animal classification)
classification_results = {
    'cat': {
        'precision': 0.88,
        'recall': 0.88,
        'f1_score': 0.88,
        'support': 1030
    },
    'dog': {
        'precision': 0.86,
        'recall': 0.90,
        'f1_score': 0.88,
        'support': 948
    },
    'wild': {
        'precision': 0.88,
        'recall': 0.84,
        'f1_score': 0.86,
        'support': 948
    }
}
```

#### 5.4.2 Per-Class Performance Analysis
1. **Cat**: Stable performance
   - Precision: 88%, Recall: 88%, F1-score: 88%
   - Stable learning due to large dataset (5,153 images)

2. **Dog**: High recall
   - Precision: 86%, Recall: 90%, F1-score: 88%
   - High recall (few false negatives)

3. **Wild**: Balanced performance
   - Precision: 88%, Recall: 84%, F1-score: 86%
   - Good performance despite containing diverse animal species

### 5.5 Experiment 2: Human-Animal Classification Results (**Completed**)

#### 5.5.1 AI-Generated Image Classification Results
```python
# Human-animal hybrid image classification results
human_animal_classification = {
    'total_images': 30,
    'accuracy': 0.3667,  # 36.67%
    'correct_predictions': 11,
    'incorrect_predictions': 19,
    'dog_like_predictions': 23,  # 76.7%
    'cat_like_predictions': 4,   # 13.3%
    'wild_like_predictions': 3,  # 10.0%
}
```

#### 5.5.2 Detailed Classification Results Analysis

**Cat-like Human Image Classification Results**:
- Out of 10 generated images, actual classification results:
  - **Correct**: 0 images (0%) - All misclassified
  - Dog-like: 10 images (100%)
  - Cat-like: 0 images (0%)
  - Wild-like: 0 images (0%)

**Dog-like Human Image Classification Results**:
- Out of 10 generated images, actual classification results:
  - **Correct**: 9 images (90%) - Highest accuracy
  - Dog-like: 9 images (90%)
  - Wild-like: 1 image (10%)
  - Cat-like: 0 images (0%)

**Wild-like Human Image Classification Results**:
- Out of 10 generated images, actual classification results:
  - **Correct**: 2 images (20%) - Low accuracy
  - Dog-like: 4 images (40%)
  - Cat-like: 4 images (40%)
  - Wild-like: 2 images (20%)

#### 5.5.3 Classification Confidence Analysis
- **High Confidence Classification** (>0.9): 13 images (43.3%)
- **Medium Confidence Classification** (0.5-0.9): 15 images (50.0%)
- **Low Confidence Classification** (<0.5): 2 images (6.7%)

#### 5.5.4 Classification Tendency Analysis
- **Dog-like Bias**: 76.7% of images classified as dog-like
- **Cat-like Detection Difficulty**: All ground truth cat-like images misclassified
- **Wild-like Identification Difficulty**: 80% of ground truth wild-like images misclassified

#### 5.5.5 Individual Image Classification Results Details

**Cat-like Images (All Misclassified)**:
- cat_human_01.jpg: dog-like (confidence: 1.000)
- cat_human_02.jpg: dog-like (confidence: 0.805)
- cat_human_06.jpg: dog-like (confidence: 0.491) - Lowest confidence
- cat_human_10.jpg: dog-like (confidence: 0.989)

**Dog-like Images (90% Correct)**:
- dog_human_01.jpg: wild-like (confidence: 0.546) - Only misclassification
- dog_human_02.jpg: dog-like (confidence: 0.655) - Lowest confidence
- dog_human_10.jpg: dog-like (confidence: 0.994) - Highest confidence

**Wild-like Images (20% Correct)**:
- wild_human_01.jpg: wild-like (confidence: 0.925) ✓
- wild_human_08.jpg: wild-like (confidence: 0.980) ✓
- wild_human_04.jpg: cat-like (confidence: 0.289) - Lowest confidence
- wild_human_05.jpg: cat-like (confidence: 0.970) - High confidence misclassification

### 5.6 Experiment 3: Comprehensive Evaluation with Applied Model (**Completed**)

#### 5.6.1 Implementation Approach Effectiveness
- **Animal Classification**: Achieved 87.29% high accuracy
- **Human Evaluation**: 36.67% accuracy (challenges identified)
- **Data Imbalance Problem**: Resolved (separate approach)

#### 5.6.2 Model Characteristics Analysis
```python
# Comprehensive model performance
model_performance = {
    'animal_classification_accuracy': 87.29,  # High accuracy for animal classification
    'human_classification_accuracy': 36.67,  # Challenges in human classification
    'dog_like_bias': 76.7,  # Strong bias toward dog-like features
    'cat_like_detection': 0.0,  # Difficulty detecting cat-like features
    'wild_like_detection': 20.0,  # Limited detection of wild-like features
}
```

#### 5.6.3 Classification Bias Root Cause Analysis
1. **Training Data Influence**: Model trained on animal faces applied to human faces
2. **Feature Space Differences**: Different feature spaces between animal and human faces
3. **Dog-like Feature Generalizability**: Human facial muscles similar to dog-like features
4. **Cat/Wild Feature Specificity**: More specific features making detection difficult

---

## 6. Comprehensive Evaluation and Discussion

### 6.1 Technical Achievements ✅

#### 6.1.1 High-Accuracy Classification System
- **Animal Classification Accuracy**: **87.29%** (3-class classification)
- **Cross-validation Accuracy**: **86.61%** (stability confirmed)
- **Large-scale Data Processing**: Successfully processed 14,630 images

#### 6.1.2 Effective Dimensionality Reduction
- **PCA Compression Ratio**: 149× (16,384 → 110 dimensions)
- **Information Retention**: **80.89%**
- **Computational Efficiency**: Significant processing time reduction

### 6.2 Experimental Design Achievements and Challenges ⚠️

#### 6.2.1 Successful Aspects
- **Data Imbalance Problem Resolution**: Separated 3-class animal classification + human evaluation
- **Stable Animal Classification**: 87.29% high accuracy
- **AI-Generated Image Utilization**: Successfully generated 30 evaluation images

#### 6.2.2 Identified Challenges
- **Low Human Classification Accuracy**: 36.67% (barely above random chance of 33.3%)
- **Dog-like Bias**: 76.7% of images classified as dog-like
- **Cat-like Detection Failure**: All prompt-based labeled cat-like images misclassified

### 6.3 Discovered Insights 💡

#### 6.3.1 Domain Adaptation Difficulties
- **Feature Space Differences**: Significant differences between animal and human facial feature spaces
- **Transfer Learning Limitations**: Difficulty applying animal-trained models to humans
- **Dog-like Feature Generalizability**: Human facial muscles most similar to dog-like features

#### 6.3.2 AI-Generated Image Characteristics
- **Generation Quality Impact**: AI-generated image features may differ from actual human faces
- **Prompt Influence**: Generation prompts may not accurately reflect intended features
- **Evaluation Difficulty**: Subjectivity and quantification challenges of "animal-like features"

### 6.4 Technical Insights 🔍

#### 6.4.1 PCA Effectiveness
- **Significant Dimensionality Reduction**: 99.3% dimension reduction while maintaining high accuracy (animal classification)
- **Noise Removal Effect**: Extraction of only principal features
- **Computational Efficiency**: Significant reduction in training and inference time

#### 6.4.2 SVM Applicability
- **Success in Animal Classification**: High-accuracy classification through non-linear separation
- **Challenges in Human Classification**: Performance degradation due to feature space differences
- **Probability Estimation Utility**: Enables classification confidence evaluation

### 6.5 Future Improvement Directions 🚀

#### 6.5.1 Technical Improvements
- **Domain Adaptation**: Methods to bridge animal and human feature spaces
- **Fine-tuning**: Additional learning with human face data
- **Multi-modal Learning**: Combination of multiple feature extraction methods

#### 6.5.2 Data Improvements
- **Real Human Images**: Actual human face images instead of AI-generated ones
- **Expert Labeling**: Objective evaluation of animal-like features
- **Data Augmentation**: Collection of more diverse human face images

---

## 7. Technical Achievements (Confirmed)

### 7.1 System Implementation ✅
- **Complete Implementation**: 3-class animal classification system (87.29% accuracy)
- **Large-scale Data Processing**: Processed 14,630 images
- **Optimization**: Hyperparameter optimization via grid search
- **Visualization**: Generated eigenfaces, confusion matrices, PCA scatter plots

### 7.2 Experimental Scale (Confirmed) ✅
- **Data Count**: 14,630 images (cat: 5,153, dog: 4,739, wild: 4,738)
- **Human Images**: 30 images (AI-generated)
- **Features**: 16,384 dimensions → 110 dimensions (80.89% information retention)
- **Classifier**: SVM (C=10, gamma=0.01, RBF kernel, probability=True)
- **Evaluation**: 5-fold cross-validation (86.61%), test accuracy (87.29%)

### 7.3 Academic Contributions ✅
1. **Technical**: High-accuracy animal face classification system using PCA+SVM
2. **Methodological**: Demonstrated dimensionality reduction effectiveness on large datasets
3. **Problem Discovery**: Identified domain adaptation difficulties and bias issues
4. **Problem Solving**: Presented effective solutions for data imbalance problems

### 7.4 Practical Applications and Challenges ⚠️
- **Success Case**: Animal classification system (87.29% accuracy)
- **Challenge**: Human animal-like feature classification (36.67% accuracy)
- **Improvement Needed**: Domain adaptation and data quality enhancement
- **Future Potential**: Practical application possible with technical improvements

---

**Last Updated**: July 12, 2025  
**Experiment Status**: Completed  
**Animal Classification Accuracy**: 87.29%  
**Human Classification Accuracy**: 36.67%  
**Dataset**: 14,630 images (3-class animal classification)  
**Human Image Evaluation**: 30 images (100% processing success, 36.67% classification accuracy)  
**Technology Stack**: Python, scikit-learn, OpenCV, Kaggle API, AI image generation service APIs  
**Key Findings**: Dog-like feature bias (76.7%), Cat-like feature detection difficulty (0% accuracy) 