# A Computational Classification of Human Facial Traits Using Animal Face Classifiers
## Technical Report

**Date**: July 12, 2025  
**Project**: Animal Face Feature Classification System

---

## 1. Data Collection and Preprocessing Methods

### 1.1 Data Collection Methods

#### 1.1.1 Animal Image Dataset
- **Data Source**: Kaggle Animal Faces Dataset (`andrewmvd/animal-faces`)
- **Collection Method**: Using Kaggle API commands
  ```bash
  pip install kaggle
  kaggle datasets download -d andrewmvd/animal-faces
  unzip animal-faces.zip -d data/kaggle_raw/
  ```
- **Data Structure**: 
  - Cat: 5,153 images
  - Dog: 4,739 images  
  - Wild: 4,738 images
- **Total**: 14,630 animal images (for training)

#### 1.1.2 Human-Animal Hybrid Images
- **Generation Method**: AI image generation services (Pollinations.ai API, Imagine4, ChatGPT)
- **Generated Count**: 30 images (10 per animal type)
- **File Naming Convention**: `{animal_name}_human_{number}.jpg`
  - Examples: `cat_human_01.jpg`, `dog_human_05.jpg`, `wild_human_03.jpg`
- **Implementation**: `scripts/generate_human_animal_images.py` for automated generation, manual browser generation
- **Features**: Realistic and anime-style human face images with animal-like characteristics

### 1.2 Preprocessing Methods

#### 1.2.1 Image Preprocessing Pipeline
1. **Image Loading**: OpenCV (`cv2.imread()`)
2. **Color Space Conversion**: BGR → RGB → Grayscale
3. **Size Standardization**: Resize to 128×128 pixels
4. **Cropping Process**: **Not implemented** - Only uniform resizing to 128×128
5. **Flattening**: Convert to 16,384-dimensional vectors
6. **Normalization**: Normalize to 0-1 range

#### 1.2.2 Preprocessing Parameters
- **Input Image Size**: 128×128 pixels
- **Feature Vector Length**: 16,384 dimensions
- **Data Split**: Training 80% (11,704 images) / Test 20% (2,926 images)

---

## 2. PCA (Principal Component Analysis) Implementation

### 2.1 Used Libraries
- **scikit-learn**: `sklearn.decomposition.PCA`
- **Function**: Dimensionality reduction through principal component analysis
- **Implementation Code**:
  ```python
  from sklearn.decomposition import PCA
  pca = PCA(n_components=110)
  X_pca = pca.fit_transform(X)
  ```

### 2.2 Hyperparameters and Their Impact

#### 2.2.1 Principal Component Number Selection
- **Selected Components**: 110 components
- **Selection Criteria**: Target of preserving 80%+ information
- **Actual Variance Ratio**: **80.89%**
- **Compression Ratio**: 16,384 → 110 (approximately 149× compression)

#### 2.2.2 Impact of Component Numbers
- **Too Few Components**: Large information loss, reduced classification accuracy
- **Too Many Components**: Include noise, risk of overfitting
- **Optimal Value**: Number of components preserving 80-90% variance ratio

---

## 3. SVM (Support Vector Machine) Implementation

### 3.1 Used Libraries
- **scikit-learn**: `sklearn.svm.SVC`
- **Grid Search**: `sklearn.model_selection.GridSearchCV`
- **Preprocessing**: `sklearn.preprocessing.StandardScaler`

### 3.2 Hyperparameters and Their Impact

#### 3.2.1 Optimized Parameters
```python
best_params = {
    'C': 10,           # Regularization parameter
    'gamma': 'scale',  # RBF kernel parameter
    'kernel': 'rbf'    # Radial basis function kernel
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
  - High value: Complex decision boundary, risk of overfitting
  - Low value: Simple decision boundary, improved generalization
- **Gamma Value**: RBF kernel influence range
  - High value: Local decision boundary, risk of overfitting
  - Low value: Smooth decision boundary, improved generalization
- **Kernel Types**: 
  - RBF: Suitable for non-linear separation
  - Poly: Non-linear separation using polynomials
  - Sigmoid: Similar to neural networks

---

## 4. Animal-like Human Face Images Used in Experiment 2

### 4.1 Collection and Labeling Methods

#### 4.1.1 AI-Generated Image Collection
- **Generation Tools**: Pollinations.ai API, Imagine4, ChatGPT
- **Generation Method**: Text prompt-based image generation
- **Selection Criteria**: Realistic and anime-style with moderate animal features, neither too animal-like nor too human-like

#### 4.1.2 Labeling Method
- **Approach**: Filename-based labeling
- **Naming Convention**: `{animal_name}_human_{number}.jpg`
- **3-Class Classification**: cat-like, dog-like, wild-like
- **Class Mapping**: `{'cat': 0, 'dog': 1, 'wild': 2}`

### 4.2 Specific Generation Examples

#### 4.2.1 Generated Image Distribution
- **Cat-like**: 10 images (cat_human_01.jpg ～ cat_human_10.jpg)
- **Dog-like**: 10 images (dog_human_01.jpg ～ dog_human_10.jpg)
- **Wild-like**: 10 images (wild_human_01.jpg ～ wild_human_10.jpg)

#### 4.2.2 Generation Prompt Examples
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

### 5.1 Model Evaluation: PCA Visualization

#### 5.1.1 Principal Component Analysis Results
- **Original Dimensions**: 16,384 dimensions
- **Reduced Dimensions**: 110 dimensions
- **Variance Ratio**: **80.89%**
- **Compression Ratio**: Approximately 149×

#### 5.1.2 Eigenfaces Visualization
- **Save Location**: `results/20250712_000423/eigenfaces.png`
- **Content**: Display top 6 principal components as images
- **Interpretation**: Each eigenface emphasizes different facial features (contour, eyes, nose, mouth, etc.)
- **Features**: Visualization of PCA eigenvectors as grayscale images

#### 5.1.3 Contribution Ratio Plot
- **Cumulative Contribution Ratio**: Progression of cumulative contribution ratio against number of components
- **Individual Contribution Ratio**: Individual contribution ratios of top 20 components
- **Save Location**: `results/20250712_000423/pca_analysis.png`

#### 5.1.4 2D Scatter Plot
- **Save Location**: `results/20250712_000423/pca_2d_visualization.png`
- **Content**: Distribution of each class in 1st and 2nd principal components
- **Color Coding**: Different colors for 3 classes (cat, dog, wild) - red, green, blue
- **Observation**: Visualization of class separability and overlapping regions

### 5.2 Model Evaluation: SVM Classification Results

#### 5.2.1 Animal Image Classification Results
- **Overall Accuracy**: **87.29%**
- **Cross-validation Accuracy**: **86.61%**
- **Confusion Matrix**: `results/20250712_000423/confusion_matrix.png`

#### 5.2.2 SVM Decision Boundary Visualization
- **Save Location**: `svm_boundary_visualization/svm_decision_boundary.png`
- **Implementation Method**: Decision boundary visualization in 2D PCA space
- **Visualization Content**: 
  - Color-coded display of decision boundary regions for 3 classes (cat, dog, wild)
  - Scatter plot of data points with color coding (red: cat, green: dog, blue: wild)
  - Confirmation of SVM's non-linear separation capability through boundary shapes
- **Technical Details**: 
  - Boundary calculation in 2D PCA-transformed feature space
  - Feature normalization using StandardScaler
  - Non-linear decision boundaries using RBF kernel
- **Observations**: 
  - Confirmation of clearly separated regions for each class
  - Visualization of classification uncertainty near boundaries
  - Complex decision boundary shapes in 3-class classification

### 5.3 Experiment 1: Accuracy for Each Animal Class

#### 5.3.1 Class-wise Performance
```python
classification_results = {
    'cat': {
        'precision': 0.88, 'recall': 0.88, 'f1_score': 0.88,
        'support': 1030
    },
    'dog': {
        'precision': 0.86, 'recall': 0.90, 'f1_score': 0.88,
        'support': 948
    },
    'wild': {
        'precision': 0.88, 'recall': 0.84, 'f1_score': 0.86,
        'support': 948
    }
}
```

#### 5.3.2 Confusion Matrix
- **Save Location**: `results/20250712_000423/confusion_matrix.png`
- **Overall Accuracy**: **87.29%**

### 5.4 Experiment 2: Human Image Animal Classification Results

#### 5.4.1 Classification Results Overview
- **Overall Accuracy**: **36.67%** (11/30 images correct)
- **Dog-like Bias**: 76.7% of images classified as dog-like
- **Cat-like Detection Difficulty**: All cat-like labeled images misclassified (0% accuracy)

#### 5.4.2 Detailed Classification Results
- **Cat-like Images (10 images)**:
   - Correct: 0 images (0%)
   - All misclassified as dog-like

- **Dog-like Images (10 images)**:
   - Correct: 9 images (90%)
   - Highest accuracy

- **Wild-like Images (10 images)**:
   - Correct: 2 images (20%)
   - Most misclassified as dog-like or cat-like

#### 5.4.3 Common Features
- **Dog-like Classification Commonalities**: Human facial muscles similar to dog-like features
- **Misclassification Tendencies**: AI-generated image features may differ from actual human faces

### 5.5 Experiment 3: Comprehensive Model Evaluation

#### 5.5.1 Overall Performance
- **Animal Classification**: High accuracy of 87.29%
- **Human Classification**: 36.67% accuracy (challenges exist)
- **Main Issues**: Difficulty in domain adaptation and bias toward dog-like features

#### 5.5.2 Discovered Insights
- **Feature Space Differences**: Large differences between animal and human facial feature spaces
- **Transfer Learning Limitations**: Difficulty applying animal-trained models to humans
- **Dog-like Feature Generality**: Human facial muscles most similar to dog-like features

---

## 6. Conclusions

### 6.1 Technical Achievements
- **High-accuracy Animal Classification System**: Achieved 87.29% accuracy
- **Effective Dimensionality Reduction**: 149× compression while preserving 80.89% information
- **Large-scale Data Processing**: Successfully processed 14,630 images
- **Complete Visualization System**: Implementation of eigenfaces, PCA scatter plots, SVM decision boundaries, confusion matrices

### 6.2 Identified Challenges
- **Domain Adaptation Difficulty**: Difficulty in feature transfer from animals to humans
- **Classification Bias Problem**: Strong bias toward dog-like features
- **AI-generated Image Limitations**: Feature differences from actual human faces

### 6.3 Future Improvement Directions
- **Domain Adaptation Methods**: Bridging animal and human feature spaces
- **Real Human Images**: Using actual human face images instead of AI-generated ones
- **Expert Labeling**: Objective evaluation of animal-like features
- **Data Augmentation**: Collection of more diverse human face images

---

## 7. Technical Achievements

### 7.1 System Implementation ✅
- **Complete Implementation**: 3-class animal classification system (87.29% accuracy)
- **Large-scale Data Processing**: Processing of 14,630 images
- **Optimization**: Hyperparameter optimization through grid search
- **Visualization**: Generation of eigenfaces, confusion matrices, PCA scatter plots, SVM decision boundaries

---

**Last Updated**: July 12, 2025  
**Animal Classification Accuracy**: 87.29%  
**Human Classification Accuracy**: 36.67%  
**Dataset**: 14,630 images (3-class animal classification)  
**Human Image Evaluation**: 30 images (100% processing success, 36.67% classification accuracy)  
**Technology Stack**: Python, scikit-learn, OpenCV, Kaggle API, AI Image Generation Service APIs  
**Key Findings**: Bias toward dog-like features (76.7%), difficulty detecting cat-like features (0% accuracy) 