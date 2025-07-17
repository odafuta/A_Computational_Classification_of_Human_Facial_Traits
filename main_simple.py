import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import cv2
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_animal_data(data_dir="data/af_data_new"):
    """Load animal image data (3 classes: cat, dog, wild)"""
    print("=== Data Collection and Preprocessing ===")
    print(f"Data source: {data_dir}")
    print("Preprocessing: RGB→Grayscale conversion, 128×128 resize, normalization (0-1)")
    print("Cropping process: None (uniform resize only)")
    
    data_path = Path(data_dir)
    animal_classes = ['cat', 'dog', 'wild']
    X, y = [], []
    
    for class_idx, class_name in enumerate(animal_classes):
        class_dir = data_path / class_name
        if not class_dir.exists():
            continue
            
        image_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
        print(f"{class_name}: {len(image_files)} images")
        
        for img_path in image_files:
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (128, 128))
                img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                img_normalized = img_gray.flatten() / 255.0
                X.append(img_normalized)
                y.append(class_idx)
            except:
                continue
    
    print(f"Total data count: {len(X)} images")
    return np.array(X), np.array(y), animal_classes

def load_human_data(data_dir="data/af_data_new"):
    """Load human image data (animal-like features)"""
    print("\n=== Experiment 2: Animal-like Human Face Images ===")
    print("Collection method: AI image generation services (Pollinations.ai, Imagine4, ChatGPT)")
    print("Labeling method: Filename-based (e.g., cat_human_01.jpg)")
    print("Examples: 'cat-like human face', 'dog-like human face', 'wild animal-like human face'")
    
    data_path = Path(data_dir) / "human_like_animal"
    if not data_path.exists():
        return np.array([]), np.array([]), []
    
    X_human, y_human, human_files = [], [], []
    class_mapping = {'cat': 0, 'dog': 1, 'wild': 2}
    
    image_files = list(data_path.glob("*.jpg")) + list(data_path.glob("*.png"))
    print(f"Human image count: {len(image_files)}")
    
    for img_path in image_files:
        try:
            filename = img_path.name
            animal_name = filename.split('_')[0].lower()
            if animal_name not in class_mapping:
                continue
            
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (128, 128))
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img_normalized = img_gray.flatten() / 255.0
            
            X_human.append(img_normalized)
            y_human.append(class_mapping[animal_name])
            human_files.append(filename)
        except:
            continue
    
    for class_name in ['cat', 'dog', 'wild']:
        count = sum(1 for f in human_files if f.startswith(class_name))
        print(f"{class_name}-like: {count} images")
    
    return np.array(X_human), np.array(y_human), human_files

def perform_pca_analysis(X, y, class_names, n_components=110, results_dir=None):
    """PCA analysis and visualization"""
    print(f"\n=== PCA Implementation ===")
    print("Library used: scikit-learn PCA")
    print("Hyperparameter: n_components (number of principal components)")
    print("Impact: More components → Higher information retention, Higher computational cost, Higher overfitting risk")
    
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    
    explained_variance = pca.explained_variance_ratio_.sum()
    print(f"Number of components: {n_components}")
    print(f"Explained variance ratio: {explained_variance:.4f}")
    print(f"Eigenvalues (top 5): {pca.explained_variance_[:5]}")
    
    # Eigenfaces visualization
    plt.figure(figsize=(12, 4))
    for i in range(min(6, n_components)):
        plt.subplot(2, 3, i + 1)
        eigenface = pca.components_[i].reshape(128, 128)
        plt.imshow(eigenface, cmap='gray')
        plt.title(f'Eigenface {i+1}')
        plt.axis('off')
    plt.suptitle('PCA Eigenfaces (Top 6 Components)')
    plt.tight_layout()
    if results_dir:
        plt.savefig(results_dir / 'eigenfaces.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2D scatter plot
    plt.figure(figsize=(10, 8))
    colors = ['red', 'green', 'blue']
    for i, class_name in enumerate(class_names):
        mask = y == i
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                   c=colors[i], label=class_name, alpha=0.6, s=50)
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title('PCA 2D Visualization (Animal Images)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    if results_dir:
        plt.savefig(results_dir / 'pca_2d_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return X_pca, pca

def train_svm_classifier(X_train, X_test, y_train, y_test, class_names, results_dir=None):
    """Train SVM classifier"""
    print(f"\n=== SVM Implementation ===")
    print("Library used: scikit-learn SVC")
    print("Hyperparameters: C (regularization), gamma (RBF kernel), kernel (kernel function)")
    print("Impact: Higher C → Complex boundary, Higher overfitting risk / Higher gamma → Local boundary, Higher overfitting risk")
    
    # Feature standardization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Grid search
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
        'kernel': ['rbf', 'poly', 'sigmoid']
    }
    
    print("Running grid search...")
    svm = SVC()
    grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Cross-validation score: {grid_search.best_score_:.4f}")
    
    # Final model
    svm_model = grid_search.best_estimator_
    y_pred = svm_model.predict(X_test_scaled)
    
    # Experiment 1: Accuracy for each animal class
    print(f"\n=== Experiment 1: Animal Classification Accuracy ===")
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Overall accuracy: {accuracy:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - Animal Classification')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    if results_dir:
        plt.savefig(results_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return svm_model, scaler

def visualize_svm_decision_boundary(X, y, class_names, results_dir=None):
    """SVM decision boundary visualization (corrected version)"""
    print(f"\n=== SVM Decision Boundary Visualization ===")
    
    # 2D PCA
    pca_2d = PCA(n_components=2)
    X_2d = pca_2d.fit_transform(X)
    
    # Standardization
    scaler = StandardScaler()
    X_2d_scaled = scaler.fit_transform(X_2d)
    
    # SVM training
    svm_vis = SVC(kernel='rbf', C=10, gamma='scale')
    svm_vis.fit(X_2d_scaled, y)
    
    # Mesh adjusted to data range (corrected)
    margin = 0.5
    x_min, x_max = X_2d_scaled[:, 0].min() - margin, X_2d_scaled[:, 0].max() + margin
    y_min, y_max = X_2d_scaled[:, 1].min() - margin, X_2d_scaled[:, 1].max() + margin
    
    h = 0.02
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))
    
    Z = svm_vis.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Visualization (corrected)
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    
    colors = ['red', 'green', 'blue']
    for i in range(len(class_names)):  # 3 classes only
        mask = y == i
        plt.scatter(X_2d_scaled[mask, 0], X_2d_scaled[mask, 1],
                   c=colors[i], label=class_names[i], edgecolor='k', alpha=0.8)
    
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel('First Principal Component (Scaled)')
    plt.ylabel('Second Principal Component (Scaled)')
    plt.title('SVM Decision Boundary (2D PCA Space)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    if results_dir:
        plt.savefig(results_dir / 'svm_decision_boundary.png', dpi=300, bbox_inches='tight')
    plt.show()

def evaluate_human_faces(pca, svm_model, scaler, class_names, X_human, y_human, human_files, results_dir=None):
    """Experiment 2: Human image classification evaluation"""
    if len(X_human) == 0:
        print("No human images found")
        return []
    
    print(f"\n=== Experiment 2: Human Image Classification Results ===")
    
    # Same preprocessing pipeline
    X_human_pca = pca.transform(X_human)
    X_human_scaled = scaler.transform(X_human_pca)
    
    # Prediction
    predictions = svm_model.predict(X_human_scaled)
    
    # Display results
    print("Classification results:")
    results = []
    for i, (filename, pred) in enumerate(zip(human_files, predictions)):
        ground_truth = class_names[y_human[i]]
        predicted = class_names[pred]
        correct = "✓" if pred == y_human[i] else "✗"
        print(f"{filename}: {predicted} | Ground truth: {ground_truth} {correct}")
        
        results.append({
            'filename': filename,
            'predicted': predicted,
            'ground_truth': ground_truth,
            'correct': pred == y_human[i]
        })
    
    # Accuracy evaluation
    accuracy = accuracy_score(y_human, predictions)
    print(f"\nHuman image classification accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Classification report
    print("\nClassification report:")
    print(classification_report(y_human, predictions, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(y_human, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - Human Image Classification')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    if results_dir:
        plt.savefig(results_dir / 'human_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Classification results visualization
    class_counts = {'cat': 0, 'dog': 0, 'wild': 0}
    for result in results:
        class_counts[result['predicted']] += 1
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(class_counts.keys(), class_counts.values(), color=['red', 'green', 'blue'])
    plt.title('Human Image Classification Results')
    plt.xlabel('Animal Class')
    plt.ylabel('Number of Images')
    for bar, count in zip(bars, class_counts.values()):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{count}', ha='center', va='bottom')
    if results_dir:
        plt.savefig(results_dir / 'human_classification_summary.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return results

def main():
    """Main processing"""
    print("=== Animal Face Feature Human Classification System ===")
    
    # Results save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path('results') / timestamp
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results save location: {results_dir}")
    
    # 1. Load animal data
    X, y, class_names = load_animal_data()
    
    # 2. PCA analysis
    X_pca, pca = perform_pca_analysis(X, y, class_names, results_dir=results_dir)
    
    # 3. Data split
    X_train, X_test, y_train, y_test = train_test_split(
        X_pca, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 4. SVM training
    svm_model, scaler = train_svm_classifier(X_train, X_test, y_train, y_test, class_names, results_dir)
    
    # 5. SVM decision boundary visualization
    visualize_svm_decision_boundary(X, y, class_names, results_dir)
    
    # 6. Human image evaluation
    X_human, y_human, human_files = load_human_data()
    if len(X_human) > 0:
        human_results = evaluate_human_faces(pca, svm_model, scaler, class_names, 
                                           X_human, y_human, human_files, results_dir)
    
    print(f"\n=== Experiment Complete ===")
    print(f"All results saved to {results_dir}")

if __name__ == "__main__":
    main() 