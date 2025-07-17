#!/usr/bin/env python3
"""
Animal Facial Classification using PCA and SVM
Human face classification into animal categories: cat, dog, wild

This script performs:
1. Data loading and preprocessing (animal faces for training)
2. PCA dimensionality reduction with visualization
3. SVM classification with detailed evaluation
4. Human face evaluation (classify human faces as animal-like)
5. Model persistence and visualization

Directory structure:
data/
├── af_data_new/
│   ├── cat/          # Training data: cat faces
│   ├── dog/          # Training data: dog faces
│   ├── wild/         # Training data: wild animal faces
│   └── human_like_animal/  # Test data: human faces to classify

Usage:
    python main_updated.py                                    # Train new model
    python main_updated.py --use-existing                     # Use existing model (interactive)
    python main_updated.py --model-dir models/20250712_000423 # Use specific model
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
import cv2
from pathlib import Path
from datetime import datetime
import warnings
import argparse
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def imgLoad(data_dir="data/af_data_new", img_size=(128, 128)):
    """Load and preprocess animal images for training"""
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_path}")
    
    print(f"Loading animal training images from: {data_path}")
    animal_classes = ['cat', 'dog', 'wild']
    X, y, class_names = [], [], []
    
    for class_idx, class_name in enumerate(animal_classes):
        class_dir = data_path / class_name
        if not class_dir.exists():
            continue
            
        class_names.append(class_name)
        image_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpeg"))
        print(f"Processing {class_name}: {len(image_files)} images")
        
        for img_path in image_files:
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, img_size)
                img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                img_normalized = img_gray.flatten() / 255.0
                
                X.append(img_normalized)
                y.append(class_idx)
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
    
    print(f"Total training images loaded: {len(X)}")
    return np.array(X), np.array(y), class_names

def load_human_faces(data_dir="data/af_data_new", img_size=(128, 128)):
    """Load human faces for evaluation with ground truth labels from filenames"""
    data_path = Path(data_dir) / "human_like_animal"
    if not data_path.exists():
        return np.array([]), np.array([]), [], []
    
    print(f"Loading human faces from: {data_path}")
    X_human, y_human, human_files = [], [], []
    class_mapping = {'cat': 0, 'dog': 1, 'wild': 2}
    class_names = ['cat', 'dog', 'wild']
    
    image_files = list(data_path.glob("*.jpg")) + list(data_path.glob("*.png")) + list(data_path.glob("*.jpeg"))
    print(f"Found {len(image_files)} human face images")
    
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
            img = cv2.resize(img, img_size)
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img_normalized = img_gray.flatten() / 255.0
            
            X_human.append(img_normalized)
            y_human.append(class_mapping[animal_name])
            human_files.append(filename)
            
        except Exception as e:
            continue
    
    return np.array(X_human), np.array(y_human), human_files, class_names

def perform_pca_analysis(X, y, class_names, n_components=110, results_dir=None):
    """Perform PCA analysis and visualization"""
    print(f"\n=== PCA Analysis ===")
    print(f"Original feature dimension: {X.shape[1]}")
    
    max_components = min(X.shape[0], X.shape[1])
    if n_components > max_components:
        n_components = max_components - 1
    
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    
    print(f"PCA-transformed dimension: {X_pca.shape[1]}")
    print(f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")
    
    # Visualize explained variance
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title('PCA Explained Variance')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.bar(range(min(20, len(pca.explained_variance_ratio_))), 
            pca.explained_variance_ratio_[:20])
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Top 20 Principal Components')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if results_dir:
        plt.savefig(results_dir / 'pca_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Visualize eigenfaces
    n_eigenfaces = min(6, n_components)
    img_shape = (128, 128)
    
    plt.figure(figsize=(12, 8))
    for i in range(n_eigenfaces):
        plt.subplot(2, 3, i + 1)
        eigenface = pca.components_[i].reshape(img_shape)
        plt.imshow(eigenface, cmap='gray')
        plt.title(f'Eigenface {i+1}')
        plt.axis('off')
    
    plt.suptitle('Principal Components (Eigenfaces)')
    plt.tight_layout()
    if results_dir:
        plt.savefig(results_dir / 'eigenfaces.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return X_pca, pca

def visualize_pca_2d(X_pca, y, class_names, results_dir=None):
    """Visualize PCA results in 2D space"""
    plt.figure(figsize=(10, 8))
    colors = plt.cm.Set3(np.linspace(0, 1, len(class_names)))
    
    for i, class_name in enumerate(class_names):
        mask = y == i
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                   c=[colors[i]], label=class_name, alpha=0.6, s=50)
    
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title('PCA Visualization (First Two Components) - Animal Training Data')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if results_dir:
        plt.savefig(results_dir / 'pca_2d_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()

def visualize_svm_decision_boundary(X, y, class_names, results_dir=None):
    """Visualize SVM decision boundary in 2D PCA space"""
    print(f"\n=== SVM Decision Boundary Visualization ===")
    
    # Use 2D PCA for visualization
    pca_2d = PCA(n_components=2)
    X_2d = pca_2d.fit_transform(X)
    
    # Train SVM on 2D data
    scaler = StandardScaler()
    X_2d_scaled = scaler.fit_transform(X_2d)
    svm_vis = SVC(kernel='rbf', C=10, gamma='scale')
    svm_vis.fit(X_2d_scaled, y)
    
    # Create mesh for decision boundary
    h = 0.02
    x_min, x_max = X_2d_scaled[:, 0].min() - 1, X_2d_scaled[:, 0].max() + 1
    y_min, y_max = X_2d_scaled[:, 1].min() - 1, X_2d_scaled[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))
    
    Z = svm_vis.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary
    plt.figure(figsize=(12, 8))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    
    colors = ['red', 'green', 'blue']
    for i, class_name in enumerate(class_names):
        mask = y == i
        plt.scatter(X_2d_scaled[mask, 0], X_2d_scaled[mask, 1],
                   c=colors[i], label=class_name, edgecolor='k', alpha=0.8)
    
    plt.xlabel('First Principal Component (Scaled)')
    plt.ylabel('Second Principal Component (Scaled)')
    plt.title('SVM Decision Boundary (2D PCA Space)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if results_dir:
        plt.savefig(results_dir / 'svm_decision_boundary.png', dpi=300, bbox_inches='tight')
    plt.show()

def train_svm_classifier(X_train, X_test, y_train, y_test, class_names, results_dir=None):
    """Train SVM classifier with hyperparameter tuning"""
    print(f"\n=== SVM Classification ===")
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Hyperparameter tuning
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
        'kernel': ['rbf', 'poly', 'sigmoid']
    }
    
    print("Performing hyperparameter tuning...")
    svm = SVC(probability=True)
    grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    # Train final model
    svm_model = grid_search.best_estimator_
    y_pred = svm_model.predict(X_test_scaled)
    
    # Evaluate model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {accuracy:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - Animal Classification')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    if results_dir:
        plt.savefig(results_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return svm_model, y_pred, scaler

def evaluate_human_faces(pca, svm_model, scaler, class_names, X_human, human_files, results_dir=None, y_human=None):
    """Evaluate human faces and classify them as animal-like"""
    if len(X_human) == 0:
        print("No human faces found for evaluation.")
        return []
    
    print(f"\n=== Human Face Evaluation ===")
    print(f"Evaluating {len(X_human)} human faces...")
    
    # Transform human faces using the same pipeline
    X_human_pca = pca.transform(X_human)
    X_human_scaled = scaler.transform(X_human_pca)
    
    # Predict animal classes for human faces
    predictions = svm_model.predict(X_human_scaled)
    prediction_proba = svm_model.predict_proba(X_human_scaled)
    
    # Display results
    print("\nHuman Face Classification Results:")
    print("=" * 50)
    
    results = []
    for i, (filename, pred, proba) in enumerate(zip(human_files, predictions, prediction_proba)):
        animal_class = class_names[pred]
        confidence = proba[pred]
        
        if y_human is not None:
            ground_truth = class_names[y_human[i]]
            correct = "✓" if pred == y_human[i] else "✗"
            print(f"{filename}: {animal_class}-like (confidence: {confidence:.3f}) | GT: {ground_truth} {correct}")
        else:
            print(f"{filename}: {animal_class}-like (confidence: {confidence:.3f})")
        
        results.append({
            'filename': filename,
            'predicted_class': animal_class,
            'predicted_idx': pred,
            'confidence': confidence,
            'ground_truth': class_names[y_human[i]] if y_human is not None else None,
            'correct': pred == y_human[i] if y_human is not None else None
        })
    
    # Calculate evaluation metrics if ground truth is available
    if y_human is not None:
        accuracy = accuracy_score(y_human, predictions)
        print(f"\n=== Human Face Classification Performance ===")
        print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_human, predictions, target_names=class_names))
        
        # Confusion matrix
        cm = confusion_matrix(y_human, predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix - Human Face Classification')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        if results_dir:
            plt.savefig(results_dir / 'human_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # Create visualization of human face classifications
    if len(results) > 0:
        visualize_human_classifications(results, class_names, results_dir, y_human)
    
    return results

def visualize_human_classifications(results, class_names, results_dir=None, y_human=None):
    """Visualize human face classification results"""
    class_counts = {name: 0 for name in class_names}
    for result in results:
        class_counts[result['predicted_class']] += 1
    
    if y_human is not None:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Predicted classifications
        bars1 = axes[0, 0].bar(class_counts.keys(), class_counts.values())
        axes[0, 0].set_title('Predicted Classifications')
        axes[0, 0].set_xlabel('Animal Class')
        axes[0, 0].set_ylabel('Number of Human Faces')
        for bar in bars1:
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')
        
        # Ground truth distribution
        ground_truth_counts = {name: 0 for name in class_names}
        for result in results:
            if result['ground_truth']:
                ground_truth_counts[result['ground_truth']] += 1
        
        bars2 = axes[0, 1].bar(ground_truth_counts.keys(), ground_truth_counts.values(), color='lightcoral')
        axes[0, 1].set_title('Ground Truth Distribution')
        axes[0, 1].set_xlabel('Animal Class')
        axes[0, 1].set_ylabel('Number of Human Faces')
        for bar in bars2:
            height = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')
        
        # Accuracy by class
        class_accuracies = {}
        for class_name in class_names:
            class_results = [r for r in results if r['ground_truth'] == class_name]
            if class_results:
                correct_count = sum(1 for r in class_results if r['correct'])
                class_accuracies[class_name] = correct_count / len(class_results)
            else:
                class_accuracies[class_name] = 0
        
        bars3 = axes[1, 0].bar(class_accuracies.keys(), class_accuracies.values(), color='lightgreen')
        axes[1, 0].set_title('Accuracy by Class')
        axes[1, 0].set_xlabel('Animal Class')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].set_ylim(0, 1)
        for bar, acc in zip(bars3, class_accuracies.values()):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{acc:.2f}', ha='center', va='bottom')
        
        # Overall accuracy
        correct_count = sum(1 for r in results if r['correct'])
        incorrect_count = len(results) - correct_count
        overall_accuracy = correct_count / len(results)
        
        bars4 = axes[1, 1].bar(['Correct', 'Incorrect'], [correct_count, incorrect_count], 
                              color=['green', 'red'], alpha=0.7)
        axes[1, 1].set_title(f'Overall Accuracy: {overall_accuracy:.2f} ({overall_accuracy*100:.1f}%)')
        axes[1, 1].set_ylabel('Number of Classifications')
        for bar in bars4:
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')
    else:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Predicted classifications
        bars1 = axes[0].bar(class_counts.keys(), class_counts.values())
        axes[0].set_title('Human Face Classifications')
        axes[0].set_xlabel('Animal Class')
        axes[0].set_ylabel('Number of Human Faces')
        for bar in bars1:
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')
        
        # Average probabilities
        if len(results) > 0:
            avg_probs = []
            for class_name in class_names:
                class_results = [r for r in results if r['predicted_class'] == class_name]
                if class_results:
                    avg_conf = sum(r['confidence'] for r in class_results) / len(class_results)
                    avg_probs.append(avg_conf)
                else:
                    avg_probs.append(0)
            
            bars2 = axes[1].bar(class_names, avg_probs)
            axes[1].set_title('Average Classification Confidence')
            axes[1].set_xlabel('Animal Class')
            axes[1].set_ylabel('Average Confidence')
            for bar, prob in zip(bars2, avg_probs):
                axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                        f'{prob:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    if results_dir:
        plt.savefig(results_dir / 'human_face_classifications.png', dpi=300, bbox_inches='tight')
    plt.show()

def save_models(pca, svm_model, scaler, class_names, metadata, timestamp_models_dir):
    """Save trained models and metadata"""
    print(f"\n=== Saving Models ===")
    
    joblib.dump(pca, timestamp_models_dir / 'pca_model.pkl')
    joblib.dump(svm_model, timestamp_models_dir / 'svm_model.pkl')
    joblib.dump(scaler, timestamp_models_dir / 'scaler.pkl')
    
    # Save metadata
    timestamp = timestamp_models_dir.name
    metadata_dict = {
        'timestamp': timestamp,
        'class_names': class_names,
        'pca_components': pca.n_components_,
        'pca_explained_variance': float(pca.explained_variance_ratio_.sum()),
        'svm_params': svm_model.get_params(),
        'model_files': {
            'pca': 'pca_model.pkl',
            'svm': 'svm_model.pkl',
            'scaler': 'scaler.pkl'
        },
        **metadata
    }
    
    import json
    with open(timestamp_models_dir / 'model_metadata.json', 'w') as f:
        json.dump(metadata_dict, f, indent=2)
    
    print(f"Models saved in: {timestamp_models_dir}")
    return timestamp, timestamp_models_dir

def load_existing_models(model_dir):
    """Load existing trained models from specified directory"""
    model_path = Path(model_dir)
    if not model_path.exists():
        raise FileNotFoundError(f"Model directory not found: {model_path}")
    
    import json
    with open(model_path / 'model_metadata.json', 'r') as f:
        metadata = json.load(f)
    
    print(f"Loading existing models from: {model_path}")
    print(f"Model timestamp: {metadata['timestamp']}")
    print(f"Model accuracy: {metadata['test_accuracy']:.4f}")
    
    pca = joblib.load(model_path / 'pca_model.pkl')
    svm_model = joblib.load(model_path / 'svm_model.pkl')
    scaler = joblib.load(model_path / 'scaler.pkl')
    
    print("Models loaded successfully!")
    return pca, svm_model, scaler, metadata

def main():
    """Main function to run the complete pipeline"""
    parser = argparse.ArgumentParser(description='Human Face to Animal Classification')
    parser.add_argument('--use-existing', action='store_true', 
                       help='Use existing trained models (interactive mode)')
    parser.add_argument('--model-dir', type=str, 
                       help='Path to specific model directory to use')
    parser.add_argument('--no-interaction', action='store_true',
                       help='Run without user interaction (use defaults)')
    
    args = parser.parse_args()
    
    print("=== Human Face to Animal Classification ===")
    print("Training on animal faces (cat, dog, wild)")
    print("Evaluating human faces for animal-like features")
    print()
    
    # Determine model usage strategy
    use_existing = False
    model_dir = None
    
    if args.model_dir:
        model_dir = Path(args.model_dir)
        if model_dir.exists():
            use_existing = True
            print(f"Using specified model directory: {model_dir}")
        else:
            print(f"Error: Specified model directory not found: {model_dir}")
            return
    elif args.use_existing:
        existing_model_dir = Path('models/20250712_000423')
        if existing_model_dir.exists():
            model_dir = existing_model_dir
            use_existing = True
            print(f"Using existing model directory: {model_dir}")
        else:
            print("No existing models found. Will train new model.")
    else:
        existing_model_dir = Path('models/20250712_000423')
        if existing_model_dir.exists() and not args.no_interaction:
            print(f"Found existing trained models in: {existing_model_dir}")
            response = input("Do you want to use existing models? (y/n): ").lower().strip()
            if response in ['y', 'yes']:
                use_existing = True
                model_dir = existing_model_dir
    
    # Create timestamp-based directory structure for results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    models_dir = Path('models')
    results_dir = Path('results')
    models_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)
    
    timestamp_results_dir = results_dir / timestamp
    timestamp_results_dir.mkdir(exist_ok=True)
    
    print(f"Results will be saved in: {timestamp_results_dir}")
    print()
    
    try:
        if use_existing and model_dir:
            # Load existing models
            pca, svm_model, scaler, metadata = load_existing_models(model_dir)
            class_names = metadata['class_names']
            
            print("Skipping training - using existing models")
            print(f"Model performance: {metadata['test_accuracy']:.4f} accuracy")
            
        else:
            # Train new models
            timestamp_models_dir = models_dir / timestamp
            timestamp_models_dir.mkdir(exist_ok=True)
            print(f"Models will be saved in: {timestamp_models_dir}")
            
            # Load animal training data
            X, y, class_names = imgLoad()
            if len(X) == 0:
                print("No animal images loaded. Please check your data directory.")
                return
            
            # Perform PCA analysis
            X_pca, pca = perform_pca_analysis(X, y, class_names, results_dir=timestamp_results_dir)
            
            # Visualize PCA in 2D
            visualize_pca_2d(X_pca, y, class_names, results_dir=timestamp_results_dir)
            
            # Visualize SVM decision boundary
            visualize_svm_decision_boundary(X, y, class_names, results_dir=timestamp_results_dir)
            
            # Split animal data for training and validation
            X_train, X_test, y_train, y_test = train_test_split(
                X_pca, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Train SVM classifier on animal data
            svm_model, y_pred, scaler = train_svm_classifier(
                X_train, X_test, y_train, y_test, class_names, results_dir=timestamp_results_dir
            )
            
            # Prepare metadata
            metadata = {
                'total_animal_images': len(X),
                'train_size': len(X_train),
                'test_size': len(X_test),
                'test_accuracy': float(accuracy_score(y_test, y_pred)),
                'image_size': (128, 128),
                'feature_dim_original': X.shape[1],
                'feature_dim_pca': X_pca.shape[1],
                'approach': 'human_to_animal_classification'
            }
            
            # Save models
            save_models(pca, svm_model, scaler, class_names, metadata, timestamp_models_dir)
        
        # Load human faces for evaluation
        X_human, y_human, human_files, class_names_human = load_human_faces()
        
        if len(X_human) == 0:
            print("No human faces found for evaluation.")
            return
        
        # Evaluate human faces
        human_results = evaluate_human_faces(
            pca, svm_model, scaler, class_names_human, X_human, human_files, 
            results_dir=timestamp_results_dir, y_human=y_human
        )
        
        print("\n=== Pipeline Complete ===")
        if use_existing:
            print("Used existing trained models successfully!")
        else:
            print("Animal classification model trained successfully!")
        print("Human faces evaluated for animal-like features.")
        print(f"Check the '{timestamp_results_dir}' directory for visualizations.")
        
        if human_results:
            print(f"\nHuman Face Summary:")
            print(f"Total human faces evaluated: {len(human_results)}")
            
            if y_human is not None and len(y_human) > 0:
                correct_predictions = sum(1 for r in human_results if r['correct'])
                accuracy = correct_predictions / len(human_results)
                print(f"Human face classification accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            
            class_summary = {}
            for result in human_results:
                animal_class = result['predicted_class']
                class_summary[animal_class] = class_summary.get(animal_class, 0) + 1
            
            for animal_class, count in class_summary.items():
                percentage = (count / len(human_results)) * 100
                print(f"  {animal_class}-like: {count} faces ({percentage:.1f}%)")
        
    except Exception as e:
        print(f"Error in main pipeline: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 