import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import cv2
from pathlib import Path
from datetime import datetime
import warnings
import joblib
import argparse
import json
from typing import Tuple, List, Dict, Optional, Any
import sys
from matplotlib.ticker import MaxNLocator

warnings.filterwarnings('ignore')

# Configuration
class Config:
    """Configuration parameters for the animal face classification system"""
    
    # Data parameters
    DATA_DIR = "data/af_data_new"
    HUMAN_DATA_SUBDIR = "human_like_animal"
    IMAGE_SIZE = (128, 128)
    ANIMAL_CLASSES = ['cat', 'dog', 'tiger']
    
    # Model parameters
    PCA_COMPONENTS = 110
    TEST_SIZE = 0.15
    VAL_SIZE = 0.1765
    RANDOM_STATE = 42
    
    # Grid search parameters
    PARAM_GRID = {
        'svc__C': [0.1, 1, 10],
        'svc__gamma': ['scale', 0.01, 0.1],
        'svc__kernel': ['rbf']
    }
    
    # Visualization parameters
    COLORS = ['red', 'green', 'blue']
    PLOT_STYLE = 'seaborn-v0_8'
    DPI = 300
    
    # Directory structure
    MODELS_DIR = "models"
    RESULTS_DIR = "results"


class Tee:
    """
    Write the same output to multiple file-like objects (e.g. console & log file)
    """
    def __init__(self, *files):
        self.files = files

    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush()

    def flush(self):            # for compatibility with sys.stdout
        for f in self.files:
            f.flush()


def setup_plotting():
    """Setup plotting configuration"""
    plt.style.use(Config.PLOT_STYLE)
    sns.set_palette("husl")


def create_directories(timestamp: str) -> Tuple[Path, Path]:
    """Create necessary directories for models and results
    
    Args:
        timestamp: Current timestamp for unique directory naming
        
    Returns:
        Tuple of (models_dir, results_dir)
    """
    models_root = Path(Config.MODELS_DIR)
    results_root = Path(Config.RESULTS_DIR)
    models_root.mkdir(exist_ok=True)
    results_root.mkdir(exist_ok=True)
    
    results_dir = results_root / timestamp
    results_dir.mkdir(exist_ok=True)
    
    return models_root, results_dir


def load_and_preprocess_image(img_path: Path) -> Optional[np.ndarray]:
    """Load and preprocess a single image
    
    Args:
        img_path: Path to the image file
        
    Returns:
        Flattened and normalized grayscale image array, or None if failed
    """
    try:
        img = cv2.imread(str(img_path))
        if img is None:
            return None
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, Config.IMAGE_SIZE)
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return img_gray.flatten() / 255.0
    except Exception:
        return None


def load_animal_data() -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load animal image data (3 classes: cat, dog, tiger)
    
    Returns:
        Tuple of (features, labels, class_names)
    """
    print("=== Data Collection and Preprocessing ===")
    print(f"Data source: {Config.DATA_DIR}")
    print(f"Preprocessing: RGB→Grayscale, {Config.IMAGE_SIZE} resize, normalization (0-1)")
    
    data_path = Path(Config.DATA_DIR)
    X, y = [], []
    
    for class_idx, class_name in enumerate(Config.ANIMAL_CLASSES):
        class_dir = data_path / class_name
        if not class_dir.exists():
            continue
            
        image_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
        print(f"{class_name}: {len(image_files)} images")
        
        for img_path in image_files:
            img_data = load_and_preprocess_image(img_path)
            if img_data is not None:
                X.append(img_data)
                y.append(class_idx)
    
    print(f"Total data count: {len(X)} images")
    return np.array(X), np.array(y), Config.ANIMAL_CLASSES


def load_human_data() -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load human image data with animal-like features
    
    Returns:
        Tuple of (features, labels, filenames)
    """
    print("\n=== Human Image Data Loading ===")
    print("Collection method: AI image generation services")
    print("Labeling method: Filename-based (e.g., cat_human_01.jpg)")
    
    data_path = Path(Config.DATA_DIR) / Config.HUMAN_DATA_SUBDIR
    if not data_path.exists():
        print("Human data directory not found")
        return np.array([]), np.array([]), []
    
    X_human, y_human, human_files = [], [], []
    class_mapping = {name: idx for idx, name in enumerate(Config.ANIMAL_CLASSES)}
    
    image_files = list(data_path.glob("*.jpg")) + list(data_path.glob("*.png"))
    print(f"Human image count: {len(image_files)}")
    
    for img_path in image_files:
        filename = img_path.name
        animal_name = filename.split('_')[0].lower()
        
        if animal_name not in class_mapping:
            continue
            
        img_data = load_and_preprocess_image(img_path)
        if img_data is not None:
            X_human.append(img_data)
            y_human.append(class_mapping[animal_name])
            human_files.append(filename)
    
    # Print distribution
    for class_name in Config.ANIMAL_CLASSES:
        count = sum(1 for f in human_files if f.startswith(class_name))
        print(f"{class_name}-like: {count} images")
    
    return np.array(X_human), np.array(y_human), human_files


def create_model_pipeline() -> Pipeline:
    """Create the ML pipeline with preprocessing and model components
    
    Returns:
        Configured sklearn Pipeline
    """
    return Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=Config.PCA_COMPONENTS)),
        ('svc', SVC())
    ])


def train_model_with_validation(
        X: np.ndarray,
        y: np.ndarray,
        results_dir: Optional[Path] = None
) -> Tuple[Pipeline, Dict[str, float]]:
    """Train model with proper train/validation/test split and validation
    
    Args:
        X: Feature matrix
        y: Labels
        
    Returns:
        Tuple of (trained_pipeline, metrics_dict)
    """
    print("\n=== Model Training with Validation ===")
    
    # Split data
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=Config.TEST_SIZE, stratify=y, random_state=Config.RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=Config.VAL_SIZE,
        stratify=y_train_full, random_state=Config.RANDOM_STATE
    )
    
    print(f"Data split → Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Create and train pipeline
    pipeline = create_model_pipeline()
    grid_search = GridSearchCV(
        pipeline, Config.PARAM_GRID, cv=5, scoring='accuracy', n_jobs=-1
    )
    
    print("Running hyperparameter optimization...")
    grid_search.fit(X_train, y_train)
    
    # Validation
    best_cv = grid_search.best_score_
    std_cv = grid_search.cv_results_['std_test_score'][grid_search.best_index_]
    threshold = best_cv - std_cv
    
    print(f"Best CV accuracy: {best_cv:.4f} ± {std_cv:.4f}")
    print(f"Best parameters: {grid_search.best_params_}")
    
    val_score = grid_search.best_estimator_.score(X_val, y_val)
    print(f"Validation accuracy: {val_score:.4f} (threshold: {threshold:.4f})")
    
    metrics = {
        'cv_score': best_cv,
        'cv_std': std_cv,
        'val_score': val_score,
        'test_score': None,
        'train_size': len(X_train),             
        'val_size'  : len(X_val),               
        'test_size' : len(X_test),              
        'best_params': grid_search.best_params_ 
    }
    
    # Test evaluation if validation passes
    if val_score >= threshold:
        print("✅ Validation passed. Evaluating on test set...")
        test_score = grid_search.best_estimator_.score(X_test, y_test)
        metrics['test_score'] = test_score
        print(f"Test accuracy: {test_score:.4f}")
        
        # Detailed test evaluation
        y_pred_test = grid_search.best_estimator_.predict(X_test)
        print("\nTest Set Classification Report:")
        report_str = classification_report(
            y_test, y_pred_test, target_names=Config.ANIMAL_CLASSES
        )
        print(report_str)

        # === 追加: 結果をファイルへ保存 ===
        if results_dir is not None:
            # ① 混同行列
            plot_confusion_matrix(
                y_test, y_pred_test,
                'Animal Test Set',
                results_dir / 'animal_confusion_matrix.png',
                Config.ANIMAL_CLASSES
            )

            # ② 分類分布バーグラフ
            class_counts = {cls: 0 for cls in Config.ANIMAL_CLASSES}
            for pred in y_pred_test:
                class_counts[Config.ANIMAL_CLASSES[pred]] += 1

            plt.figure(figsize=(10, 6))
            bars = plt.bar(class_counts.keys(), class_counts.values(),
                           color=Config.COLORS)
            plt.title('Animal Test Classification Distribution')
            plt.xlabel('Animal Class')
            plt.ylabel('Number of Images')
            for bar, count in zip(bars, class_counts.values()):
                plt.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                         f'{count}', ha='center', va='bottom')
            plt.savefig(results_dir / 'animal_classification_summary.png',
                        dpi=Config.DPI, bbox_inches='tight')
            plt.close()

            # ③ classification report
            with open(results_dir / 'animal_classification_report.txt',
                      'w', encoding='utf-8') as f:
                f.write(report_str)
    else:
        print("❌ Validation failed. Consider hyperparameter re-tuning.")
    
    return grid_search.best_estimator_, metrics


def visualize_pca_analysis(X: np.ndarray, y: np.ndarray, pca: PCA, results_dir: Path):
    """Visualize PCA analysis results
    
    Args:
        X: Original feature matrix
        y: Labels
        pca: Fitted PCA model
        results_dir: Directory to save plots
    """
    print(f"\n=== PCA Analysis ===")
    print(f"Components: {pca.n_components_}")
    print(f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")
    
    X_pca = pca.transform(X)
    
    # === New: PCA explained‐variance plots ==========================
    plt.figure(figsize=(12, 4))

    # 1) 累積寄与率 ---------------------------------------------------
    plt.subplot(1, 2, 1)
    x_vals = np.arange(1, len(pca.explained_variance_ratio_) + 1)
    plt.plot(x_vals, np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title('PCA Explained Variance')
    plt.grid(True)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    # 2) 各主成分の寄与率（上位 20）----------------------------------
    plt.subplot(1, 2, 2)
    top_k = min(20, len(pca.explained_variance_ratio_))
    plt.bar(range(1, top_k + 1), pca.explained_variance_ratio_[:top_k])
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title(f'Top {top_k} Components')
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()

    # 画像として保存
    plt.savefig(results_dir / 'pca_analysis.png', dpi=Config.DPI, bbox_inches='tight')
    plt.show()
    # ===============================================================

    # Eigenfaces visualization
    plt.figure(figsize=(12, 4))
    for i in range(min(6, pca.n_components_)):
        plt.subplot(2, 3, i + 1)
        eigenface = pca.components_[i].reshape(*Config.IMAGE_SIZE)
        plt.imshow(eigenface, cmap='gray')
        plt.title(f'Eigenface {i+1}')
        plt.axis('off')
    plt.suptitle('PCA Eigenfaces (Top 6 Components)')
    plt.tight_layout()
    plt.savefig(results_dir / 'eigenfaces.png', dpi=Config.DPI, bbox_inches='tight')
    plt.show()
    
    # 2D scatter plot
    plt.figure(figsize=(10, 8))
    for i, class_name in enumerate(Config.ANIMAL_CLASSES):
        mask = y == i
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=Config.COLORS[i], label=class_name, alpha=0.6, s=50)
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title('PCA 2D Visualization')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(results_dir / 'pca_2d_visualization.png', dpi=Config.DPI, bbox_inches='tight')
    plt.show()


def visualize_svm_decision_boundary(X: np.ndarray, y: np.ndarray, pipeline: Pipeline, results_dir: Path):
    """Visualize the SVM decision boundary using the already-trained *pipeline*.

    The data are first projected onto the first two principal components that were
    learned during training.  A prediction grid is then built directly in this
    2-D PCA space (all remaining PCA dimensions are set to zero) and fed to the
    trained SVM classifier.  This avoids re-training another model and therefore
    guarantees that the visualisation reflects the model that is actually used
    for inference.

    Args:
        X: Original feature matrix (flattened images)
        y: Ground-truth labels corresponding to *X*
        pipeline: Trained ``sklearn`` pipeline returned by ``train_model_with_validation``
        results_dir: Directory where the plot will be saved
    """
    print("\n=== SVM Decision Boundary Visualization ===")

    # ------------------------------------------------------------------
    # Extract the fitted components from the pipeline
    # ------------------------------------------------------------------
    scaler = pipeline.named_steps['scaler']
    pca_full = pipeline.named_steps['pca']
    svc = pipeline.named_steps['svc']

    # Project the *original* data into PCA space (after scaling)
    X_scaled = scaler.transform(X)
    X_pca = pca_full.transform(X_scaled)  # shape -> (n_samples, n_components)

    # Use only the first two principal components for visualisation
    X_2d = X_pca[:, :2]

    # ------------------------------------------------------------------
    # Build a mesh grid in the 2-D PCA space
    # ------------------------------------------------------------------
    margin = 0.5
    x_min, x_max = X_2d[:, 0].min() - margin, X_2d[:, 0].max() + margin
    y_min, y_max = X_2d[:, 1].min() - margin, X_2d[:, 1].max() + margin

    h = 0.02  # step size in the mesh
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))

    # Flatten the grid and embed it back into the *full* PCA space expected by SVC
    mesh_2d = np.c_[xx.ravel(), yy.ravel()]  # (N, 2)
    n_components = pca_full.n_components_
    mesh_full = np.zeros((mesh_2d.shape[0], n_components))
    mesh_full[:, :2] = mesh_2d  # fill first two components, keep others at zero

    # ------------------------------------------------------------------
    # Predict class for each grid point using the trained SVC (operates in PCA space)
    # ------------------------------------------------------------------
    Z = svc.predict(mesh_full)
    Z = Z.reshape(xx.shape)

    # ------------------------------------------------------------------
    # Plot the decision regions together with the training samples
    # ------------------------------------------------------------------
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)

    for i, class_name in enumerate(Config.ANIMAL_CLASSES):
        mask = y == i
        plt.scatter(X_2d[mask, 0], X_2d[mask, 1],
                    c=Config.COLORS[i], label=class_name, edgecolor='k', alpha=0.8)

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title('SVM Decision Boundary (2D PCA Space)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(results_dir / 'svm_decision_boundary.png', dpi=Config.DPI, bbox_inches='tight')
    plt.show()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                         title: str, save_path: Path, class_names: List[str]):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(save_path, dpi=Config.DPI, bbox_inches='tight')
    plt.show()

# ---------------------------------------------------------------------
# New: Animal test-set evaluation (always executed, even for loaded model)
# ---------------------------------------------------------------------
def evaluate_animal_test_set(pipeline: Pipeline,
                             X: np.ndarray,
                             y: np.ndarray,
                             results_dir: Path) -> None:
    """
    Evaluate the pipeline on a deterministic animal test split and
    save confusion matrix, distribution plot, and classification report.
    """
    if len(X) == 0:
        print("No animal data available for evaluation.")
        return

    print("\n=== Animal Test Set Evaluation ===")

    # Recreate the same test split used during training
    _, X_test, _, y_test = train_test_split(
        X, y,
        test_size=Config.TEST_SIZE,
        stratify=y,
        random_state=Config.RANDOM_STATE
    )

    y_pred = pipeline.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy (re-evaluated): {test_acc:.4f}")

    # Classification report
    report_str = classification_report(
        y_test, y_pred, target_names=Config.ANIMAL_CLASSES
    )
    print("\nClassification Report (Animal Test):")
    print(report_str)

    # ① Confusion matrix
    plot_confusion_matrix(
        y_test, y_pred,
        'Animal Test Set',
        results_dir / 'animal_confusion_matrix.png',
        Config.ANIMAL_CLASSES
    )

    # ② Distribution bar graph
    class_counts = {cls: 0 for cls in Config.ANIMAL_CLASSES}
    for pred in y_pred:
        class_counts[Config.ANIMAL_CLASSES[pred]] += 1

    plt.figure(figsize=(10, 6))
    bars = plt.bar(class_counts.keys(), class_counts.values(),
                   color=Config.COLORS)
    plt.title('Animal Test Classification Distribution')
    plt.xlabel('Animal Class')
    plt.ylabel('Number of Images')
    for bar, count in zip(bars, class_counts.values()):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                 f'{count}', ha='center', va='bottom')
    plt.savefig(results_dir / 'animal_classification_summary.png',
                dpi=Config.DPI, bbox_inches='tight')
    plt.close()

    # ③ classification report txt
    with open(results_dir / 'animal_classification_report.txt',
              'w', encoding='utf-8') as f:
        f.write(report_str)


def evaluate_human_faces(pipeline: Pipeline, X_human: np.ndarray, y_human: np.ndarray, 
                        human_files: List[str], results_dir: Path) -> List[Dict[str, Any]]:
    """Evaluate pipeline on human face images
    
    Args:
        pipeline: Trained ML pipeline
        X_human: Human image features
        y_human: Human image labels
        human_files: Human image filenames
        results_dir: Directory to save results
        
    Returns:
        List of classification results
    """
    if len(X_human) == 0:
        print("No human images found for evaluation")
        return []
    
    print("\n=== Human Face Evaluation ===")
    
    # Predict
    predictions = pipeline.predict(X_human)
    accuracy = accuracy_score(y_human, predictions)
    
    # Collect results
    results = []
    for i, (filename, pred) in enumerate(zip(human_files, predictions)):
        ground_truth = Config.ANIMAL_CLASSES[y_human[i]]
        predicted = Config.ANIMAL_CLASSES[pred]
        correct = pred == y_human[i]
        
        results.append({
            'filename': filename,
            'predicted': predicted,
            'ground_truth': ground_truth,
            'correct': correct
        })
        
        status = "✓" if correct else "✗"
        print(f"{filename}: {predicted} | GT: {ground_truth} {status}")
    
    print(f"\nHuman image classification accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_human, predictions, target_names=Config.ANIMAL_CLASSES))
    
    # Confusion matrix
    plot_confusion_matrix(y_human, predictions, 'Human Image Classification', 
                         results_dir / 'human_confusion_matrix.png', Config.ANIMAL_CLASSES)
    
    # Distribution plot
    class_counts = {cls: 0 for cls in Config.ANIMAL_CLASSES}
    for result in results:
        class_counts[result['predicted']] += 1
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(class_counts.keys(), class_counts.values(), color=Config.COLORS)
    plt.title('Human Image Classification Distribution')
    plt.xlabel('Animal Class')
    plt.ylabel('Number of Images')
    
    for bar, count in zip(bars, class_counts.values()):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{count}', ha='center', va='bottom')
    
    plt.savefig(results_dir / 'human_classification_summary.png', dpi=Config.DPI, bbox_inches='tight')
    plt.show()
    
    return results


def save_model_and_metadata(pipeline: Pipeline, metrics: Dict[str, float], 
                           timestamp: str, models_root: Path):
    """Save trained model and metadata"""
    model_dir = models_root / timestamp
    model_dir.mkdir(parents=True, exist_ok=True)

    # モデル保存
    joblib.dump(pipeline, model_dir / 'pipeline.pkl')

    # ---- 豊富なメタデータを収集 ----
    pca = pipeline.named_steps['pca']
    svc = pipeline.named_steps['svc']

    metadata = {
        'timestamp'             : timestamp,
        'class_names'           : Config.ANIMAL_CLASSES,
        'image_size'            : Config.IMAGE_SIZE,
        'feature_dim_original'  : Config.IMAGE_SIZE[0] * Config.IMAGE_SIZE[1],
        'feature_dim_pca'       : pca.n_components_,
        'pca_explained_variance': float(pca.explained_variance_ratio_.sum()),
        'svm_params'            : svc.get_params(),
        **metrics,                                # ← ここに train_size などが含まれる
        'total_images'          : metrics.get('train_size', 0)
                                 + metrics.get('val_size', 0)
                                 + metrics.get('test_size', 0),
        'model_file'            : 'pipeline.pkl'
    }

    with open(model_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Model saved to {model_dir}")


def load_model_and_metadata(model_dir: Path) -> Tuple[Pipeline, Dict[str, Any]]:
    """Load model and metadata from disk
    
    Args:
        model_dir: Directory containing the model
        
    Returns:
        Tuple of (pipeline, metadata)
    """
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    pipeline = joblib.load(model_dir / 'pipeline.pkl')
    
    with open(model_dir / 'metadata.json', 'r') as f:
        metadata = json.load(f)
    
    print(f"Loaded model from {model_dir}")
    if 'test_score' in metadata and metadata['test_score'] is not None:
        print(f"Model test accuracy: {metadata['test_score']:.4f}")
    
    return pipeline, metadata


def find_latest_model(models_root: Path) -> Optional[Path]:
    """Find the latest trained model
    
    Args:
        models_root: Root directory for models
        
    Returns:
        Path to latest model directory, or None if not found
    """
    candidates = [
        d for d in models_root.iterdir() 
        if d.is_dir() and (d / 'metadata.json').exists()
    ]
    
    if not candidates:
        return None
    
    return max(candidates, key=lambda p: p.name)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Animal Face Feature Human Classification System'
    )
    parser.add_argument(
        '--use-existing', 
        action='store_true',
        help='Use latest trained model instead of training new one'
    )
    parser.add_argument(
        '--model-dir', 
        type=str,
        help='Path to specific model directory to use'
    )
    parser.add_argument(
        '--skip-boundary', 
        action='store_true',
        help='Skip decision boundary visualization (faster execution)'
    )
    
    return parser.parse_args()


def main():
    """Main execution function"""
    # Setup
    setup_plotting()
    args = parse_arguments()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    models_root, results_dir = create_directories(timestamp)

    # ―――――――――――――――――――――――――――――――――――――――――――――――――――――――
    #  Tee 標準出力 / 標準エラー出力 → log ファイルにも保存
    # ―――――――――――――――――――――――――――――――――――――――――――――――――――――――
    log_file = open(results_dir / 'experiment_log.txt', 'w', encoding='utf-8')
    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)
    # ----------------------------------------------------------

    print(f"Results will be saved to: {results_dir}")
    
    # Load data
    X, y, class_names = load_animal_data()
    X_human, y_human, human_files = load_human_data()
    
    # Determine whether to use existing model or train new one
    pipeline = None
    
    if args.model_dir:
        model_dir = Path(args.model_dir)
        if model_dir.exists():
            pipeline, metadata = load_model_and_metadata(model_dir)
        else:
            print(f"Specified model directory not found: {model_dir}")
    
    elif args.use_existing:
        latest_model_dir = find_latest_model(models_root)
        if latest_model_dir:
            pipeline, metadata = load_model_and_metadata(latest_model_dir)
        else:
            print("No existing models found.")
    
    # Train new model if needed
    if pipeline is None:
        print("Training new model...")
        pipeline, metrics = train_model_with_validation(X, y, results_dir)
        save_model_and_metadata(pipeline, metrics, timestamp, models_root)
    
    # Visualizations
    pca = pipeline.named_steps['pca']
    visualize_pca_analysis(X, y, pca, results_dir)
    
    if not args.skip_boundary:
        visualize_svm_decision_boundary(X, y, pipeline, results_dir)
    
    # --- 新規: 動物テストセット評価（学習済みモデル読込時も必ず実行） ---
    evaluate_animal_test_set(pipeline, X, y, results_dir)
    # --------------------------------------------------------------------

    # Evaluate on human faces
    if len(X_human) > 0:
        human_results = evaluate_human_faces(
            pipeline, X_human, y_human, human_files, results_dir
        )
    
    print(f"\n=== Experiment Complete ===")
    print(f"All results saved to {results_dir}")


if __name__ == "__main__":
    main() 