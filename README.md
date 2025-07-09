# A Computational Classification of Human Facial Traits
## Which Animal Are You? 🐱🐶🦊🐅

**Assignment 2 - International Fusion Science Course**  
**Dr. Suyong Eum**  
**Osaka University**

---

## Overview

This project uses Principal Component Analysis (PCA) and Support Vector Machine (SVM) to classify human faces based on their resemblance to specific animals (cat, dog, fox, tiger). The system analyzes facial features to determine which animal a person's face most closely resembles.

### Features
- PCA-based feature extraction from facial images
- SVM classification with RBF kernel
- Visualization of PCA components and decision boundaries
- Model performance evaluation and optimization
- Support for new image classification

---

## Quick Start

### Prerequisites
- Python 3.8 or higher
- Git (for cloning the repository)

### Installation

#### Option 1: Automatic Setup (Recommended)

**For Windows:**
```batch
git clone <repository-url>
cd A_Computational_Classification_of_Human_Facial_Traits
setup_environment.bat
```

**For Linux/Mac:**
```bash
git clone <repository-url>
cd A_Computational_Classification_of_Human_Facial_Traits
chmod +x setup_environment.sh
./setup_environment.sh
```

#### Option 2: Manual Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd A_Computational_Classification_of_Human_Facial_Traits
```

2. Create virtual environment:
```bash
# Windows
python -m venv facial_classification_env
facial_classification_env\Scripts\activate

# Linux/Mac
python3 -m venv facial_classification_env
source facial_classification_env/bin/activate
```

3. Install dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Usage

### Basic Classification
```bash
# Activate environment (if not already activated)
# Windows: facial_classification_env\Scripts\activate
# Linux/Mac: source facial_classification_env/bin/activate

# Run the main program
python main.py
```

### Available Functions

The main script provides several functions:

- `imgLoad()`: Load and preprocess animal face images
- `figPCA()`: Perform PCA and optionally visualize components
- `xSVM()`: Train SVM classifier and evaluate performance
- `accTest()`: Test different hyperparameter combinations
- `figSVM()`: Visualize SVM decision boundaries
- `imgEva()`: Classify a new image

### Custom Image Classification

To classify your own image, replace `eva.jpg` with your image file, or modify the `imgEva()` function:

```python
imgEva('./path/to/your/image.jpg')
```

---

## Project Structure

```
A_Computational_Classification_of_Human_Facial_Traits/
├── af_data/                    # Dataset directory
│   ├── cat/                   # Cat face images
│   ├── dog/                   # Dog face images
│   ├── fox/                   # Fox face images
│   └── tiger/                 # Tiger face images
├── main.py                    # Main classification script
├── eva.jpg                    # Sample evaluation image
├── requirements.txt           # Python dependencies
├── setup_environment.bat      # Windows setup script
├── setup_environment.sh       # Linux/Mac setup script
├── improvement_plan.md        # Project improvement roadmap
└── README.md                  # This file
```

---

## Technical Details

### Algorithm
1. **Data Loading**: Images are loaded, converted to grayscale, and resized to 128x128 pixels
2. **Feature Extraction**: PCA reduces dimensionality from 16,384 to 110 components for the SVM classification pipeline by default. For visualization (e.g. using `figPCA(..., com=2)` or `figSVM()`), you can set PCA to 2 components to project data into 2D space for plotting.
3. **Classification**: SVM with RBF kernel classifies faces into 4 animal categories
4. **Evaluation**: Model performance is assessed using accuracy metrics

### Default Parameters
- Image size: 128x128 pixels
- PCA components: 110
- SVM C parameter: 1.5
- Kernel: RBF
- Test split: 20%

### Performance Optimization
Use the `accTest()` function to find optimal hyperparameters:

```python
# Test different PCA components (120-121) and C values (2-3)
accTest(lowN=120, highN=121, stepN=1, lowC=2, highC=3, stepC=1)
```

---

## Data Requirements

### Current Dataset
- Each animal category contains ~16 images
- Images are from Pixabay and Flickr
- Format: JPG, preprocessed to grayscale

### For Human Face Classification
To use this system for human faces (as intended in the assignment):
1. Collect human face images categorized by animal-like features
2. Organize in `af_data/human_faces/` with subdirectories:
   - `cat_like/`
   - `dog_like/`
   - `fox_like/`
   - `tiger_like/`

---

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError**: Ensure virtual environment is activated and dependencies are installed
2. **File not found errors**: Check that `af_data` directory exists with animal subdirectories
3. **Memory issues**: Reduce PCA components or image size for large datasets

### Getting Help
- Check the `improvement_plan.md` for detailed modification guidelines
- Ensure all dependencies are correctly installed
- Verify Python version (3.8+ required)

---

## Development

### Contributing
This is an academic assignment project. Team contributions should be documented as per assignment requirements.

### Team Roles
- **Project Lead**: Overall coordination and integration
- **Data Collector**: Dataset collection and preprocessing
- **Data Analyst**: PCA/SVM optimization and evaluation
- **App Developer**: Web application development (bonus)
- **Report Writer**: Academic paper preparation
- **Presenter**: Conduct the presentation

---

## Academic Requirements

### Deliverables
- [x] Functional PCA+SVM classification system
- [ ] 4-page academic paper
- [ ] Progress presentations (July 10/17)
- [ ] Final submission (July 24)
- [ ] Web application (bonus: +10%)

### Submission
- Email: suyong@ist.osaka-u.ac.jp
- Subject: "G[X]-assignment2" (replace [X] with group number)
- Include: Code files + Academic report

---

## License

This project is for academic purposes as part of classes at Graduated Osaka University.

## Conventional Commits

The following is a list of common commit types used as the `type` in `git commit -m "type(scope): message"`.

- `feat`: a new feature
- `fix`: a bug fix
- `docs`: documentation changes (README, comments, etc.)
- `style`: code formatting changes (indentation, semicolons, etc.) that do not affect functionality
- `refactor`: code refactoring (improving code structure without changing functionality)
- `perf`: performance improvements
- `test`: adding or modifying tests
- `chore`: chores, build tasks, or auxiliary tool updates not directly related to source code or tests
- `build`: changes to the build system or external dependencies (npm, webpack, Gradle, etc.)
- `ci`: changes to CI configuration files (GitHub Actions, Travis CI, etc.)
- `revert`: reverting a previous commit