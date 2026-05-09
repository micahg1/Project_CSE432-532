# CSE432/532 — Speech Emotion Recognition Project

## Project Structure

```
Project_CSE432-532/
├── SER_Project/
│   ├── data/                        # Downloaded RAVDESS audio + generated CSVs
│   │   ├── Audio_Speech_Actors_01-24/   # Extracted from Zenodo zip
│   │   ├── Audio_Song_Actors_01-24/     # Extracted from Zenodo zip
│   │   ├── metadata.csv             # Generated in Week 4 (filename parsing)
│   │   └── features.csv             # Generated in Week 5 (audio feature vectors)
│   │
│   ├── minilearn/                   # From-scratch ML library (implements sklearn API)
│   │   ├── __init__.py              # Package init, exports all submodules
│   │   ├── preprocessing.py         # StandardScaler, train_test_split
│   │   ├── classifiers.py           # LogisticRegression, KNN, NaiveBayes, LinearSVM, DecisionTree
│   │   ├── metrics.py               # accuracy, precision, recall, F1, confusion_matrix, AUC
│   │   ├── validation.py            # StratifiedKFold, cross_val_score, GridSearchCV
│   │   ├── clustering.py            # KMeans (Lloyd's algorithm)
│   │   ├── decomposition.py         # PCA
│   │   ├── regression.py            # LinearRegression, RidgeRegression, MSE, R²
│   │   └── ann.py                   # MLP (feedforward, backprop, mini-batch SGD)
│   │
│   └── notebooks/                   # Weekly analysis notebooks
│       ├── week4_data_exploration.ipynb        # Filename parsing, EDA, waveforms
│       ├── week5_feature_extraction.ipynb      # MFCC, chroma, ZCR, spectral features → CSV
│       ├── week6_regression.ipynb              # Linear/Ridge regression on audio features
│       ├── week7_classification.ipynb          # LR, Naive Bayes, KNN — MiniLearn vs sklearn
│       ├── week8_svm.ipynb                     # SVM with linear/RBF/poly kernels
│       ├── week9_decision_trees.ipynb          # CART, Random Forest, AdaBoost
│       ├── week10_model_validation.ipynb       # Stratified k-fold CV, hyperparameter tuning
│       ├── week11_clustering.ipynb             # K-Means, ARI/NMI, PCA/t-SNE visualization
│       ├── week12_dimensionality_reduction.ipynb  # PCA, scree plot, classifiers on PCA features
│       └── week13_14_ann.ipynb                 # MiniLearn MLP + Keras/PyTorch DL model
│
├── Papers/                          # Reference papers (RAVDESS, etc.)
├── .gitignore
├── README.md
└── setup_virtual_environment.md
```

## Quick Start

```bash
# 1. Set up virtual environment (see setup_virtual_environment.md)
pip install -r requirements.txt

# 2. Download RAVDESS data from https://zenodo.org/records/1188976
#    Extract into SER_Project/data/

# 3. Work through notebooks in order (week4 → week5 → ... → week13_14)
```

## MiniLearn Usage

```python
from minilearn.classifiers import LogisticRegression, KNearestNeighbors
from minilearn.preprocessing import StandardScaler, train_test_split
from minilearn.metrics import accuracy, f1_score, confusion_matrix
from minilearn.validation import StratifiedKFold, cross_val_score

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit on train only!
X_test_scaled  = scaler.transform(X_test)

clf = LogisticRegression(lr=0.01, n_iter=500)
clf.fit(X_train_scaled, y_train)
print(clf.score(X_test_scaled, y_test))
```

## Dataset

RAVDESS — Ryerson Audio-Visual Database of Emotional Speech and Song  
Citation: Livingstone SR, Russo FA (2018). PLoS ONE 13(5): e0196391.  
Download: https://zenodo.org/records/1188976 (audio-only zip files)

**8 emotion classes:** neutral, calm, happy, sad, angry, fearful, disgust, surprised
