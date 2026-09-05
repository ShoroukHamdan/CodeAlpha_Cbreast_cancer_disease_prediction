# 🩺 Disease Prediction from Medical Data — Breast Cancer Classification

Predicting whether a breast tumor is **Malignant** or **Benign** from diagnostic measurements, using classical and ensemble machine learning models.

---

## 📌 Overview

| | |
|---|---|
| **Objective** | Predict the possibility of disease (breast cancer) from structured medical data |
| **Dataset** | Breast Cancer Wisconsin (Diagnostic) — UCI ML Repository (via `sklearn.datasets`) |
| **Task type** | Binary classification (Malignant = 0, Benign = 1) |
| **Samples / Features** | 569 samples × 30 numeric features |
| **Algorithms** | Logistic Regression, SVM (RBF kernel), Random Forest, XGBoost |
| **Notebook** | `breast_cancer_disease_prediction.ipynb` |

---

## 🧬 Dataset

The dataset contains digitized measurements from fine needle aspirate (FNA) images of breast masses. For each cell nucleus, 10 real-valued features are computed (radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension), each reported as **mean**, **standard error**, and **"worst"** (largest) value — giving 30 features in total.

- **Target classes:** `malignant` (212 samples) and `benign` (357 samples)
- **Missing values:** none
- **Source:** identical data to the UCI Breast Cancer Wisconsin (Diagnostic) dataset, loaded directly via `sklearn.datasets.load_breast_cancer()` for reproducibility (no external download needed)

---

## ⚙️ Pipeline

1. **Exploratory Data Analysis (EDA)**
   - Class balance check, summary statistics, missing-value check
   - Correlation heatmap of key mean features
   - Feature distributions by diagnosis (KDE plots)
2. **Preprocessing**
   - Stratified train/test split (80% / 20%)
   - Feature scaling with `StandardScaler`
3. **Modeling** — four classifiers trained and compared:
   - Logistic Regression
   - Support Vector Machine (RBF kernel)
   - Random Forest
   - XGBoost
4. **Evaluation**
   - Accuracy, Precision, Recall, F1-Score, ROC-AUC
   - Confusion matrices per model
   - ROC curves comparison
   - 5-fold stratified cross-validation (F1 scoring)
5. **Interpretability**
   - Top 10 feature importances (Random Forest & XGBoost)
6. **Inference demo**
   - Single-patient prediction example with per-model probability output

---

## 📊 Results

### Test set performance (held-out 20%)

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** | **0.982** | **0.986** | **0.986** | **0.986** | **0.995** |
| SVM (RBF Kernel) | 0.982 | 0.986 | 0.986 | 0.986 | 0.995 |
| XGBoost | 0.956 | 0.947 | 0.986 | 0.966 | 0.993 |
| Random Forest | ~0.96 | ~0.95 | ~0.98 | ~0.96 | ~0.99 |

### 5-fold cross-validation (F1 score)

| Model | CV F1 Mean | CV F1 Std |
|---|---|---|
| SVM (RBF Kernel) | 0.982 | 0.013 |
| Logistic Regression | 0.979 | 0.013 |
| XGBoost | 0.972 | 0.008 |
| Random Forest | 0.962 | 0.010 |

**Best overall model:** Logistic Regression (by F1-Score), closely matched by SVM — both linear/margin-based models slightly outperform the tree ensembles on this dataset, likely because the classes are close to linearly separable after scaling.

> Note: exact values may vary slightly by run due to random initialization in Random Forest/XGBoost.

---

## 🗂️ Project Structure

```
.
├── breast_cancer_disease_prediction.ipynb   # Main notebook (EDA → training → evaluation)
└── README.md                                # This file
```

---

## 🚀 Getting Started

### Requirements

```
python >= 3.9
numpy
pandas
matplotlib
seaborn
scikit-learn
xgboost
jupyter
```

### Installation

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost jupyter
```

### Run

```bash
jupyter notebook breast_cancer_disease_prediction.ipynb
```

Run all cells top to bottom — the dataset loads automatically via `sklearn`, no manual download required.

---

## 🔍 Key Insights

- Features like **worst concave points**, **worst perimeter**, and **worst radius** are consistently among the most predictive of malignancy.
- Simple linear models (Logistic Regression, SVM) perform competitively with — and here slightly better than — ensemble methods, showing the value of always benchmarking a linear baseline.
- Cross-validation standard deviations are small (≤ 0.013) across all models, indicating stable performance rather than a lucky train/test split.

---

## ⚠️ Disclaimer

This project is for **educational and research purposes only**. It is not a certified diagnostic tool and must not be used for real clinical decision-making. Any real-world medical application requires validation by qualified healthcare professionals and regulatory approval.

---

## 📄 License

This project is released under the MIT License. The dataset is publicly available via the UCI Machine Learning Repository / scikit-learn.
