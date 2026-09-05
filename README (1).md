# 🩺 Disease Prediction from Medical Data — Breast Cancer Classification

A machine learning classification project for predicting whether a breast tumor is **Malignant** or **Benign** using diagnostic measurements from the Breast Cancer Wisconsin (Diagnostic) dataset.

---

## 📌 Project Overview

|                    |                                                                      |
| ------------------ | -------------------------------------------------------------------- |
| **Objective**      | Classify breast tumors as Malignant or Benign using machine learning |
| **Dataset**        | Breast Cancer Wisconsin (Diagnostic) — UCI ML Repository             |
| **Task Type**      | Binary Classification                                                |
| **Target Classes** | Malignant = 0, Benign = 1                                            |
| **Samples**        | 569                                                                  |
| **Features**       | 30 numerical features                                                |
| **Algorithms**     | Logistic Regression, SVM (RBF), Random Forest, XGBoost               |
| **Evaluation**     | Accuracy, Precision, Recall, F1-Score, ROC-AUC, Cross-Validation     |
| **Notebook**       | `breast_cancer_disease_prediction.ipynb`                             |

---

## 🎯 Objective

The main objective of this project is to develop and compare multiple machine learning classification models for breast tumor diagnosis.

The models learn patterns from numerical diagnostic measurements and classify each tumor as:

* **Malignant (0)**
* **Benign (1)**

The project also evaluates model performance using multiple classification metrics and stratified cross-validation.

---

## 🧬 Dataset

The project uses the **Breast Cancer Wisconsin (Diagnostic)** dataset, originally provided through the **UCI Machine Learning Repository** and loaded using:

```python
sklearn.datasets.load_breast_cancer()
```

The dataset contains **569 samples** and **30 numerical features** derived from digitized images of fine needle aspirate (FNA) samples of breast masses.

The features describe characteristics of cell nuclei, including:

* Radius
* Texture
* Perimeter
* Area
* Smoothness
* Compactness
* Concavity
* Concave points
* Symmetry
* Fractal dimension

Each characteristic is represented using three measurements:

* Mean
* Standard Error
* Worst

This results in a total of **30 features**.

### Target Distribution

| Class     | Label | Samples |
| --------- | ----: | ------: |
| Malignant |     0 |     212 |
| Benign    |     1 |     357 |

### Dataset Properties

* **Total samples:** 569
* **Total features:** 30
* **Missing values:** None
* **Problem type:** Binary Classification

---

## 🔄 Machine Learning Pipeline

The project follows a complete machine learning workflow:

```text
Data Loading
     ↓
Data Exploration
     ↓
Exploratory Data Analysis (EDA)
     ↓
Data Preprocessing
     ↓
Train/Test Split
     ↓
Feature Scaling
     ↓
Model Training
     ↓
Model Evaluation
     ↓
Cross-Validation
     ↓
Model Comparison
     ↓
Feature Importance Analysis
     ↓
Single-Sample Prediction
```

---

## 🔍 Exploratory Data Analysis

The EDA stage includes:

* Dataset structure inspection
* Statistical summary
* Class distribution analysis
* Missing-value detection
* Feature distribution analysis
* Correlation analysis
* Diagnosis-based feature comparison
* Correlation heatmap
* Feature distribution visualizations

These steps help understand the dataset and identify patterns that may be useful for classification.

---

## ⚙️ Data Preprocessing

The following preprocessing steps were applied:

### 1. Train/Test Split

The dataset was divided into:

* **80% Training Data**
* **20% Testing Data**

A **stratified split** was used to preserve the original class distribution.

### 2. Feature Scaling

Numerical features were standardized using:

```python
StandardScaler
```

Scaling is particularly important for models such as Logistic Regression and SVM because they are sensitive to feature magnitude.

---

## 🤖 Machine Learning Models

Four classification algorithms were implemented and compared.

### 1. Logistic Regression

Used as a strong and interpretable baseline classification model.

### 2. Support Vector Machine — RBF Kernel

An SVM with an **RBF (Radial Basis Function) kernel** was used to capture nonlinear relationships between features.

### 3. Random Forest

An ensemble of decision trees that can model nonlinear relationships and provide feature importance estimates.

### 4. XGBoost

A gradient-boosting ensemble algorithm designed to build strong predictive models through sequential decision trees.

---

## 📊 Evaluation Metrics

The models were evaluated using:

### Accuracy

Measures the percentage of correctly classified samples.

### Precision

Measures how many samples predicted as positive are actually positive.

### Recall

Measures how many actual positive samples are correctly identified.

### F1-Score

The harmonic mean of Precision and Recall.

### ROC-AUC

Measures the model's ability to distinguish between the two classes across different classification thresholds.

### Cross-Validation

A **5-fold Stratified Cross-Validation** was performed using **F1-Score** to evaluate model stability across different subsets of the data.

---

# 📈 Results

## Test Set Performance

The following results were obtained on the held-out **20% test set**:

| Model                   |  Accuracy | Precision |    Recall |  F1-Score |   ROC-AUC |
| ----------------------- | --------: | --------: | --------: | --------: | --------: |
| **Logistic Regression** | **0.982** | **0.986** | **0.986** | **0.986** | **0.995** |
| **SVM (RBF Kernel)**    | **0.982** | **0.986** | **0.986** | **0.986** | **0.995** |
| XGBoost                 |     0.956 |     0.947 |     0.986 |     0.966 |     0.993 |
| Random Forest           |     ~0.96 |     ~0.95 |     ~0.98 |     ~0.96 |     ~0.99 |

> **Note:** Random Forest values are approximate and should be replaced with the exact values generated by the final notebook run before submission.

---

## 🔁 5-Fold Cross-Validation Results

F1-Score was used as the cross-validation metric.

| Model                   | Mean F1-Score | Standard Deviation |
| ----------------------- | ------------: | -----------------: |
| **SVM (RBF Kernel)**    |     **0.982** |              0.013 |
| **Logistic Regression** |     **0.979** |              0.013 |
| XGBoost                 |         0.972 |              0.008 |
| Random Forest           |         0.962 |              0.010 |

### Cross-Validation Interpretation

The relatively small standard deviations indicate that the models achieved consistent performance across the five folds.

The **SVM with RBF kernel** achieved the highest mean F1-score, while Logistic Regression was a very close second.

---

## 🏆 Best Performing Models

Based on the current evaluation:

* **Logistic Regression** achieved the highest F1-score on the held-out test set.
* **SVM (RBF Kernel)** achieved the highest mean F1-score during 5-fold cross-validation.
* Both models demonstrated strong classification performance.
* The ensemble tree-based models also achieved high performance but were slightly behind Logistic Regression and SVM on this dataset.

Overall, the results demonstrate that relatively simple machine learning models can perform extremely well on this structured medical dataset.

---

## 📊 Model Comparison

The project includes visual comparisons of model performance through:

* Confusion matrices
* ROC curves
* ROC-AUC comparison
* Classification metric comparison
* Feature importance plots

These visualizations provide additional insight into the strengths and weaknesses of each model.

---

## 🧠 Feature Importance & Interpretability

Feature importance analysis was performed for the tree-based models:

* Random Forest
* XGBoost

The analysis identifies the features that contribute most strongly to the model's predictions.

Features related to measurements such as:

* Worst concave points
* Worst perimeter
* Worst radius
* Worst area

are among the important predictive features in the dataset.

---

## 🔮 Single-Sample Prediction

The notebook also includes an inference example demonstrating how the trained models can be used to make a prediction for an individual sample.

For a new patient/sample, the pipeline performs:

```text
Input Features
      ↓
Feature Scaling
      ↓
Trained Model
      ↓
Prediction
      ↓
Prediction Probability
```

The output includes the predicted diagnosis and, where supported, the model's estimated class probabilities.

> **Important:** This project is an educational machine learning application and is **not intended for clinical diagnosis or medical decision-making**.

---

## 🗂️ Project Structure

```text
Disease-Prediction/
│
├── breast_cancer_disease_prediction.ipynb
│
└── README.md
```
---

## 🚀 Getting Started

### Requirements

The project requires:

```text
Python >= 3.9
NumPy
Pandas
Matplotlib
Seaborn
Scikit-learn
XGBoost
Jupyter
```

### Installation

Install the required libraries using:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost jupyter
```

---

## ▶️ Running the Project

Clone the repository:

```bash
git clone YOUR_REPOSITORY_URL
```

Navigate to the project directory:

```bash
cd Disease-Prediction
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Then open:

```text
breast_cancer_disease_prediction.ipynb
```

Run the notebook cells from top to bottom.

The dataset is loaded directly through `scikit-learn`, so no manual dataset download is required.

---

---

## ⚠️ Limitations

Although the models achieve high performance, several limitations should be considered:

* The dataset is relatively small compared with modern medical datasets.
* The dataset contains only structured numerical features.
* Results on this dataset do not necessarily generalize to other hospitals, populations, or clinical settings.
* High test performance does not mean the model is suitable for real-world medical diagnosis.
* Further external validation would be required before any clinical application.

---

---

## 📚 References

* UCI Machine Learning Repository — Breast Cancer Wisconsin (Diagnostic) Dataset
* Scikit-learn documentation
* XGBoost documentation

---

## 👩‍💻 Author

**Shorouk Hamdan**

Machine Learning 
Electrical Engineering Graduate

---

## ⭐ Project Summary

This project demonstrates an end-to-end machine learning workflow for binary classification using medical data, including data exploration, preprocessing, model training, evaluation, cross-validation, interpretability, and inference.

The results show that classical machine learning algorithms can achieve strong predictive performance on the Breast Cancer Wisconsin (Diagnostic) dataset while providing a useful foundation for further experimentation in medical machine learning.
