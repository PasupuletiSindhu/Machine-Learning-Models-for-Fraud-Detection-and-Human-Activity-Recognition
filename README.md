# Machine Learning Models for Fraud Detection and Human Activity Recognition

This repository contains two machine learning classification tasks focused on solving real-world problems using Random Forest models and advanced data preprocessing techniques.

---

## Problem Statements

### Task 1: Fraud Detection
The goal is to detect fraudulent credit card transactions. This is a highly imbalanced classification problem where the number of fraudulent transactions is significantly lower than legitimate ones. Early and accurate detection is critical for minimizing financial losses.

### Task 2: Human Activity Recognition
This task aims to classify various human physical activities using sensor data. The dataset includes measurements from motion sensors (such as accelerometers and gyroscopes) and is used to build a model that accurately distinguishes between different types of activities.

---

## Datasets

- **Task 1 Dataset**: Credit card transaction data with features indicative of fraudulent behavior.
- **Task 2 Dataset**: Sensor data recording human activities, including features from motion and position sensors.

---

## Methodology

The core methodology applied to both tasks includes:

- **Model**: Random Forest Classifier
- **Cross-Validation**: Stratified K-Fold to preserve class distribution
- **Imbalanced Data Handling**: SMOTETomek (a hybrid of SMOTE oversampling and Tomek links undersampling)
- **Feature Selection**: RFECV (Recursive Feature Elimination with Cross Validation)
- **Model Evaluation**: Classification report, confusion matrix, and key metrics
- **Model Persistence**: Exported using `joblib` for reuse

---

## Tools and Libraries

- Python 3
- Pandas
- NumPy
- Scikit-learn
- Imbalanced-learn
- Matplotlib
- Graphviz
- PyTorch (used for GPU detection only)
- Joblib

---

## Getting Started

### Installation

1. Clone the repository.
2. Install the required packages:

```bash
pip install -r requirements.txt
```

### Running the Notebooks

Run each task using Jupyter Notebook:

```bash
jupyter notebook t1.ipynb  # For Fraud Detection  
jupyter notebook t2.ipynb  # For Human Activity Recognition
```

---

## Evaluation Metrics

The performance of the models is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Macro-averaged scores for robustness in imbalanced classification

---

## Results

Both tasks leverage robust evaluation techniques and preprocessing to yield high-quality, generalizable models. The combination of SMOTETomek and RFECV improves the handling of rare classes and removes irrelevant features, thus boosting performance and interpretability.

---

## Notes

- GPU availability is detected using PyTorch; however, model training runs on the CPU.
- Model and evaluation code are modular and reproducible.
- Feature importance and decision tree structures can be visualized using Graphviz.

