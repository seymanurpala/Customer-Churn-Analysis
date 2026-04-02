# Customer Churn Prediction

A machine learning pipeline built to predict customer churn using the Cell2Cell telecom dataset. The project covers data preprocessing, feature selection, dimensionality reduction and multi-model classification.

## Dataset

- Training set: `cell2celltrain.csv`
- Holdout set: `cell2cellholdout.csv`
- Target variable: `Churn` (Yes / No → 1 / 0)

## Pipeline

- Duplicate and missing value analysis
- Missing values filled with mode (categorical) and mean (numerical)
- Label Encoding for categorical columns
- Feature selection with SelectKBest (mutual information, top 25 features)
- Dimensionality reduction with TruncatedSVD
- Feature scaling with StandardScaler
- Stratified 5-Fold Cross-Validation for class-balanced splits

## Models

- Naive Bayes
- Logistic Regression
- AdaBoost
- Bagging Classifier
- Neural Network (Keras Sequential with EarlyStopping)
- PyTorch Neural Network

## Evaluation

- Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Confusion Matrix
- Classification Report
- ROC Curve per model

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
jupyter notebook customer-churn-prediction.ipynb
```

## Requirements

- Python 3.9+
- scikit-learn, pandas, numpy, matplotlib, seaborn
- tensorflow, torch
