#  ML Classification Pipeline

End-to-end machine learning pipeline with preprocessing, multi-model classification and evaluation.

##  Models

- Random Forest
- Logistic Regression
- Naive Bayes
- Decision Tree

##  Pipeline

| Step | Detail |
|---|---|
| Data Cleaning | Duplicates, outliers, inconsistencies |
| Missing Values | Mean/Median/Mode imputation |
| Encoding | Label Encoding, One-Hot Encoding |
| Scaling | StandardScaler |

##  Evaluation

| Metric | Detail |
|---|---|
| Accuracy | Overall correct predictions |
| F1-Score | Precision & recall balance |
| ROC-AUC | Classification threshold performance |
| Confusion Matrix | TP, TN, FP, FN breakdown |
| Cross-Validation | 5-fold CV |
| GridSearchCV | Hyperparameter tuning |

##  Usage

```bash
pip install -r requirements.txt
jupyter notebook notebooks/ml_pipeline.ipynb
```

> Python 3.9+, scikit-learn 1.x required.
