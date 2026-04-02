#  ML Classification Pipeline

End-to-end machine learning pipeline with preprocessing, multi-model classification and evaluation.

##  Models

- Random Forest
- Logistic Regression
- Naive Bayes
- Decision Tree

##  Pipeline

**Data Cleaning** — duplicates, outliers and inconsistencies are removed.
**Missing Values** — numerical features are filled with mean/median, categorical features with mode imputation.
**Encoding** — ordinal features use Label Encoding, nominal features use One-Hot Encoding.
**Scaling** — all features are standardized using StandardScaler.

## Evaluation

Models are evaluated with Accuracy, F1-Score and ROC-AUC scores. Confusion matrix is plotted for each model. 5-fold Cross-Validation is applied for robust performance estimation. Hyperparameters are tuned using GridSearchCV.

##  Usage

```bash
pip install -r requirements.txt
jupyter notebook notebooks/ml_pipeline.ipynb
```

> Python 3.9+ and scikit-learn 1.x required.

