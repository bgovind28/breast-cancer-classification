# Breast Mass Classification — Machine Learning Assignment 2

## a. Problem Statement

Compare five classification models for predicting whether a breast
mass is benign or malignant from numerical measurements of cell nuclei.

The project includes model training, evaluation, saved pipelines,
and a Streamlit application that evaluates uploaded test data
without retraining.

This is an educational project, not a clinical diagnostic tool.

## b. Dataset Description

- Dataset: Breast Cancer Wisconsin (Diagnostic)
- Original source: UCI Machine Learning Repository
- Source URL: https://archive.ics.uci.edu/dataset/17/breast-cancer
- Loading method: scikit-learn's bundled copy, using load_breast_cancer
- Records: 569
- Input features: 30 numerical features
- Classification task: Binary
- Target column: diagnosis
- Target encoding used in this project: 0 = benign, 1 = malignant
- Class counts: 357 benign and 212 malignant
- Missing values: 0
- Duplicate rows: 0

The original scikit-learn target encoding was reversed so that
malignant is the positive class.

### Data Preparation

An 80/20 stratified split with random_state=42 produced:
- Training set: 455 records
- Test set: 114 records

All five models use the same split.

Logistic Regression, KNN and SVM use StandardScaler inside their
pipelines. Scaling is fitted only on the training data.
Decision Tree and Gaussian Naive Bayes use unscaled features.

The exported test_data.csv contains the 30 input features and
the diagnosis column.

## c. GitHub Repository Link

https://github.com/bgovind28/breast-cancer-classification

## d. Models Used

| Model | Main settings |
|---|---|
| Logistic Regression | StandardScaler; max_iter=2000; random_state=42 |
| Decision Tree | max_depth=5; min_samples_leaf=2; random_state=42 |
| KNN | StandardScaler; n_neighbors=5 |
| Gaussian Naive Bayes | Default settings |
| Support Vector Machine | StandardScaler; RBF kernel; C=1.0; gamma=scale; probability=True; random_state=42 |

### Evaluation Metrics

Precision, Recall and F1 use malignant (1) as the positive class.
AUC uses predicted probabilities for the malignant class.
Results below are from the held-out test set of 114 records.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree | 0.9123 | 0.9444 | 0.9444 | 0.8095 | 0.8718 | 0.8113 |
| KNN | 0.9561 | 0.9823 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Gaussian Naive Bayes | 0.9386 | 0.9934 | 1.0000 | 0.8333 | 0.9091 | 0.8715 |
| Support Vector Machine | 0.9737 | 0.9947 | 1.0000 | 0.9286 | 0.9630 | 0.9442 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Highest AUC at 0.9960. It produced one false positive and three false negatives, with accuracy of 0.9649. |
| Decision Tree | Lowest accuracy, F1 and MCC among the compared models. It produced two false positives and eight false negatives. |
| KNN | Accuracy was 0.9561. It produced one false positive and four false negatives, placing it below Logistic Regression and SVM on F1. |
| Gaussian Naive Bayes | Precision was 1.0000, but recall was 0.8333. It produced no false positives but missed seven malignant cases. |
| Support Vector Machine | Highest accuracy, F1 and MCC. It produced no false positives and three false negatives. |
| Overall winner for this experiment | SVM offers the strongest overall balance of classification metrics on this split. Logistic Regression has a higher AUC and equal recall. |

These findings describe one held-out test split. They do not establish
performance on other populations or clinical suitability.

## e. Streamlit Application

Live application link: https://govind-breast-cancer-classification.streamlit.app/

Features:
- Download a sample test CSV.
- Upload a test CSV.
- Select any of the five saved classifiers.
- Display Accuracy, AUC, Precision, Recall, F1 and MCC.
- Display a confusion matrix and classification report.
- Preview uploaded data.

The app validates required columns, numeric values and target labels.
AUC is displayed as N/A if the uploaded data contains only one true class.

The application loads saved pipelines. It does not train models
on uploaded data.

## f. Project Files

- ML_Assignment_2.ipynb: training, evaluation and export source.
- app.py: Streamlit application.
- requirements.txt: package versions.
- metadata.json: feature names, target mapping and model filenames.
- test_data.csv: held-out test data with labels.
- model/: five saved model pipelines.
- results/metrics.csv: model comparison results.
- results/confusion_counts.csv: confusion-matrix counts.

## g. Run the Application

Use Python 3.11 with the dependencies listed in requirements.txt.

From the project folder:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Upload test_data.csv and select a model to view its evaluation results.
