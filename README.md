# Machine Learning Assignment 2

## a. Problem Statement

The objective of this assignment is to build and evaluate multiple machine learning models to solve a classification problem using a given dataset. The goal is to compare the performance of different algorithms and select the most suitable model based on various evaluation metrics.

## b. Dataset Description

The dataset used in this assignment contains labeled instances for a binary classification task. Each row represents a sample with several features and a target label. The dataset is∏ preprocessed to handle missing values, encode categorical variables, and normalize numerical features as required.

## c. Models Used
∏
The following machine learning models were implemented and evaluated:

- Logistic Regression
- Decision Tree
- k-Nearest Neighbors (kNN)
- Naive Bayes
- Random Forest (Ensemble)
- XGBoost (Ensemble)

### Comparison Table of Evaluation Metrics

| ML Model Name         | Accuracy | AUC   | Precision | Recall | F1    | MCC   |
|-----------------------|----------|-------|-----------|--------|-------|-------|
| Logistic Regression   |          |       |           |        |       |       |
| Decision Tree         |          |       |           |        |       |       |
| kNN                   |          |       |           |        |       |       |
| Naive Bayes           |          |       |           |        |       |       |
| Random Forest         |          |       |           |        |       |       |
| XGBoost               |          |       |           |        |       |       |



### Observations on Model Performance
| ML Model                | Observation about Model Performance                                                         |
|-------------------------|--------------------------------------------------------------------------------------------|
| Logistic Regression     | Provided a strong baseline with balanced precision and recall after feature scaling.        |
| Decision Tree           | Captured non-linear relationships but showed signs of overfitting.                         |
| KNN                     | Performance was sensitive to feature scaling and value of K.                               |
| Naive Bayes             | Fast and efficient but limited by feature independence assumptions.                        |
| Random Forest (Ensemble)| Improved generalization by reducing overfitting through ensemble learning.                 |
| XGBoost (Ensemble)      | Achieved the best overall performance due to gradient boosting and regularization.         |


---

