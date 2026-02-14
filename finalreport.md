# ML Assignment 2 Submission: M.Tech (AIML/DSE)

**Student Name:** APAANPURI RANJITH KUMAR 

**BITS ID:** 2025AB05272

**Submission Date:** 02-08-2026

---

## 1. Mandatory Submission Links


**GitHub Repository Link:** [Paste Clickable Link Here] 


 
**Live Streamlit App Link:** [Paste Clickable Link Here] 



---

## 2. BITS Virtual Lab Execution Screenshot

The screenshot below serves as proof that the assignment was performed on the BITS Virtual Lab.

> **[INSERT SCREENSHOT HERE]**

---

## 3. GitHub README Documentation

The following content is also included in the repository README.md.


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

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| --- | --- | --- | --- | --- | --- | --- |
| **Logistic Regression** | 0.89 | 0.87 | 0.59 | 0.23 | 0.33 | 0.32 |
| **KNN** | 0.89 | 0.81 | 0.57 | 0.32 | 0.41 | 0.37 |
| **Decision Tree** | 0.88 | 0.71 | 0.48 | 0.49 | 0.49 | 0.42 |
| **Naive Bayes** | 0.84 | 0.81 | 0.36 | 0.47 | 0.41 | 0.32 |
| **Random Forest (Ensemble)** | 0.91 | 0.92 | 0.66 | 0.41 | 0.51 | 0.47 |
| **XGBoost (Ensemble)** | 0.91 | 0.93 | 0.63 | 0.48 | 0.55 | 0.50 |

---


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

## 4. Final Submission Checklist

* [x] GitHub repository contains `app.py`, `requirements.txt`, and saved models.


* [] Live Streamlit app is deployed and interactive.


* [x] App includes CSV upload for test data.


* [x] App displays Evaluation Metrics, Confusion Matrix, and Classification Report.


* [x] GitHub commit history reflects original development.


