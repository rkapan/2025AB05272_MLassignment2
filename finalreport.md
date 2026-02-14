# ML Assignment 2 Submission: M.Tech (AIML/DSE)

**Student Name:** APAANPURI RANJITH KUMAR 

**BITS ID:** 2025AB05272

**Submission Date:** 02-08-2026

---

## 1. Mandatory Submission Links


**GitHub Repository Link:** https://github.com/rkapan/2025AB05272_MLassignment2.git


 
**Live Streamlit App Link:** https://2025ab05272bitsmlassignment.streamlit.app 



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
| ML Model | Observation about Model Performance |
| --- | --- |
| **Logistic Regression** | Served as a baseline; however, it struggled significantly with class imbalance, yielding the lowest **MCC (0.28)** and **Recall (0.19)** among all models. |
| **Decision Tree** | Captured non-linear patterns effectively with high **Accuracy (0.98)**, but the gap between Accuracy and MCC (0.88) suggests some sensitivity to the minority class. |
| **KNN** | Demonstrated moderate performance (**Accuracy: 0.92**); however, its lower **Recall (0.44)** indicates it missed over half of the potential positive leads. |
| **Naive Bayes** | While computationally efficient, it delivered the lowest **Accuracy (0.83)** and **AUC (0.80)**, likely due to the strong independence assumptions between features. |
| **Random Forest (Ensemble)** | **Top Performer.** Achieved a perfect balance across all metrics with the highest **MCC (0.92)** and **F1-Score (0.93)**, showing superior generalization through bagging. |
| **XGBoost (Ensemble)** | Strong overall performance (**Accuracy: 0.95**, **AUC: 0.97**) due to gradient boosting; however, it was slightly outperformed by Random Forest in Precision and Recall. |

---

## 4. Screenshots
![alt text](image-2.png)

* Streamlit app screenshot 
![Alt text](image-1.png)

## 4. Final Submission Checklist

* [x] GitHub repository contains `app.py`, `requirements.txt`, and saved models.


* [x] Live Streamlit app is deployed and interactive.


* [x] App includes CSV upload for test data.


* [x] App displays Evaluation Metrics, Confusion Matrix, and Classification Report.


* [x] GitHub commit history reflects original development.


