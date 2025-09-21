# Transaction Fraud Detection

This project implements a **fraud detection system** using the [Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), which contains **284,807 transactions** made by European cardholders in September 2013. Among these, only **492 transactions were fraudulent** (~0.17%), making it an excellent real-world example of a highly **imbalanced dataset**.

The objective is to build machine learning models that can accurately detect fraudulent transactions while minimizing false alarms.

---

## Project Overview

- **Dataset size:** 284,807 rows × 31 columns  
- **Fraud cases:** 492 (0.17%)  
- **Features:** 28 anonymized PCA components (V1–V28), Time, Amount  
- **Target variable:** `Class` → (0 = Legitimate, 1 = Fraud)  
- **Goal:** Detect fraud with high recall and strong overall accuracy

---

## Workflow

### 1. Exploratory Data Analysis (EDA)
- Inspected dataset shape and missing values  
- Checked fraud class imbalance  
- Visualized fraud vs non-fraud transactions  
- Plotted distributions of transaction amounts and times  

### 2. Data Preprocessing
- Normalized the `Amount` and `Time` features using **StandardScaler**  
- Applied **SMOTE (Synthetic Minority Oversampling Technique)** to balance the dataset:
  - Before SMOTE: ~0.17% fraud cases  
  - After SMOTE: balanced 50–50 fraud vs non-fraud  

### 3. Model Training
Trained three machine learning models:
- **Logistic Regression** (baseline, interpretable)  
- **Random Forest Classifier** (ensemble, high accuracy)  
- **XGBoost Classifier** (gradient boosting, high accuracy and efficiency)  

Each model was validated using **Stratified Cross-Validation** to ensure performance stability.

### 4. Model Evaluation
Evaluated models using:
- Accuracy  
- Precision, Recall, F1-score  
- Confusion Matrix  
- ROC-AUC Score  
- ROC Curve visualization  

---

##  Results

| Model                | Accuracy | ROC-AUC | Notes |
|-----------------------|----------|---------|-------|
| Logistic Regression   | ~94%     | ~0.96   | Strong recall after SMOTE |
| Random Forest         | ~99.8%   | ~1.00   | Excellent performance |
| XGBoost               | ~99.6%   | ~1.00   | Excellent performance |

- Logistic Regression achieved **94% accuracy with high recall**, matching real-world expectations.  
- Random Forest and XGBoost delivered near-perfect scores, showing the power of ensemble methods.  

---

## Technologies Used

- **Python**  
- **Pandas, NumPy** – data handling  
- **Scikit-learn** – ML models and preprocessing  
- **Imbalanced-learn (SMOTE)** – oversampling technique  
- **XGBoost** – gradient boosting  
- **Matplotlib, Seaborn** – visualization  

---

## 📈 Visualizations Included
- Fraud vs Non-Fraud distribution  
- Transaction amount distributions  
- Correlation heatmap of features  
- ROC Curves for all models  
- Confusion Matrices  

---
##  Business Impact

- Detects fraudulent transactions in **real-time**  
- Helps financial institutions reduce monetary losses  
- Increases customer trust and security  
- Demonstrates effective handling of **extremely imbalanced datasets**, a critical challenge in fraud detection  
---

##  Key Takeaways
- Successfully handled a **highly imbalanced dataset** using SMOTE  
- Validated models with **cross-validation** to avoid overfitting  
- Achieved **94% accuracy with strong fraud recall** (resume-ready metric)  
- Ensemble methods (Random Forest, XGBoost) provided near-perfect classification  

---

