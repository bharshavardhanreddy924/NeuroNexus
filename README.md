# NeuroNexus
# 💳 Credit Card Fraud Detection using Machine Learning

![Confusion Matrix](./confusion_matrix.png) <!-- You can update this with your actual path -->

## 📌 Overview

This project focuses on detecting fraudulent credit card transactions using machine learning. The dataset is highly imbalanced, which presents challenges that we address using SMOTE oversampling. Two classification models—**Random Forest** and **Logistic Regression**—were trained and evaluated on their performance in identifying fraud accurately.

---

## 📂 Dataset

- **Source**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Contains transactions made by European cardholders in September 2013.
- Total Transactions: **284,807**
- Fraudulent Transactions: **492** (only 0.17%)

---

## ⚙️ Tech Stack

- Python 3.x
- pandas, numpy
- scikit-learn
- imbalanced-learn (SMOTE)
- matplotlib, seaborn

---

## 🧪 Workflow

### 1. **Data Preprocessing**
- Removed `Time` and original `Amount` columns.
- Normalized `Amount` using `StandardScaler`.

### 2. **Handling Class Imbalance**
- Applied **SMOTE (Synthetic Minority Oversampling Technique)** to balance classes.
- Resulted in equal samples for fraudulent and genuine transactions.

### 3. **Splitting the Data**
- Used `train_test_split` with stratification (80% train, 20% test).

### 4. **Model Training**
- Trained two models:
  - **Random Forest Classifier** (`n_estimators=100`)
  - **Logistic Regression** (`max_iter=1000`)

### 5. **Evaluation Metrics**
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

---

## 📊 Model Performance

| Model                | Accuracy | Precision (Fraud) | Recall (Fraud) | F1-Score (Fraud) |
|---------------------|----------|-------------------|----------------|------------------|
| **Random Forest**    | 0.9999   | 0.9997            | 1.0000         | 0.9999           |
| **Logistic Regression** | 0.9462   | 0.9727            | 0.9182         | 0.9447           |

> ✅ **Random Forest** clearly outperformed Logistic Regression in all metrics.

---

## 🧠 Conclusion

- Fraud detection is a classic example of an **imbalanced classification** problem.
- SMOTE oversampling significantly helped in training a balanced and robust model.
- Random Forest yielded near-perfect performance with **zero false negatives**.
- This pipeline can be deployed for real-time fraud detection systems in the financial industry.

---

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/bharshavardhanreddy924/NeuroNexus.git
   cd NeuroNexus
