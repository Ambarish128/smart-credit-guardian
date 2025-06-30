#  Smart Credit Guardian

A machine learning project to predict personal loan applicants — not just using raw numbers, but also based on how people behave financially.

---

## Overview

Most banks assess personal loan eligibility using features like income, credit card usage, and existing loans. But numbers don't tell the whole story.

In this project, I experimented with clustering users into behavior-based groups before passing the data to machine learning models. This added context around user habits — whether they’re big spenders, cautious borrowers, or aggressive investors.

I trained and compared two models:
- **Random Forest**
- **XGBoost**

Both were optimized using GridSearchCV to get the best performance.

---

## Dataset

- Source: [Bank Personal Loan Modelling Dataset (Kaggle)](https://www.kaggle.com/datasets/itsmesunil/bank-loan-modelling)
- Each row represents one customer. Columns include:
  - `Income`, `Experience`, `CCAvg`, `Mortgage`
  - `Education`, `Family`, `CreditCard`, `Online`, etc.
- Target variable: `Personal Loan` (1 = accepted a loan offer, 0 = didn't)

---

##  What’s Different About This Project

- **Behavior-Based Clustering:**  
  I used KMeans to group users by income, spending, mortgage, and family size. These clusters became a new feature in the model (`Behaviour_Group`) to capture financial personality types.

- **Model Optimization:**  
  Both Random Forest and XGBoost models were tuned using `GridSearchCV` for better generalization.

- **Interpretability:**  
  Feature importances and ROC curves were plotted to understand what influenced predictions the most.

---

##  Model Results (After Tuning)

| Model               | Precision | F1 Score | 
|---------------------|----------|----------|
| Random Forest        | 0.990   | 0.990    | 
| XGBoost              | 0.991   | 0.991   

The `Behaviour_Group` feature (from clustering) consistently ranked among the top predictors.

---

## Key Visuals

### Feature Importance
![Random Forest](https://github.com/Ambarish128/smart-credit-guardian/blob/main/Feature_Importance_rf.png)  
![XGBoost](https://github.com/Ambarish128/smart-credit-guardian/blob/main/Feature_Importance_xgb.png)

### ROC Curve Comparison
![ROC Curve](https://github.com/Ambarish128/smart-credit-guardian/blob/main/ROC_Curve.png)

---

## Tech Stack

- Python (Pandas, NumPy)
- Scikit-Learn
- XGBoost
- Matplotlib, Seaborn
- Google Colab

---

## How to Run

1. Clone this repository
2. Upload the CSV dataset to your runtime
3. Open and run `loan_prediction_pipeline.ipynb` in Colab or Jupyter

---

## About Me

**Ambarish Shashank Gadgil**  
ML Enthusiast
---

## What Could Be Improved

- Build a Streamlit or Flask frontend to serve the model
- Use SHAP values for deeper interpretability
- Train on live or API-fed data
