# **Customer Churn Prediction for SyriaTel**

I created this project to predict **customer churn** for **SyriaTel**, a leading telecommunications company, using machine learning techniques. By identifying patterns in customer behavior, I aim to develop strategies that reduce churn and enhance customer retention.  

---

## **Table of Contents**

- [Project Overview](#project-overview)  
- [Technologies Used](#technologies-used)  
- [Dataset Overview](#dataset-overview)  
- [Steps Taken](#steps-taken)  
  - [1. Data Preprocessing](#1-data-preprocessing)  
  - [2. Feature Engineering](#2-feature-engineering)  
  - [3. Model Training](#3-model-training)  
  - [4. Model Evaluation](#4-model-evaluation)  
- [Evaluation Metrics](#evaluation-metrics)  
- [Model Comparison & Hyperparameter Tuning](#model-comparison--hyperparameter-tuning)  
- [Confusion Matrix (XGBoost)](#confusion-matrix-xgboost)  
- [Example Prediction](#example-prediction)  
- [How to Run](#how-to-run)  
- [Future Work](#future-work)  
- [Conclusion](#conclusion)  

---

## **Project Overview**

SyriaTel faces a major business challenge—**customer churn**. By predicting which customers are likely to leave, the company can proactively engage them with personalized retention strategies. In this project, I use machine learning models to classify customer churn based on historical data.  

### **Objective**

- Build a **churn prediction model** that identifies customers likely to leave.  
- Train and compare **classification models** to evaluate their performance and provide actionable insights.  
- Optimize models using **hyperparameter tuning** and **cross-validation**.  

---

## **Technologies Used**

I used the following tools and libraries to build this project:  

- **Python**: The main programming language for data manipulation, modeling, and analysis.  
- **Libraries**:  
  - `pandas`, `numpy`: Data manipulation and analysis.  
  - `scikit-learn`: Machine learning and model evaluation.  
  - `matplotlib`, `seaborn`: Data visualization.  
  - `imbalanced-learn`: Handling imbalanced data (SMOTE).  
  - `XGBoost`: A powerful gradient boosting model.  
  - `GridSearchCV`: Hyperparameter tuning via cross-validation.  

---

## **Dataset Overview**

I worked with a dataset containing **3,333 customer records** and **21 features**. Key features include:  

- **Customer account details**: Account length, area code, service plans.  
- **Usage statistics**: Total minutes and charges for day, evening, night, and international calls.  
- **Customer behavior**: Number of customer service calls, voice mail usage, etc.  
- **Target variable**: `churn` (binary: 1 = churned, 0 = retained).  

### **Dataset Features**

| Feature                     | Description                                  |
|-----------------------------|----------------------------------------------|
| account length              | Length of time the customer has been with SyriaTel |
| total day minutes           | Total number of minutes the customer spent on calls during the day |
| total day calls             | Number of calls made during the day |
| churn                       | Target variable (1 = churned, 0 = retained) |

---

## **Steps Taken**

### 1. **Data Preprocessing**

I started by loading and cleaning the dataset:  

- **Missing Values**: No missing values found.  
- **Encoding Categorical Variables**: Features like `state`, `international plan`, and `voice mail plan` were converted into numerical values.  
- **Feature Scaling**: Applied scaling to numerical features to ensure equal importance for all features during model training.  

---

### 2. **Feature Engineering**

I created new features to enhance the model’s predictive power:  

- **Call Rate Features**: I engineered features like `intl_call_rate`, `daytime_call_rate`, and `evening_call_rate` to capture **customer behavior patterns** more effectively.  
  - **Why?** High international call rates may indicate **loyal business customers**, whereas unusually low call rates could be a **churn risk indicator**.  
  - These features help **differentiate customer types** and improve model accuracy.  

---

### 3. **Model Training**

I trained and compared four machine learning models:  

1. **Logistic Regression**  
2. **Decision Tree**  
3. **Random Forest**  
4. **XGBoost**  

I used **GridSearchCV** to fine-tune the hyperparameters for **Decision Tree**, **Random Forest**, and **XGBoost** to improve model performance.  

---

### 4. **Model Evaluation**

I evaluated each model based on **accuracy**, **precision**, **recall**, **F1-score**, and **ROC-AUC**. I also used **confusion matrices** and **ROC curves** to visualize performance.  

---

## **Evaluation Metrics**

I assessed the models using the following metrics:  

- **Accuracy**: The ratio of correct predictions to total predictions.  
- **Precision**: The proportion of positive predictions that are actually correct.  
- **Recall**: The proportion of actual positives correctly predicted.  
- **F1-Score**: The harmonic mean of precision and recall, balancing the two.  
- **AUC-ROC**: The area under the ROC curve, showing how well the model distinguishes between churned and retained customers.  

---

## **Model Comparison & Hyperparameter Tuning**

After tuning hyperparameters using **GridSearchCV**, the models performed as follows:  

| Model            | Best Parameters                           | Accuracy | AUC-ROC | Precision | Recall | F1-Score |
|------------------|-------------------------------------------|----------|---------|-----------|--------|----------|
| Decision Tree    | `max_depth=10, min_samples_split=2`       | 0.85     | 0.80    | 0.82      | 0.78   | 0.80     |
| Random Forest    | `n_estimators=100, max_depth=20`         | 0.88     | 0.84    | 0.87      | 0.85   | 0.86     |
| XGBoost          | `n_estimators=100, learning_rate=0.1`    | 0.90     | 0.87    | 0.89      | 0.88   | 0.89     |

---

## **Confusion Matrix (XGBoost)**

The confusion matrix below shows how well the **XGBoost model** classifies customers:  

![Confusion Matrix](Untitled.png)  

---

## **Example Prediction**

Here’s how to use the trained XGBoost model to predict churn for a new customer:  

```python
import pandas as pd
import joblib  # For loading the trained model

# Load the trained model
model = joblib.load("xgboost_churn_model.pkl")

# Example new customer data
new_customer = pd.DataFrame({
    'account_length': [120],
    'international_plan': [0],
    'voice_mail_plan': [1],
    'total_day_minutes': [200],
    'total_day_calls': [100],
    'total_eve_minutes': [150],
    'total_eve_calls': [80],
    'total_night_minutes': [180],
    'total_night_calls': [90],
    'total_intl_minutes': [10],
    'total_intl_calls': [5],
    'customer_service_calls': [2],
})

# Predict churn probability
churn_probability = model.predict_proba(new_customer)[:, 1]
prediction = model.predict(new_customer)

print(f"Churn Probability: {churn_probability[0]:.2f}")
print(f"Predicted Class: {'Churn' if prediction[0] == 1 else 'Retained'}")
```

---

## **How to Run**

1. **Clone the Repository:**  
```bash
git clone https://github.com/blexolonde/syriatel-churn-prediction.git  
cd syriatel-churn-prediction  
```

2. **Install Dependencies:**  
```bash
pip install -r requirements.txt  
```

---

## **Future Work**

- **Ensemble Learning:** Combine the predictions from multiple models.  
- **Deep Learning:** Experiment with neural networks or advanced models.  
- **Real-time Data Updates:** Adapt models to changing behavior patterns.  

---

## **Conclusion**

This project demonstrates how machine learning can be used to predict customer churn in the telecommunications industry. By implementing multiple models, tuning hyperparameters, and evaluating performance thoroughly, I provide actionable insights that can help SyriaTel reduce churn and improve customer retention.  

---
