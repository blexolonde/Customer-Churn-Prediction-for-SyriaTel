# **SyriaTel Customer Churn Prediction**

## **Project Goal**  
The goal of this project is to predict customer churn for **SyriaTel**, a telecommunications company, in order to take proactive measures to reduce customer attrition and improve retention strategies.

## **Table of Contents**  
- [Project Overview](#project-overview)  
- [Dataset](#dataset)  
- [Data Preparation](#data-preparation)  
- [Modeling Approach](#modeling-approach)  
- [Evaluation & Optimization](#evaluation--optimization)  
- [Key Insights](#key-insights)  
- [Business Recommendations](#business-recommendations)  
- [Next Steps](#next-steps)  
- [How to Run](#how-to-run)  

---

## **Project Overview**  
Customer churn is a significant problem for telecom companies, causing **substantial revenue loss**. This project creates a **machine learning model** to predict churn based on customer behavior, call usage, and service interactions.  

### **Why It Matters**  
- **Losing customers** is costly; retention is more cost-effective than acquiring new ones.  
- **Early detection** of churn helps in offering targeted promotions, discounts, or improved services to retain customers.  

### **Key Outcomes**  
- Best-performing model: **XGBoost (AUC-ROC: 0.85)**  
- Most predictive features: **Total Day Minutes, Customer Service Calls, Total Intl Minutes**  
- Strategic recommendations for reducing churn  

---

## **Dataset**  
📁 **Source:** SyriaTel customer data (`bigml.csv`)  from kaggle

### **Features Overview:**  
- **Usage Data:** `total day minutes`, `total eve minutes`, `total night minutes`, `total intl minutes`  
- **Customer Service Interaction:** `customer service calls`  
- **Subscription Plans:** `international plan`, `voice mail plan`  
- **Demographics:** `state`, `area code`  
- **Target Variable:** `Churn` (1 = Churned, 0 = Not Churned)  

---

## **Data Preparation**  
1. **Data Cleaning & Feature Engineering**  
   - Removed irrelevant columns (`phone number`, `account length`, `area code`)  
   - Converted categorical variables (`yes/no` plans → binary encoding)  
   - Checked for missing values & outliers  

2. **Balancing Classes with SMOTE**  
   - The dataset had an imbalanced class distribution, so I applied **Synthetic Minority Over-sampling Technique (SMOTE)** to balance the classes.  

3. **Feature Scaling**  
   - Standardized numerical features using **StandardScaler** to improve model performance.  

---

## **Modeling Approach**  
I trained and compared four machine learning models:  

| Model                | AUC-ROC Score |
|----------------------|--------------|
| Logistic Regression  | 0.78         |
| Decision Tree        | 0.81         |
| **Random Forest**    | **0.83**     |
| **XGBoost**          | **0.85** ✅ |

### **Best Model:** **XGBoost** (Extreme Gradient Boosting)  
- Handles complex relationships effectively  
- Robust to feature importance weighting  
- Performs well with structured data  

---

## **Evaluation & Optimization**  
I fine-tuned the classification threshold to improve **recall** and reduce false negatives (incorrectly predicting customers as not churning).

| Threshold | Precision | Recall | F1-Score |
|-----------|----------|--------|----------|
| **0.30**  | 0.74     | 0.75   | 0.74     |
| **0.40**  | 0.77     | 0.73   | 0.75 ✅  |
| **0.50**  | 0.82     | 0.70   | 0.76     |

### **Best Threshold:** 0.40  
- Balanced **precision & recall**  

---

## **Key Insights**  
### **Top 3 Most Important Features**  
- **Total Day Minutes:** Customers with higher day-time usage are more likely to churn.  
- **Customer Service Calls:** Frequent customer support calls indicate dissatisfaction, which is a strong churn predictor.  
- **Total Intl Minutes:** Customers with high international minutes tend to churn more frequently.  

### **Churners vs Non-Churners Analysis**  
- **High service usage** doesn’t always lead to retention.  
- **Frequent complaints correlate with churn** — addressing these could improve loyalty.  
- **International plan users** have a higher churn rate — special retention efforts are needed.  

---

## **Business Recommendations**  
1. **Improve Customer Support**  
   - Customers making **3+ support calls** are at high risk of churning.  
   - Implement a **proactive support strategy** where dissatisfied customers are contacted before they leave.  

2. **Loyalty Discounts for High Usage Customers**  
   - Offer **discounted plans** for users with high day minutes to encourage retention.  
   - Reward **long-term customers** with loyalty bonuses.  

3. **Special Retention Offers for Intl Callers**  
   - Customers with **high international minutes** are more likely to churn.  
   - Provide **exclusive discounts on international plans** to retain them.  

---

## **Next Steps**  
- Deploy the **XGBoost model** with API integration for **real-time churn prediction**.  
- Implement **automated alerts** for at-risk customers.  
- Test **A/B retention strategies** to measure impact.  

---

## **How to Run**  
### **Prerequisites**  
Make sure you have the following installed:  
- Python 3.8+  
- Jupyter Notebook  
- Required libraries:  
  ```bash
  pip install pandas numpy scikit-learn xgboost imbalanced-learn matplotlib seaborn
  ```

### **Steps to Run**  
1. Clone this repository:  
   ```bash
   git clone https://github.com/blexolonde/syriatel-churn-prediction.git
   cd syriatel-churn-prediction
   ```
2. Open Jupyter Notebook:  
   ```bash
   jupyter notebook
   ```
3. Run **SyriaTel_Churn_Prediction.ipynb** in Jupyter  

---

## **Contributors**  
**Blex Olonde** - Data Science & Machine Learning  
📧 Contact: olonde.blex@gmail.com 

---

## **Final Thoughts**  
This project successfully identifies the key drivers of churn and builds a robust predictive model for SyriaTel. **By implementing proactive retention strategies, SyriaTel can reduce churn and maximize customer lifetime value.**  

---

If you found this project useful, feel free to ⭐ this repo!
