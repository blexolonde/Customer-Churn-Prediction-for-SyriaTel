# Customer Churn Prediction for SyriaTel

## 📌 Project Overview

This project focuses on predicting customer churn for **SyriaTel**, a telecommunications company. Churn prediction is essential for reducing customer loss and improving retention strategies. The goal is to build a classification model that can identify at-risk customers before they leave the service.

## 📂 Dataset

The dataset is sourced from Kaggle and contains **customer information, usage patterns, and churn labels**. It includes **demographics, service plan details, call usage statistics, and customer service interactions**. The target variable is **"churn"**, which indicates whether a customer has left the company (1 = churned, 0 = stayed).  

The dataset includes the following key features:  

### **Customer Demographics**  
- `state`: The U.S. state where the customer resides.  
- `account length`: Duration (in days) the customer has been with the company.  

### **Service Plans**  
- `international plan`: Whether the customer has an international calling plan (Yes/No).  
- `voice mail plan`: Whether the customer has a voicemail plan (Yes/No).  

### **Usage Metrics**  
- number vmail messages: Number of voicemail messages.  
- total day minutes, total day calls, total day charge: Daytime call usage.  
- total eve minutes, total eve calls, total eve charge: Evening call usage.  
- total night minutes, total night calls, total night charge: Nighttime call usage.  
- total intl minutes, total intl calls, total intl charge: International call usage.  

### **Customer Service Interactions**  
- customer service calls: Number of calls made to customer service.  

### **Target Variable**  
- churn: Whether the customer has churned (1) or not (0).  

---

## 🎯 Objective

- Develop and evaluate machine learning models to predict customer churn.
- Identify key features that influence churn.
- Provide actionable insights to help reduce churn.

---

## 📊 Data Preprocessing

- **Feature Engineering**: Transformed and combined categorical and numerical features to improve model performance.
- **Handling Class Imbalance**: Applied **SMOTE (Synthetic Minority Over-sampling Technique)** to address class imbalance and prevent the model from favoring the majority class.
- **Encoding & Scaling**: Used **one-hot encoding** for categorical variables and **MinMax scaling** for numerical features.

---

## 🏆 Machine Learning Models

## **Machine Learning Models Used**
The following models were trained and evaluated:
1. **Random Forest Classifier**
2. **XGBoost Classifier** (Best-performing model)

### **Performance Metrics**
- **ROC-AUC Score**: Measures model discrimination between churners and non-churners.
  - **XGBoost AUC:** 0.8857
  - **Random Forest AUC:** 0.8722
- **Confusion Matrix** (for XGBoost):
  - True Positives: **69** (Churners correctly identified)
  - False Positives: **8** (Non-churners incorrectly classified as churners)
  - True Negatives: **563** (Correctly identified non-churners)
  - False Negatives: **27** (Churners incorrectly classified as non-churners)
- **Precision-Recall Curve**: XGBoost demonstrated better precision and recall, making it the preferred model.


---

## 📈 Model Evaluation: Confusion Matrices

Below are the confusion matrices for each model:

![Confusion Matrices](model_comparison_confusion_matrices.png)

**Key Observations**:


---

## 📉 ROC Curve Analysis

Below is the **ROC Curve** comparing the performance of all models:

![ROC Curve](Roc.png)


---

## Model Evaluation: Precision-Recall Curves



![Precision-Recall Curve](precesion.png)

---

## 🚀 Business Impact & Recommendations

1. **Retention Strategy Optimization**:
   - Focus retention efforts on customers with high churn probability (as identified by the model).
   - Improve customer experience for users making frequent customer service calls.
2. **Service Plan Enhancements**:
   - Customers with **no voice mail plans** have a higher churn risk—consider promotions to encourage adoption.
   - International plan subscribers show varying churn behavior—optimize offers based on usage trends.
3. **Predictive Monitoring**:
   - Deploy the XGBoost model in production to monitor churn risk in real-time.
   - Set up alerts for customers exceeding churn probability thresholds.

---

## 🏗️ Technologies Used

- **Python** (pandas, numpy, seaborn, matplotlib, scikit-learn, XGBoost, LightGBM)
- **Jupyter Notebook**
- **GridSearchCV & RandomizedSearchCV** (for hyperparameter tuning)

---

## 🚀 How to Run the Notebook

To run the notebook, follow these steps:

1. **Clone the Repository**:
   

git clone https://github.com/blexolonde//Customer-Churn-Prediction-for-SyriaTel
   cd Customer-Churn-Prediction-for-SyriaTel



2. **Install Dependencies**:
   

pip install -r requirements.txt



3. **Run the Jupyter Notebook**:
   

Customer-Churn-Prediction-for-SyriaTel.ipynb



Ensure you have all dependencies installed. If you encounter any issues, feel free to check the troubleshooting section below or open an issue on GitHub.

---

## 📌 Future Work

- Enhance the model with **feature selection** and **ensemble techniques**.
- Collect more behavioral data for better predictions and insights into customer churn.
- Deploy the model for real-time churn prediction in a production environment.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📌 Conclusion

This project developed a robust **customer churn prediction model** using multiple machine learning algorithms. **Random Forest** and **xgboost** emerged as the best models for churn prediction, showing strong performance across key metrics. With further refinement and deployment, this model will assist SyriaTel in reducing churn and improving customer retention strategies.

---

## 📬 Contact

For any inquiries or collaboration opportunities, feel free to reach out!

📧 Email: [olonde.blex@gmail.com]  
[![LinkedIn Profile](https://th.bing.com/th/id/OIP.EpUtPNFJAX-rfRrKJtYHvgHaD4?rs=1&pid=ImgDetMain)](https://www.linkedin.com/in/blexolonde)

---
