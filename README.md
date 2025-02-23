# Customer Churn Prediction for SyriaTel

## 📌 Project Overview
This project aims to predict customer churn for **SyriaTel**, a telecommunications company. Churn prediction is crucial for reducing customer loss and improving retention strategies. The goal is to build a classification model that identifies at-risk customers before they leave the service.

## 📂 Dataset
The dataset, sourced from Kaggle, contains customer records with various attributes such as:

Customer service calls

Call minutes (day, evening, night, international)

International plan usage

Account length and total charge information

Churn status (target variable)

## 🎯 Objective
- Develop and evaluate machine learning models to predict customer churn.
- Identify key features that influence churn.
- Provide actionable insights to mitigate customer churn.

---

## 📊 Data Preprocessing
- **Feature Engineering**: Combined and transformed categorical and numerical features.
- **Handling Imbalance**: Used **SMOTE (Synthetic Minority Over-sampling Technique)** to address class imbalance.
- **Encoding & Scaling**: Applied **one-hot encoding** for categorical variables and **MinMax scaling** for numerical features.

---

## 🏆 Machine Learning Models
Five classification models were trained and optimized using **hyperparameter tuning**:

### Model Performance Comparison:

| Model                        | Accuracy | Precision | Recall | F1-score | ROC-AUC |
| ---------------------------- | -------- | --------- | ------ | -------- | ------- |
| Logistic Regression          | 70.3%    | 30.4%     | 81.3%  | 44.2%    | 81.8%   |
| Decision Tree Classifier     | 82.4%    | 42.3%     | 59.6%  | 49.5%    | 75.2%   |
| Random Forest Classifier     | 89.7%    | 63.0%     | 68.9%  | 65.8%    | 85.8%   |
| LightGBM                     | 89.4%    | 62.9%     | 65.8%  | 64.3%    | 84.6%   |
| XGBoost                      | 86.9%    | 53.9%     | 64.8%  | 58.8%    | 83.4%   |

The **Random Forest model** performed the best with **89.7% accuracy** and the highest **ROC-AUC (85.8%)**, making it the ideal choice for predicting customer churn. LightGBM also performed well, but with slightly lower recall and ROC-AUC. 
### Best Hyperparameters for Each Model
Hyperparameters were chosen using ** Random Search** techniques, optimizing for **F1-score and ROC-AUC** to balance precision and recall.

- **Logistic Regression**: `solver='liblinear', penalty='l1', C=0.1`
  - Chosen to encourage feature selection via L1 regularization and handle multicollinearity.
- **Decision Tree**: `min_samples_split=5, min_samples_leaf=2, max_depth=None`
  - Optimized for better generalization and reduced overfitting.
- **Random Forest**: `n_estimators=500, min_samples_split=2, min_samples_leaf=1, max_depth=None`
  - A higher number of estimators improves stability, while default depth allows deeper trees.
- **LightGBM**: `subsample=0.7, num_leaves=40, n_estimators=500, learning_rate=0.1, boosting_type='gbdt'`
  - Balanced between fast training and preventing overfitting with controlled leaf growth.
- **XGBoost**: `subsample=0.8, n_estimators=500, max_depth=6, learning_rate=0.2`
  - Tuned for optimal feature interactions while maintaining computational efficiency.

---

## 📈 Model Evaluation: Confusion Matrices

Below are the confusion matrices for each model:

![Confusion Matrices](model_comparison_confusion_matrices.png)

Churn Rate is Relatively Low: The number of actual churn cases seems to be lower than non-churn cases, indicating that most customers tend to stay.
Misclassification Trends:
False Positives (FP): Some customers are predicted to churn but actually stay, especially in Logistic Regression.
False Negatives (FN): Some customers who actually churn are missed, which is more dangerous for the business.
Importance of Reducing False Negatives: False negatives mean losing customers without taking preventive actions—Random Forest & LightGBM perform best in minimizing this.

---

## 📉 ROC Curve Analysis

Below is the **ROC Curve** comparing the performance of all models:

![ROC Curve](Roc.png)

### 🔍 Interpretation:
- **Random Forest (AUC = 0.86) and LightGBM (AUC = 0.85)** perform best, showing strong predictive power.
- **XGBoost (AUC = 0.83)** follows closely, indicating good performance.
- **Logistic Regression (AUC = 0.82)** has decent performance but struggles with precision.
- **Decision Tree (AUC = 0.76)** performs the worst, meaning it is less reliable for churn prediction.

---

## 📊 Model Evaluation: Precision-Recall Curves - Model Comparison

### Key Insights:
- **LightGBM** has the best **precision-recall balance**, with the highest **Average Precision (AP = 0.65)**, indicating it performs well in identifying churners with minimal false positives.
- **Random Forest** follows closely with an **AP = 0.62**, offering a strong balance between precision and recall.
- **XGBoost** shows competitive results with an **AP = 0.60**, though it slightly lags behind LightGBM and Random Forest in overall performance.
- **Decision Tree** and **Logistic Regression** show lower **precision-recall trade-offs**, making them less optimal choices for **churn prediction**, as they are more prone to misclassifying churners and non-churners.

### Precision-Recall Curve Comparison:

Below is the **Precision-Recall Curve** comparing the performance of all models.

![Precision-Recall Curve](precesion.png)

### 🔍 Interpretation:
- **LightGBM** achieves the highest **precision-recall trade-off**, meaning it correctly identifies churners with high precision while maintaining a reasonable recall.
- **Random Forest** also shows a strong performance with a solid balance, just behind LightGBM.
- **XGBoost** has decent performance but is not as well-balanced in precision and recall.
- **Decision Tree** and **Logistic Regression** have wider **precision-recall trade-offs**, meaning they are more prone to **false positives and false negatives**.

---

## 🚀 Business Impact & Recommendations

Based on the findings, the following actions are recommended:
- **Customer Retention Programs**: Target customers identified as high churn risk with personalized offers or discounts.
- **Improve Customer Support**: Customers with high service usage variations may require better support.
- **Feature Optimization**: Focus on the most important features affecting churn to enhance prediction accuracy.
- **Threshold Tuning**: Adjust classification thresholds to balance false positives and false negatives based on business needs.

---

## 🏗️ Technologies Used

- **Python** (pandas, numpy, seaborn, matplotlib, scikit-learn, XGBoost, LightGBM)
- **Jupyter Notebook**
- **SHAP** (for model interpretability)
- **GridSearchCV & RandomizedSearchCV** (for hyperparameter tuning)

## 🚀 How to Run the Notebook

```
   ```

## 📌 Future Work

- Improve model performance with **feature selection** and **ensemble techniques**.
- Enhance data collection with more customer behavioral metrics.


---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📌 Conclusion
This project successfully developed a robust **customer churn prediction model** using multiple machine learning techniques. **Random Forest and LightGBM** emerged as the best models, providing a strong balance between precision and recall. Future work can focus on further refining predictions and deploying the model into a production environment for real-time monitoring.

---



## 📬 Contact
For any inquiries or collaboration opportunities, feel free to reach out!

📧 Email: [olonde.blex@gmail.com]  
[![LinkedIn Profile](https://th.bing.com/th/id/OIP.EpUtPNFJAX-rfRrKJtYHvgHaD4?rs=1&pid=ImgDetMain)](https://www.linkedin.com/in/blexolonde)

---



