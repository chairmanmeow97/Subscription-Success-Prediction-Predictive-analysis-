# Predicting Subscription Success in Bank Telemarketing Campaigns

## 📌 Overview
This project applies machine learning techniques to predict customer subscription behavior for bank telemarketing campaigns. The analysis compares multiple classification algorithms to identify the most effective approach for targeting potential subscribers and optimizing marketing resource allocation.
The study addresses a critical business challenge: identifying high-probability customers for term deposit subscriptions to improve campaign efficiency and ROI.

## 📊 Dataset Summary
The project utilizes a split dataset consisting of:
* **Training Data**: 31,649 observations.
* **Test Data**: 13,562 observations.
* **Variables**: 17 predictors including Client Information (Age, Job, Balance), Contact Data, and Previous Campaign Outcomes.
* **Target Variable**: Binary outcome (Yes/No subscription).
<img width="745" height="209" alt="image" src="https://github.com/user-attachments/assets/c062b3ee-62bd-4c8e-8054-46ae071f5748" />
<img width="584" height="725" alt="image" src="https://github.com/user-attachments/assets/908c741e-f0da-4ae6-8422-f3eb1e054b25" />

## 🛠️ Tools & Technologies
Language: R programming language
Algorithms: Random Forest, XGBoost, Naive Bayes
Data Processing: SMOTE for class imbalance, one-hot encoding, feature scaling
Evaluation Metrics: Accuracy, Precision, Recall, F1 Score, ROC-AUC, Confusion Matrix

### Data Preprocessing
1. **Imbalance Treatment**: Addressed a subscription rate of only 11.7% in the training set using **SMOTE** (Synthetic Minority Over-sampling Technique) to improve model fairness.
2. **Categorical Encoding**: Utilized **One-hot encoding** to transform categorical variables into binary features for machine learning tasks.
3. **Numerical Scaling**: Standardized features to address range differences (e.g., between Age and Income) to ensure optimal performance for algorithms like XGBoost.

**B. Model Development**
1. **Random Forest**: Ensemble method for capturing non-linear relationships and feature interpretability.
2. **XGBoost**: Gradient boosting for superior predictive performance and handling imbalanced data.
3. **Naive Bayes**: Probabilistic baseline for comparison and efficiency benchmark.
   
## 🤖 Model Development
Three classification algorithms were explored and evaluated:
* **XGBoost**: An advanced gradient boosting library optimized for speed, memory, and handling imbalanced datasets.
* **Random Forest**: A meta estimator that uses averaging across multiple decision trees to improve accuracy and prevent overfitting.
* **Naive Bayes**: A probabilistic classifier based on Bayes' theorem used as a baseline

## 📈 Performance Results
| Metric | XGBoost | Random Forest | Naive Bayes |
|--------|---------|---------------|------------|
| Accuracy | 91.08% | 97.2% | 88.28% |
| ROC-AUC | 0.935 | 0.93 | 0.86 |
| Recall (Sensitivity) | 44.83% | 42.2% | 0% |
| F1 Score | Best | Very Good | Poor |
| **Winner** | ✓ Best Overall | Strong Second | Baseline Only |
<img width="399" height="290" alt="image" src="https://github.com/user-attachments/assets/d50e443d-9b43-4686-a0ff-01716c63565e" />
<img width="444" height="323" alt="image" src="https://github.com/user-attachments/assets/5f93ee93-a4a1-426e-9a0b-96e55649a916" />
<img width="411" height="299" alt="image" src="https://github.com/user-attachments/assets/eb2e2196-87f4-457f-b201-30a8371d0263" />
<img width="537" height="386" alt="image" src="https://github.com/user-attachments/assets/b86444c4-5789-4dd7-99f2-9e93c86799f2" />

**Key Finding:** XGBoost emerged as the superior model, achieving the best balance between precision and recall while minimizing false negatives—critical for identifying actual subscribers.

### 🎯 Feature Importance
The following predictors had the strongest influence on subscription likelihood:

1. **Duration** — Most significant; longer conversations strongly correlate with subscription
2. **Previous Outcome (Success)** — Customers successfully contacted previously are more likely to resubscribe
3. **Month of Last Contact** — March and October showed higher conversion rates
4. **Contact Type** — Specific contact methods influence subscription probability

### 💡 Actionable Insights

- XGBoost is recommended for production deployment due to superior predictive performance
- Training telemarketers to maximize call engagement duration can significantly improve conversion rates
- Targeting customers with positive previous interactions yields higher success rates
- Seasonal patterns in contact timing present optimization opportunities

## 📌 Conclusion

This analysis demonstrates that **advanced tree-based models (XGBoost and Random Forest) substantially outperform simpler probabilistic approaches** for telemarketing subscription prediction.

By implementing XGBoost, the bank can:
- **Improve campaign targeting** through accurate subscriber identification
- **Reduce operational costs** by focusing resources on high-probability prospects
- **Enhance conversion rates** by leveraging feature insights (call duration, previous outcomes)
- **Achieve competitive advantage** in customer acquisition

The predictive framework provides a data-driven foundation for optimizing telemarketing strategies and resource allocation in financial services marketing.
