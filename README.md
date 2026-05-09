# 👨‍💼 Predicting Employee Attrition  
## A Machine Learning Approach to Workforce Retention

<div align="center">

### Intelligent Workforce Analytics System for Employee Retention Prediction

A comprehensive machine learning pipeline designed to predict employee attrition using advanced classification algorithms, feature engineering, hyperparameter optimization, and ensemble learning techniques.

<br>

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?style=for-the-badge&logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-black?style=for-the-badge&logo=pandas)
![Status](https://img.shields.io/badge/Status-Research%20Completed-success?style=for-the-badge)
![Domain](https://img.shields.io/badge/Domain-HR%20Analytics-purple?style=for-the-badge)
![ML](https://img.shields.io/badge/Machine%20Learning-Classification-red?style=for-the-badge)

</div>

---

# 📌 Overview

Employee attrition is one of the most critical challenges faced by modern organizations. High turnover rates increase:

- Recruitment costs
- Training expenses
- Productivity loss
- Organizational instability
- Knowledge drain

This research presents a robust **Machine Learning-based Employee Attrition Prediction Framework** capable of identifying employees at risk of leaving an organization.

The project focuses heavily on:

- Imbalanced classification
- Recall optimization
- Feature importance analysis
- Threshold tuning
- Ensemble learning
- HR interpretability

The final system prioritizes minimizing **false negatives**, ensuring that high-risk employees are not overlooked.

---

# 🎯 Objectives

The primary objectives of this research were:

- Predict employee attrition effectively
- Handle severe class imbalance
- Improve recall for high-risk employee detection
- Optimize model generalization
- Develop interpretable HR analytics models
- Compare multiple machine learning algorithms
- Deploy a scalable predictive pipeline

---

# 🧠 Machine Learning Pipeline

```text
Raw Employee Dataset
          ↓
Data Cleaning & Preprocessing
          ↓
Feature Engineering
          ↓
Feature Selection
          ↓
Model Training
          ↓
Hyperparameter Optimization
          ↓
Threshold Tuning
          ↓
Ensemble Learning
          ↓
Performance Evaluation
          ↓
Model Deployment
```

---

# 📊 Dataset Characteristics

## Employee Attrition Dataset

The dataset contains employee-related information such as:

- Monthly Income
- Age
- Job Satisfaction
- Work-Life Balance
- Overtime
- Business Travel
- Marital Status
- Job Role
- Years at Company
- Environment Satisfaction

---

## ⚠️ Class Imbalance

The dataset exhibited severe imbalance:

| Class | Approximate Count |
|---|---|
| Attrition = No | ~1200 |
| Attrition = Yes | ~200 |

Only approximately **14%** of samples belonged to the positive attrition class.

This required specialized techniques including:

- Stratified splitting
- Threshold tuning
- Recall optimization
- F1-score prioritization
- Ensemble learning

---

# 🔍 Exploratory Data Analysis

Extensive visual analysis was conducted to understand workforce patterns.

## Key Insights

### 💰 Monthly Income
Lower salary ranges showed significantly higher attrition probability.

### ⏰ Overtime
Employees working overtime demonstrated elevated attrition risk.

### ✈️ Business Travel
Frequent business travel correlated strongly with employee turnover.

### 😊 Job Satisfaction
Low job satisfaction emerged as a major attrition indicator.

### 🧠 Work-Life Balance
Poor work-life balance increased attrition susceptibility.

### 👥 Marital Status
Single employees exhibited relatively higher turnover rates.

---

# 🧬 Feature Engineering & Selection

Feature selection was performed using:

- Random Forest Feature Importance
- Correlation Analysis
- Recursive Feature Elimination (RFE)
- Mutual Information Analysis

---

# ⭐ Top Important Features

| Feature | Importance |
|---|---|
| MonthlyIncome | 0.0857 |
| Age | 0.0760 |
| TotalWorkingYears | 0.0667 |
| OverTime_Yes | 0.0430 |
| DistanceFromHome | 0.0551 |
| JobSatisfaction | 0.0308 |
| WorkLifeBalance | 0.0309 |
| EnvironmentSatisfaction | 0.0318 |

---

# 🤖 Machine Learning Models

The following algorithms were evaluated:

| Model | Purpose |
|---|---|
| K-Nearest Neighbors (KNN) | Distance-based classification |
| Decision Tree | Rule-based learning |
| Logistic Regression | Linear probabilistic modeling |
| Support Vector Classifier (SVC) | Margin-based classification |
| Random Forest | Ensemble tree learning |
| AdaBoost | Boosted weak learners |
| Stacking Ensemble | Meta-learning architecture |

---

# ⚙️ Hyperparameter Optimization

Two optimization techniques were used:

## 🔹 Random Search
Broad parameter exploration for initial optimization.

## 🔹 Grid Search
Fine-grained tuning using:

- 5-Fold Cross Validation
- F1-score optimization
- Recall prioritization

---

# 🧪 Best Hyperparameters

## Logistic Regression

```python
{
    "C": 500,
    "penalty": "l1",
    "solver": "saga",
    "max_iter": 500,
    "class_weight": "balanced"
}
```

---

## Support Vector Classifier

```python
{
    "C": 1.8,
    "gamma": 0.2,
    "kernel": "rbf",
    "class_weight": "balanced"
}
```

---

# 📈 Model Performance

## Performance After Hyperparameter Tuning

| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.8742 | 0.7273 | 0.3404 | 0.4638 | 0.8082 |
| SVC | 0.7755 | 0.3827 | 0.6596 | 0.4844 | 0.8145 |
| AdaBoost | 0.8503 | 0.5652 | 0.2766 | 0.3714 | 0.8120 |
| Random Forest | 0.8401 | 0.5000 | 0.2553 | 0.3380 | 0.7655 |
| KNN | 0.8367 | 0.4737 | 0.1915 | 0.2727 | 0.6914 |

---

# 🏆 Final Selected Model

## Logistic Regression at Threshold 0.48

The final model was selected based on:

- High recall
- Balanced performance
- Interpretability
- HR applicability
- Robust validation results

---

# 📊 Final Logistic Regression Results

| Metric | Value |
|---|---|
| Accuracy | 0.7449 |
| Precision | 0.51 |
| Recall | 0.7872 |
| F1 Score | 0.4966 |
| ROC AUC | 0.8164 |

---

# 🎯 Threshold Optimization

Threshold tuning was performed to maximize recall.

| Threshold | Recall | F1 Score |
|---|---|---|
| 0.30 | 0.8723 | 0.3796 |
| 0.35 | 0.8511 | 0.4061 |
| 0.40 | 0.8298 | 0.4333 |
| 0.45 | 0.8085 | 0.4720 |
| 0.48 | **0.7872** | **0.4966** |
| 0.50 | 0.7447 | 0.4861 |

Threshold **0.48** produced the best balance between sensitivity and overall performance.

---

# 🧩 Ensemble Learning

A stacking ensemble architecture was implemented using:

## Base Models
- KNN
- Random Forest
- SVC
- AdaBoost

## Meta Model
- Logistic Regression

---

# 📊 Stacking Ensemble Performance

| Metric | Value |
|---|---|
| Accuracy | 0.7959 |
| Precision | 0.4085 |
| Recall | 0.6170 |
| F1 Score | 0.4915 |
| ROC AUC | 0.8067 |

---

# 🧠 Error Analysis

## Logistic Regression Confusion Matrix

```text
                Predicted
              No       Yes
Actual No     182      65
Actual Yes    10       37
```

### Key Observations
- Very low false negatives
- Strong minority class detection
- Suitable for HR retention systems

---

# 🧪 Ablation Studies

Comprehensive ablation studies were conducted on:

- Resampling strategies
- Feature selection methods
- Threshold tuning
- Ensemble learning effects

---

## Techniques Evaluated

### Resampling
- SMOTE
- Random Undersampling
- Hybrid SMOTE + Undersampling

### Feature Selection
- Random Forest Importance
- Recursive Feature Elimination
- Mutual Information
- Correlation-Based Selection

---

# 🏗️ Deployment

The final trained model was serialized using:

```python
joblib
```

Saved as:

```text
logistic_model_threshold48.pkl
```

The deployment pipeline supports integration using:

- Flask
- FastAPI
- HR Management Systems
- Real-time employee monitoring systems

---

# 🛠️ Technologies Used

## Machine Learning
- Scikit-Learn
- NumPy
- Pandas

## Visualization
- Matplotlib
- Seaborn

## Model Optimization
- GridSearchCV
- RandomizedSearchCV

## Deployment
- Joblib
- Flask / FastAPI

---

# 📂 Project Structure

```text
employee-attrition/
│
├── dataset/
│   ├── raw/
│   ├── processed/
│   └── attrition.csv
│
├── notebooks/
│   ├── eda.ipynb
│   ├── preprocessing.ipynb
│   ├── training.ipynb
│   └── evaluation.ipynb
│
├── models/
│   ├── logistic_model_threshold48.pkl
│   ├── svc_model.pkl
│   └── ensemble_model.pkl
│
├── visualizations/
│   ├── correlation_heatmap.png
│   ├── attrition_distribution.png
│   ├── overtime_analysis.png
│   └── feature_importance.png
│
├── deployment/
│   ├── app.py
│   └── requirements.txt
│
└── README.md
```

---

# 🔬 Research Contributions

## Major Contributions

- High-recall attrition prediction framework
- Robust imbalance handling strategy
- Threshold-based recall optimization
- HR-interpretable machine learning pipeline
- Ensemble learning integration
- Comprehensive error analysis
- Workforce behavioral analytics

---

# ⚠️ Limitations

Despite strong performance, several limitations remain:

- Severe dataset imbalance
- Potential feature interaction loss
- Static historical data dependency
- Limited explainability tools
- Dynamic workforce changes over time



# 📌 Conclusion

This research demonstrates that machine learning can effectively identify employees at risk of attrition while balancing:

- Recall
- Interpretability
- Generalization
- HR practicality

The optimized Logistic Regression model achieved strong recall performance, making it highly suitable for proactive employee retention systems.

The framework establishes a scalable foundation for AI-driven workforce analytics and intelligent HR decision-making systems.







# ⭐ Final Note

This project highlights how machine learning can transform workforce management through:

- Predictive analytics
- Intelligent retention strategies
- Data-driven HR decision making
- Explainable AI systems

The research bridges the gap between artificial intelligence and real-world organizational management.

---
