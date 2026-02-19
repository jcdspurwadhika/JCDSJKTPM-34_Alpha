# 🏦 Bank Marketing Campaign - Term Deposit Prediction

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0%2B-red)](https://streamlit.io/)

A machine learning project predicting term deposit subscriptions for bank marketing campaigns using advanced classification algorithms and imbalanced learning techniques.

## Author
- Fatimah Azzahra
- Tengku Arika Hazera
- Yonathan Hary Hutagalung

## Overview

This project is part of the **Purwadhika Data Science Bootcamp Final Capstone Project**. It aims to predict whether a bank client will subscribe to a term deposit based on demographic information, financial status, and previous campaign interactions.

### Key Objectives

- Maximize **Recall** (≥60%) to identify most potential subscribers
- Balance precision through **F2-Score** (≥0.55)
- Achieve **PR-AUC** (≥0.80) for reliable predictions on imbalanced data
- Optimize marketing campaign ROI through targeted client prioritization

## Business Problem

Banks invest significant resources in marketing campaigns for term deposits, but face several challenges:

1. Wasted resources on customers unlikely to subscribe
2. Inefficient allocation of call center time
3. Potential customer fatigue from excessive contact attempts
4. Suboptimal return on investment (ROI) from marketing campaigns

### Solution

`Primary objective`: Generating ML model to analyze customers' data features and build a predictive model to identify customers most likely to subscribe to term deposits, enabling the bank to:

- Optimize marketing campaign efficiency
- Reduce operational costs
- Improve customer experience by reducing unnecessary contacts
- Increase overall campaign success rate

`Success metrics`:

- Maximize Recall (minimize missed potential subscribers)
- Maintain acceptable Precision (avoid too many false positives)
- Optimize F2-Score (prioritize recall over precision)
- Achieve strong PR-AUC score for imbalanced data

## Stakeholders

`Primary stakeholders`:

1. Marketing Team: Use predictions to prioritize customer contacts
2. Call Center Operations: Optimize resource allocation
3. Bank Management: Strategic decision-making on campaign investments
4. Finance Team: Manages liquidity planning and funding risk based on projected deposit acquisition, especially during crisis and recovery periods

`Secondary stakeholders`:

5. Data Science Team: Model development and maintenance
6. Business Strategy/Product Team: Develops product offerings and customer segmentation strategies based on analytical insights
7. Customers: Benefit from an improved experience through more targeted and relevant communication

## Dataset

**Source**: [Bank Marketing Campaigns Dataset - Kaggle](https://www.kaggle.com/datasets/volodymyrgavrysh/bank-marketing-campaigns-dataset)

### Dataset Statistics
- **Total Records**: 41,188
- **Features**: 20
- **Target**: Binary (yes/no subscription)
- **Class Distribution**: Imbalanced (~11% positive class)

### Feature Categories

#### 1. Client Demographics
- Age, Job type, Marital status, Education level

#### 2. Financial Status
- Housing loan, Personal loan, Is default status known

#### 3. Campaign Contact Details
- Contact type, Month, Day of week

#### 4. Campaign Statistics
- Number of contacts, Days since last contact, Previous contacts, Previous outcome

#### 5. Macroeconomic Indicators
- Number of employment, consumer confidence index, 3 month EURIBOR Interest rate.

##  Methodology

### 1. Data Understanding & Cleaning
- Missing value analysis and imputation strategy
- Drop duplicates
- Outlier detection and treatment
- Feature type identification and conversion
- Data quality assessment

### 2. Exploratory Data Analysis
- Univariate and bivariate analysis
- Inferential statistics (Chi-square tests)
- Correlation analysis
- Target variable imbalance detection

### 3. Feature Engineering
- Transform `pdays` to binary `was_contacted_before`
- Transform `default` to binary `is default status known`
- Remove unimportant categorical features
- Create preprocessing pipelines and Transformer
- Use KNN inputing for better imputing result (n=3)
- Use One-Hot-Encodign for categorical encoding

### 4. Model Development

#### Models Benchmarked
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest
- Gradient Boosting
- AdaBoost
- CatBoost
- LightGBM
- XGBoost

#### Handling Class Imbalance
- Random Over-Sampling
- Random Under-Sampling
- Near Miss
- SMOTE (Synthetic Minority Over-sampling)
- SMOTEENN (Hybrid approach)
- Class Weight Balancing (For Available Model)

#### Optimization
- Hyperparameter tuning with GridSearchCV
- 5-fold cross-validation
- Custom F2-Score optimization

### 5. Model Evaluation
- Confusion Matrix analysis
- Precision-Recall trade-offs
- ROC-AUC and PR-AUC curves
- Cost-benefit analysis
- SHAP interpretability

## Results

### Model Performance
- **Recall**: Achieved target (60%)
- **F2-Score**: Achieved target (≥0.55)
- **PR-AUC**: Achieved target (≥0.80)

### Business Impact
   - Estimated net benefit: reduce operational calling time by -74.3% or almost 4 times more efficient or `337% more profit` gain compared to baseline approach
   - Enables targeted marketing campaigns with higher efficiency
   - Reduces unnecessary customer contacts improving satisfaction
   - Provides data-driven prioritization for call center operations

### Top Predictive Features
1. Macroeconomic Indicator: nr.employed, consumer confidence index and euribor 3m affects the decision and psychological factors of a person to take a term deposits
2. Age: It shows that slightly mature individuals likely to subscribe to term deposit (more than 30 yo)
3. Contact type: customer that called using telephone will likely not to subscribe

### Dependencies

- Python 3.8 or higher
- Miniconda
- Virtual environment / Miniconda (recommended)

### Features of Streamlit App
- **Home**: Homepage navigation into other section
- **Single Client Prediction**: Input client details for instant prediction
- **Batch Prediction**: Upload CSV for bulk predictions
- **Model Training**: Section to train the model
- **Model Documentation**: Learn about methodology and results
- **Data Dictionary**: Learn more about the dataset and what it means
- **Macroeconomic Analysis**: Learn about Macroeconomic behind the scene

##  Project Structure

```
bank-marketing-prediction/
│
├── bank_marketing_analysis.ipynb     # Complete analysis pipeline
├── app.py                            # Streamlit web application
├── README.md                         # This file
├── requirements.txt                  # Streamlit Requirement file 
├── raw_data.csv                      # Bank Marketing Campaingn unprocessed dataset
├── 01_bank_marketing_model.sav       # Final Model (Tuned LGBM)
└── Bank_campaign.twbx                # Tableau files
```

## Model Performance

### Classification Metrics
| Metric | Score | 
|--------|-------|
| Accuracy | 0.843 |
| Precision | 0.385 |
| **Recall** | 0.657 | 
| F1-Score | 0.485 |
| **F2-Score** | 0.576 |
| ROC-AUC | 0.482 |
| **PR-AUC** | 0.815 |


### Confusion Matrix Interpretation
- **True Positives (TP)**: Correctly identified subscribers
- **True Negatives (TN)**: Correctly identified non-subscribers
- **False Positives (FP)**: Incorrectly predicted subscriptions (wasted calls)
- **False Negatives (FN)**: Missed potential subscribers (lost revenue)

### Cost-Benefit Analysis
- **Cost per contact**: $5 (phone call + agent time)
- **Benefit per subscription**: $50 (term deposit profit)
- **Optimization**: Model minimizes FN (missed revenue) while managing FP (wasted costs)

## Technologies Used

### Core Libraries
- **Python 3.8+**: Programming language
- **Pandas & NumPy**: Data manipulation and analysis
- **Scikit-learn**: Machine learning algorithms and preprocessing
- **Imbalanced-learn**: Handling class imbalance

### Visualization
- **Matplotlib & Seaborn**: Static visualizations
- **Plotly**: Interactive charts for Streamlit app

### Machine Learning Models
- **Ensemble Methods**: Random Forest, Gradient Boosting, AdaBoost
- **Boosting Frameworks**: XGBoost, LightGBM, CatBoost
- **Linear Models**: Logistic Regression
- **Instance-based**: K-Nearest Neighbors

### Model Interpretation
- **SHAP**: Model explainability and feature importance

### Deployment
- **Streamlit**: Web application framework
- **Joblib**: Model serialization
