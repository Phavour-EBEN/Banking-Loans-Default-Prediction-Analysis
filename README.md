# Loan Default Prediction

## Overview
This project focuses on predicting loan defaults using machine learning techniques. The dataset contains borrower details, financial metrics, and historical loan repayment records. The goal is to build models that can identify potential defaulters, helping financial institutions mitigate risks.

## Dataset
The dataset consists of 9578 entries with 14 features, including:
- **credit.policy**: Indicates if the customer meets credit underwriting criteria.
- **purpose**: The purpose of the loan.
- **int.rate**: Interest rate of the loan.
- **installment**: Monthly installment payment.
- **log.annual.inc**: Logarithm of the annual income.
- **dti**: Debt-to-income ratio.
- **fico**: FICO credit score.
- **days.with.cr.line**: Number of days with a credit line.
- **revol.bal**: Revolving balance.
- **revol.util**: Revolving line utilization rate.
- **inq.last.6mths**: Number of inquiries in the last 6 months.
- **delinq.2yrs**: Number of delinquencies in the past 2 years.
- **pub.rec**: Number of public records.
- **not.fully.paid**: Target variable (1 = not fully paid, 0 = fully paid).

## Data Preprocessing
- Categorical variable `purpose` was encoded using `LabelEncoder`.
- Checked for missing values and ensured data consistency.
- Features were analyzed for distributions using boxplots.

## Model Training
### Logistic Regression
- Baseline model for classification.
- Achieved an accuracy of **84.2%**.

### Random Forest Classifier
- Improved predictive performance using an ensemble approach.
- Accuracy increased slightly to **84.3%**.

### XGBoost Classifier
- Gradient boosting-based approach for better classification.
- Final model achieved **83.0% accuracy**, balancing precision and recall.

## Model Evaluation
- **Confusion Matrix**: Visualized predictions for each model.
- **Classification Report**: Precision, recall, and F1-score calculated.
- **F1 Score**: Balanced between precision and recall.

## Results
| Model  | Accuracy  | Precision | Recall | F1-Score |
|--------|----------|-----------|--------|----------|
| Logistic Regression | 84.2% | 84.3% | 99.8% | 91.4% |
| Random Forest | 84.3% | 84.3% | 99.9% | 91.4% |
| XGBoost | 83.0% | 84.8% | 97.2% | 90.6% |

## Installation & Usage
### Prerequisites
- Python 3.7+
- Required libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run the Model
```bash
python loan_prediction.py
```

## Future Improvements
- Experiment with hyperparameter tuning for improved accuracy.
- Implement deep learning models for better classification.
- Explore feature engineering techniques to improve predictive power.

## Conclusion
This project demonstrates the use of machine learning models for predicting loan defaults, highlighting the importance of data preprocessing, model selection, and evaluation techniques.

Feel free to explore, modify, and contribute!

