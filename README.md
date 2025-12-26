# Customer Churn Prediction (IBM Telco Dataset)

## Project Overview
Customer churn is a critical business problem in subscription-based industries such as **telecom, banking, and SaaS**.  
This project builds a **machine learning pipeline** to predict whether a customer is likely to churn using the **IBM Telco Customer Churn dataset**.  
The goal is to help businesses **identify high-risk customers early** and take proactive retention actions.

---

## Problem Statement
Predict whether a customer will **churn (leave the service)** based on demographic, service usage, and billing information.

---

## Solution Approach
1. Load and clean real-world telecom customer data  
2. Remove **data leakage features** (post-churn information)  
3. Encode categorical variables and scale numeric features  
4. Train a **Logistic Regression model** with class imbalance handling  
5. Evaluate performance using **business-relevant metrics**

---

## Dataset
- **Source**: IBM Telco Customer Churn Dataset  
- **Format**: Excel (`.xlsx`)  
- **Target Variable**: `Churn Value`  
  - `1` → Customer churned  
  - `0` → Customer retained  

### Data Leakage Handling
The following columns were **removed** to prevent data leakage:
- `Churn Label`
- `Churn Score`
- `Churn Reason`
- `CLTV`

Only features available **before churn** were used for prediction.

---

## Technologies Used
- Python  
- Pandas & NumPy  
- Scikit-learn  
- Matplotlib & Seaborn  
- Git & GitHub  

---

## Model Details
- **Algorithm**: Logistic Regression  
- **Imbalance Handling**: `class_weight="balanced"`  
- **Feature Scaling**: StandardScaler  

---

## Evaluation Metrics
Accuracy alone is misleading for churn problems.  
The following metrics were used:

| Metric | Purpose |
|--------|---------|
| Precision | Avoid unnecessary retention actions |
| Recall | Identify maximum churn-risk customers |
| F1-Score | Balance precision and recall |
| ROC-AUC | Overall classification quality |

---

## Results
- Model successfully identifies high-risk churn customers  
- Recall and F1-score prioritized over raw accuracy  
- Confusion matrix used for interpretability  

---

## Project Structure
```
customer-churn-prediction/
│
├── churn_model.py
├── README.md
├── requirements.txt
├── .gitignore
├── Confusion matrix.png
└── data/
    └── (dataset not tracked in git)
```

---

## How to Run the Project

### Clone the repository
```bash
git clone https://github.com/Rakshaa93/customer-churn-prediction.git
cd customer-churn-prediction
```

### Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate  # On Windows
# source venv/bin/activate  # On macOS/Linux
```

### Install dependencies
```bash
pip install -r requirements.txt
```

### Add dataset
Download the IBM Telco Customer Churn dataset and place it in:
```
data/Telco-Customer-Churn.xlsx
```

### Run the model
```bash
python churn_model.py
```

---

## Business Impact
- Enables early churn detection
- Supports customer retention strategies
- Reduces customer acquisition costs
- Applicable to telecom, SaaS, banking, and subscription businesses

---

## Key Learnings
- Importance of handling class imbalance
- Avoiding data leakage
- Choosing metrics based on business cost
- End-to-end ML pipeline design

---

## Future Improvements
- Try ensemble models (Random Forest, XGBoost)
- Add SHAP for model explainability
- Tune decision thresholds based on business cost
- Deploy as a REST API

---

## Contact
**Rakshaa**  
GitHub: [@Rakshaa93](https://github.com/Rakshaa93)

---

## License
This project is open source and available under the [MIT License](LICENSE).
