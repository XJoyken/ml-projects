# Heart Failure Prediction

This project aims to predict the likelihood of heart failure in patients based on various health parameters (e.g., age, blood pressure, smoking status). A Logistic Regression model was trained on a dataset containing patient health records and their corresponding outcomes.

To prioritize patient safety and accurately identify those at risk, the classification threshold was explicitly set to 0.25. This adjustment significantly boosts Recall, thereby minimizing Type II Errors (False Negatives) - ensuring that critical cases are not missed.

| Metric | Value |
| :--- | :--- |
| **Accuracy** | 0.6833 |
| **Precision** | 0.5000 |
| **Recall** | 0.9474 |
| **F1 score** | 0.6545 |