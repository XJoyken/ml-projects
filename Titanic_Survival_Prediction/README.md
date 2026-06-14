# Titanic Survival Prediction
This project aims to predict the survival of passengers on the Titanic using machine learning techniques. The dataset used for this project is the Titanic dataset, which contains information about the passengers, such as their age, gender, class, and other relevant features. The goal is to build a predictive model that can accurately classify whether a passenger survived or not based on these features.


Below is the summary table demonstrating the performance across key classification metrics: **Accuracy**, **Precision**, **Recall**, and **F1 Score**.

(Models are tuned)

| Model | Accuracy | Precision | Recall | F1 Score |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **0.8492** | **0.8182** | **0.7826** | **0.8000** |
| **SVM** | 0.7989 | 0.7797 | 0.6667 | 0.7188 |
| **Random Forest** | 0.7989 | 0.7619 | 0.6957 | 0.7273 |
| **Gradient Boosting** | 0.8156 | 0.7903 | 0.7101 | 0.7481 |
| **Decision Tree** | 0.7821 | 0.7500 | 0.6522 | 0.6977 |
| **Naive Bayes** | 0.7263 | 0.7941 | 0.3913 | 0.5243 |
| **KNN** | 0.7989 | 0.7705 | 0.6812 | 0.7231 |

---

### Key Insights & Observations

Logistic Regression showed the best result on all metrics, beating more complex models in predictive accuracy. This was helped by the fact that the dataset consists of only 891 data, and it also shows strong linear relationships between `Sex`/`Title`, and survival rates.
