import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import mlflow
import mlflow.sklearn
import os
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


load_dotenv()

df = pd.read_csv('data/cleaned_train.csv')

df = df.set_index('PassengerId')
y = df['Survived']
X = df.copy().drop(columns=['Survived'])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,
    random_state=42,
    stratify=y
)

std_list = ['Age', 'Fare', 'GroupSize', 'FamilySize']
ordinal_list = ['Sex', 'Cabin']
ohe_list = ['Embarked', 'Title']

preprocessor = ColumnTransformer(
    transformers=[
        ('std', StandardScaler(), std_list),
        ('ord', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), ordinal_list),
        ('ohe', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), ohe_list)
    ],
    remainder='passthrough'
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LogisticRegression(max_iter=200))
])

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", 'http://localhost:5000')
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment('Titanic_Survival_Experiment')

with mlflow.start_run() as run:
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_probs = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_probs)
    }

    mlflow.log_metrics(metrics)

    mlflow.sklearn.log_model(
        pipeline, 
        name='titanic_pipeline',
        registered_model_name='Titanic_Survival_Prediction')
    print(f'Run ID: {run.info.run_id}')
    print("Model registy name: Titanic_Survival_Prediction")