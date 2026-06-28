import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import os
from dotenv import load_dotenv

load_dotenv()

df = pd.read_csv('dataSP23.csv')

df = df.drop(columns=['name', 'id', 'host_id', 'host_name', 'neighbourhood', 'last_review', 'last_review_year', 'cut_available_365', 'cut_available_365_q', 'nights', 'minimum_nights', 'number_of_reviews', 'reviews_per_month'])

X = df.copy().drop(columns=['price'])
y = df.copy()['price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
    )

preprocessor = ColumnTransformer(
    transformers=[
        ('std', StandardScaler(), ['latitude', 'longitude', 'calculated_host_listings_count', 'availability_365']),
        ('ohe', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), ['neighbourhood_group']),
        ('ord', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), ['room_type'])
    ],
    remainder='passthrough'
)

model = LinearRegression(n_jobs=-1)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', model)
])

MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment('Rental_Price_AirBnb_Experiment')

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

    mlflow.log_metrics(metrics=metrics)
    reg_name = "Rental Price AirBnb Prediction"
    mlflow.sklearn.log_model(
        pipeline,
        name="rental_airbnb_pipeline",
        registered_model_name=reg_name
    )
    print(f"Run ID: {run.info.run_id}")
    print(f"Model registry name: {reg_name}")