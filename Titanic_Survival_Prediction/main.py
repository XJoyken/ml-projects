import os
import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, Float, DateTime, VARCHAR, CHAR
from sqlalchemy.orm import declarative_base, sessionmaker
from dotenv import load_dotenv
from contextlib import asynccontextmanager
import mlflow.pyfunc
import pandas as pd

load_dotenv()

titanic_pipeline = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global titanic_pipeline
    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000'))
    model_uri = "models:/Titanic_Survival_Prediction/latest"

    try:
        print(f"Loading model from MLflow Registry {model_uri}.")
        titanic_pipeline = mlflow.pyfunc.load_model(model_uri=model_uri)
        print("Model loaded.")
    except Exception as e:
        print(f"Error: {e}")
        raise
    
    yield
    print('Closing...')

    if titanic_pipeline is not None:
        del titanic_pipeline
        print("Model deleted out of memory")
    print("Closed.")


DB_NAME = os.getenv('POSTGRES_DB')
DB_USER = os.getenv('POSTGRES_USER')
DB_PASSWORD = os.getenv('POSTGRES_PASSWORD')
DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')

DATABASE_URL = f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}'

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class PredictionLog(Base):
    __tablename__ = 'prediction_history'

    id = Column(Integer, primary_key=True, index=True)
    pclass = Column(Integer)
    sex = Column(VARCHAR(6))
    age = Column(Float)
    fare = Column(Float)
    cabin = Column(CHAR(1))
    embarked = Column(CHAR(1))
    title = Column(VARCHAR(10))
    group_size = Column(Float)
    family_size = Column(Float)
    prediction = Column(Integer)
    created_at = Column(DateTime, default=datetime.datetime.now(datetime.timezone.utc))

Base.metadata.create_all(bind=engine)

app = FastAPI(title="Titanic Survival Prediction", lifespan=lifespan)

class PredictRequest(BaseModel):
    pclass: int
    sex: str
    age: float
    fare: float
    cabin: str
    embarked: str
    title: str
    group_size: float
    family_size: float

@app.post("/predict")
async def predict(payload: PredictRequest):
    if titanic_pipeline is None:
        raise HTTPException(status_code=503, detail="Model is unavailable")
    try:
        df = pd.DataFrame([payload.model_dump()])
        prediction = titanic_pipeline.predict(df)
        db = SessionLocal()
        log_entry = PredictionLog(
            pclass=payload.pclass,
            sex=payload.sex,
            age=payload.age,
            fare=payload.fare,
            cabin=payload.cabin,
            embarked=payload.embarked,
            title=payload.title,
            group_size=payload.group_size,
            family_size=payload.family_size
        )
        db.add(log_entry)
        db.commit()
        db.refresh(log_entry)
        db.close()
            
        return {
            "status": "success",
            "prediction": prediction,
            "log_id": log_entry.id
        }
    except Exception as e:
        return HTTPException(status_code=400, detail=f'Error: {e}')