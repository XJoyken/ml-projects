import os
import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, Float, DateTime, VARCHAR, CHAR
from sqlalchemy.orm import declarative_base, sessionmaker

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

app = FastAPI(title="Titanic Survival Prediction")

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
def predict(payload: PredictRequest):
    None