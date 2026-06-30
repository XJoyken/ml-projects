from fastapi import FastAPI
from contextlib import asynccontextmanager
import os
import datetime
from dotenv import load_dotenv
from pydantic import BaseModel
from sqlalchemy import create_engine, Column
import mlflow.pyfunc
import pandas as pd
from train import REG_NAME

load_dotenv()

rental_pipeline = None

@asynccontextmanager
async def lifespan():
    global rental_pipeline
    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000'))
    model_uri = f'models/{REG_NAME}/latest'

    try:
        print(f'Loading model from uri: {model_uri}')
        rental_pipeline = mlflow.pyfunc.load_model(model_uri)
        print('Model loaded.')
    except Exception as e:
        print(f"Error: {e}")
        raise

    yield
    print('Closing...')

    if rental_pipeline is not None:
        del rental_pipeline
        print('Model deleted out of memory')
    print('Closed.')

DB_NAME = os.getenv('POSTGRES_DB')
DB_USER = os.getenv('POSTGRES_USER')
DB_PASSWORD = os.getenv('POSTGRES_PASSWORD')
DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')

DATABASE_URL = f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}'