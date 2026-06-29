from fastapi import FastAPI
from contextlib import asynccontextmanager
import os
import datetime
from dotenv import load_dotenv

load_dotenv()

rental_pipeline = None

@asynccontextmanager
async def lifespan():
    None