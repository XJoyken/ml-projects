#!/bin/bash
pip install mlflow psycopg2-binary
mlflow server --backend-store-uri postgresql://$POSTGRES_USER:$POSTGRES_PASSWORD@db:5432/$MLFLOW_DB --artifacts-destination /data/mlruns --host 0.0.0.0 --port 5000 --serve-artifacts --allowed-hosts mlflow,mlflow:5000,localhost,localhost:5000,0.0.0.0