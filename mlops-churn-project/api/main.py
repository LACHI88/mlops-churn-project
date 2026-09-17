"""API FastAPI qui sert le modèle enregistré dans le registry MLflow."""
from typing import Any

import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from src.config import load_config

cfg = load_config()
mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
MODEL = mlflow.sklearn.load_model(f"models:/{cfg.model.registry_name}/latest")

app = FastAPI(title="Churn Prediction API")


class PredictionRequest(BaseModel):
    features: dict[str, Any]


class PredictionResponse(BaseModel):
    churn_prediction: int
    churn_probability: float


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    df = pd.DataFrame([request.features])
    pred = int(MODEL.predict(df)[0])
    proba = float(MODEL.predict_proba(df)[0][1])
    return PredictionResponse(churn_prediction=pred, churn_probability=proba)
