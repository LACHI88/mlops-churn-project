"""Étape 3 : construction d'un Pipeline scikit-learn (ColumnTransformer + modèle)."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import Config


def _get_model(model_type: str):
    if model_type == "random_forest":
        return RandomForestClassifier(random_state=42)
    if model_type == "logistic_regression":
        return LogisticRegression(max_iter=1000)
    raise ValueError(f"Type de modèle inconnu : {model_type}")


def build_pipeline(cfg: Config, X: pd.DataFrame) -> Pipeline:
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    model = _get_model(cfg.model.type)

    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
