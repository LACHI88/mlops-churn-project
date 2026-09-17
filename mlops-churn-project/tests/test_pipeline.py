"""Sanity checks : le pipeline doit se construire et transformer les données correctement."""
import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import load_config
from src.pipeline import build_pipeline


def _dummy_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tenure": [1, 12, 24],
            "MonthlyCharges": [29.9, 56.5, 90.0],
            "Contract": ["Month-to-month", "One year", "Two year"],
            "gender": ["Female", "Male", "Female"],
        }
    )


def test_build_pipeline_returns_sklearn_pipeline():
    cfg = load_config()
    X = _dummy_df()
    pipeline = build_pipeline(cfg, X)
    assert isinstance(pipeline, Pipeline)
    assert "preprocessor" in pipeline.named_steps
    assert "model" in pipeline.named_steps


def test_pipeline_fit_predict_shapes():
    cfg = load_config()
    X = _dummy_df()
    y = pd.Series([0, 1, 0])
    pipeline = build_pipeline(cfg, X)
    pipeline.fit(X, y)
    preds = pipeline.predict(X)
    assert len(preds) == len(y)
