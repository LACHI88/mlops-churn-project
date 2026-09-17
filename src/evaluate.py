"""Étape 5 : évaluation finale du modèle 'Production'/dernière version depuis le registry MLflow."""
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from src.config import load_config
from src.utils import get_logger

logger = get_logger(__name__)


def evaluate() -> None:
    cfg = load_config()
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)

    df = pd.read_csv(cfg.data.processed_path)
    X = df.drop(columns=[cfg.data.target])
    y = df[cfg.data.target]
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=cfg.data.test_size, random_state=cfg.data.random_state, stratify=y
    )

    model_uri = f"models:/{cfg.model.registry_name}/latest"
    logger.info("Chargement du modèle depuis le registry : %s", model_uri)
    model = mlflow.sklearn.load_model(model_uri)

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    logger.info("Rapport de classification final :\n%s", report)
    print(report)


if __name__ == "__main__":
    evaluate()
