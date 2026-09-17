"""Étape 4 : entraînement, tuning d'hyperparamètres, tracking et registry MLflow."""
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split

from src.config import load_config
from src.pipeline import build_pipeline
from src.preprocess import preprocess
from src.utils import get_logger

logger = get_logger(__name__)


def train() -> str:
    cfg = load_config()

    processed_path = preprocess()
    df = pd.read_csv(processed_path)

    X = df.drop(columns=[cfg.data.target])
    y = df[cfg.data.target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.data.test_size, random_state=cfg.data.random_state, stratify=y
    )

    pipeline = build_pipeline(cfg, X_train)
    param_grid = cfg.training.param_grid.get(cfg.model.type, {})

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)
    mlflow.sklearn.autolog(log_models=False)  # on logue le modèle nous-mêmes pour contrôler le nom

    with mlflow.start_run() as run:
        search = GridSearchCV(
            pipeline, param_grid=param_grid, cv=cfg.training.cv_folds, scoring="f1", n_jobs=-1
        )
        search.fit(X_train, y_train)
        best_model = search.best_estimator_

        y_pred = best_model.predict(X_test)
        metrics = {
            "test_accuracy": accuracy_score(y_test, y_pred),
            "test_precision": precision_score(y_test, y_pred),
            "test_recall": recall_score(y_test, y_pred),
            "test_f1": f1_score(y_test, y_pred),
        }
        mlflow.log_params(search.best_params_)
        mlflow.log_metrics(metrics)

        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
            registered_model_name=cfg.model.registry_name,
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )

        logger.info("Run MLflow : %s", run.info.run_id)
        logger.info("Meilleurs hyperparamètres : %s", search.best_params_)
        logger.info("Métriques test : %s", metrics)

        return run.info.run_id


if __name__ == "__main__":
    train()