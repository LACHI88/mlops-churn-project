"""Étape 2 du pipeline : nettoyage minimal + sauvegarde des données traitées."""
from pathlib import Path

import pandas as pd

from src.config import load_config
from src.get_data import get_data
from src.utils import get_logger

logger = get_logger(__name__)


def preprocess() -> Path:
    cfg = load_config()
    raw_path = get_data()

    df = pd.read_csv(raw_path)

    # TotalCharges est parfois une chaîne avec des espaces vides -> conversion numérique
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # La colonne identifiant n'apporte rien au modèle
    if cfg.data.id_column in df.columns:
        df = df.drop(columns=[cfg.data.id_column])

    # Cible binaire 0/1
    df[cfg.data.target] = df[cfg.data.target].map({"Yes": 1, "No": 0}).fillna(df[cfg.data.target])

    processed_path = Path(cfg.data.processed_path)
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_path, index=False)
    logger.info("Données prétraitées sauvegardées dans %s (%d lignes)", processed_path, len(df))
    return processed_path


if __name__ == "__main__":
    preprocess()
