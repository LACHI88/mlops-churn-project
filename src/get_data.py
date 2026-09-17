"""Étape 1 du pipeline : télécharger les données brutes."""
from pathlib import Path

import requests

from src.config import load_config
from src.utils import get_logger

logger = get_logger(__name__)


def get_data() -> Path:
    cfg = load_config()
    raw_path = Path(cfg.data.raw_path)
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    if raw_path.exists():
        logger.info("Fichier déjà présent : %s (pas de re-téléchargement)", raw_path)
        return raw_path

    logger.info("Téléchargement du dataset depuis %s", cfg.data.url)
    response = requests.get(cfg.data.url, timeout=30)
    response.raise_for_status()
    raw_path.write_bytes(response.content)
    logger.info("Dataset sauvegardé dans %s", raw_path)
    return raw_path


if __name__ == "__main__":
    get_data()
