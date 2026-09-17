"""Chargement de la configuration centralisée (YAML -> objet Python)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "config.yaml"


@dataclass
class DataConfig:
    url: str
    raw_path: str
    processed_path: str
    target: str
    id_column: str
    test_size: float
    random_state: int


@dataclass
class ModelConfig:
    type: str
    registry_name: str


@dataclass
class MlflowConfig:
    tracking_uri: str
    experiment_name: str


@dataclass
class TrainingConfig:
    cv_folds: int
    param_grid: dict[str, Any] = field(default_factory=dict)


@dataclass
class Config:
    data: DataConfig
    model: ModelConfig
    mlflow: MlflowConfig
    training: TrainingConfig


def load_config(path: str | Path = CONFIG_PATH) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Config(
        data=DataConfig(**raw["data"]),
        model=ModelConfig(**raw["model"]),
        mlflow=MlflowConfig(**raw["mlflow"]),
        training=TrainingConfig(**raw["training"]),
    )
