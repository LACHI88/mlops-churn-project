# Pipeline MLOps — Prédiction de Churn Client

Projet final du cours MLOps (M2 Campus Cyber). Pipeline complet, reproductible et versionné, du téléchargement des données jusqu'au service du modèle via une API.

## Dataset

[Telco Customer Churn](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv) — prédiction binaire (le client va-t-il résilier son abonnement ?), dataset tabulaire mixte (numérique + catégoriel).

## Architecture du repo

```
mlops-churn-project/
├── configs/
│   └── config.yaml          # toute la config centralisée (chemins, hyperparamètres, MLflow)
├── src/
│   ├── config.py             # chargement typé du YAML
│   ├── get_data.py           # téléchargement des données brutes
│   ├── preprocess.py         # nettoyage minimal
│   ├── pipeline.py           # ColumnTransformer + modèle scikit-learn
│   ├── train.py               # entraînement + GridSearchCV + tracking/registry MLflow
│   ├── evaluate.py            # évaluation finale depuis le modèle enregistré
│   └── utils.py
├── api/
│   └── main.py                 # API FastAPI de service du modèle
├── tests/
│   └── test_pipeline.py        # sanity checks pytest
├── Makefile
├── Dockerfile
└── requirements.txt
```

## Bonnes pratiques MLOps appliquées

- **Reproductibilité** : pipeline en étapes séquentielles indépendantes (get-data → preprocess → train → evaluate)
- **Configuration centralisée** : aucun paramètre en dur, tout est dans `configs/config.yaml`
- **Pipeline scikit-learn unique** : `ColumnTransformer` + modèle, pour garantir que le preprocessing est identique à l'entraînement et à l'inférence
- **Tracking d'expériences** : MLflow autologging + logging manuel des métriques de test (accuracy, precision, recall, F1)
- **Tuning d'hyperparamètres** : `GridSearchCV` avec grille définie dans la config
- **Model Registry** : chaque run entraîné est enregistré sous un nom versionné (`churn-classifier`) ; l'API et l'évaluation chargent toujours la dernière version depuis le registry, jamais un fichier local codé en dur
- **Tests automatisés** : `pytest` vérifie que le pipeline se construit et s'entraîne correctement
- **Service du modèle** : API FastAPI avec endpoint `/predict`, découplée de l'entraînement
- **Conteneurisation** : `Dockerfile` qui installe les dépendances, entraîne le modèle, puis sert l'API
- **Orchestration** : `Makefile` avec une cible par étape, tout tourne en local sans dépendance cloud payante

## Utilisation

```bash
# 1. Environnement
make init
source .venv/bin/activate

# 2. Pipeline complet
make get-data
make preprocess
make train        # entraîne + logue dans MLflow + enregistre dans le registry

# 3. Visualiser les expériences
make mlflow-ui    # http://localhost:5000

# 4. Évaluation finale
make evaluate

# 5. Tests
make test

# 6. Servir le modèle
make api          # http://localhost:8000/docs

# 7. (optionnel) conteneuriser
make docker-build
make docker-run
```

## Exemple d'appel à l'API

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {"tenure": 12, "MonthlyCharges": 70.5, "Contract": "Month-to-month", "gender": "Female", ...}}'
```
