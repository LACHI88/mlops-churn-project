PY=python
ENV?=.venv
EXP?=churn-exp

init:
	python3 -m venv $(ENV) && . $(ENV)/bin/activate && pip install -U pip -r requirements.txt

get-data:
	$(PY) -m src.get_data

preprocess:
	$(PY) -m src.preprocess

train:
	$(PY) -m src.train

evaluate:
	$(PY) -m src.evaluate

test:
	pytest -q

mlflow-ui:
	mlflow ui --backend-store-uri mlruns

api:
	uvicorn api.main:app --reload --port 8000

docker-build:
	docker build -t churn-mlops .

docker-run:
	docker run -p 8000:8000 churn-mlops

.PHONY: init get-data preprocess train evaluate test mlflow-ui api docker-build docker-run
