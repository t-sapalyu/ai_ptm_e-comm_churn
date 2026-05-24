.PHONY: help install install-dev format lint test clean prepare train evaluate \
        serve mlflow-ui docs docker-build docker-run drift rollback-list

PYTHON := python
PIP := pip

help:
	@echo "RetailGenius Churn — common targets"
	@echo ""
	@echo "  Setup"
	@echo "    install         Install runtime dependencies"
	@echo "    install-dev     Install runtime + dev dependencies + pre-commit hooks"
	@echo ""
	@echo "  Quality"
	@echo "    format          Run black and isort"
	@echo "    lint            Run flake8"
	@echo "    test            Run pytest"
	@echo ""
	@echo "  Pipeline"
	@echo "    prepare         Build train/val/test splits + reference snapshot"
	@echo "    train           Train all three models, log to MLflow"
	@echo "    evaluate        Evaluate Staging on test set, promote to Production"
	@echo "    serve           Serve Production model locally on port 5001"
	@echo "    mlflow-ui       Open MLflow UI on port 5000"
	@echo ""
	@echo "  Monitoring & operations"
	@echo "    drift           Run drift report (CURRENT=path/to/sample.parquet)"
	@echo "    rollback-list   List every registered model version"
	@echo ""
	@echo "  Docs & deploy"
	@echo "    docs            Build Sphinx HTML docs"
	@echo "    docker-build    Build the serving Docker image"
	@echo "    docker-run      Run the serving container on port 8080"
	@echo ""
	@echo "    clean           Remove caches"

install:
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

install-dev:
	$(PIP) install -r requirements-dev.txt
	$(PIP) install -e .
	pre-commit install

format:
	black src tests
	isort src tests

lint:
	flake8 src tests

test:
	pytest

prepare:
	$(PYTHON) -m churn.data.make_dataset

train:
	$(PYTHON) -m churn.models.train

evaluate:
	$(PYTHON) -m churn.models.evaluate

serve:
	mlflow models serve -m "models:/churn_classifier/Production" -p 5001 --no-conda

mlflow-ui:
	mlflow ui --port 5000

# Usage: make drift CURRENT=data/incoming/last_week.parquet
drift:
	@if [ -z "$(CURRENT)" ]; then \
		echo "Usage: make drift CURRENT=path/to/sample.parquet"; exit 1; \
	fi
	$(PYTHON) -m churn.monitoring.report --current $(CURRENT)

rollback-list:
	$(PYTHON) -m churn.models.rollback --list

docs:
	sphinx-build -b html docs docs/_build/html

docker-build:
	docker build -t retailgenius-churn:latest .

docker-run:
	docker run -p 8080:8080 retailgenius-churn:latest

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	rm -rf .coverage htmlcov docs/_build
