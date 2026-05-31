# ai_ptm_e-comm_churn
**AI Project Technical Methodologies - Enterprise Customer Churn Prediction**

This repository contains a production-grade, containerized machine learning microservice designed to predict customer churn for RetailGenius. It features automated data drift monitoring, an MLflow model registry, and an Explainable AI (XAI) pipeline using SHAP.

## System Architecture
* **Modeling:** XGBoost, Random Forest, and Logistic Regression with automated hyperparameter tuning.
* **Tracking & Registry:** MLflow (local tracking server with automated Staging -> Production promotion).
* **Explainability (XAI):** SHAP TreeExplainer for global and local feature importance visualization.
* **Serving:** Containerized REST API via Docker.
* **Task Runner:** Automated pipeline execution via PowerShell (`tasks.ps1`).

---

## 1. Environment Setup

Clone the repository and set up the isolated virtual environment using `uv`. 

```powershell
# clone the repo
git clone <repo-url>
cd ai_ptm_e-comm_churn

# create the environment (locked to Python 3.11 for MLflow compatibility)
uv venv --python 3.11 aim_env
.\aim_env\Scripts\activate

# install all core, dev, and pre-commit dependencies
.\tasks.ps1 install-dev
```

## 2. Data Ingestion

Before running the pipeline, you must place the raw dataset into the correct directory.
Drop `E_Commerce_Dataset.xlsx` into `data\raw\`.

## 3. The Task Runner (`tasks.ps1`)

We use a Windows-native PowerShell task runner to standardize operations. Run `.\tasks.ps1 help` to see all available commands. 

**Code Quality & Pre-commit**
Code is strictly formatted to 79 characters using `black` and `isort`.
```powershell
.\tasks.ps1 format  # auto-formats the codebase
.\tasks.ps1 lint    # checks for PEP8 violations
.\tasks.ps1 test    # runs the pytest suite
```

## 4. Running the ML Pipeline

The machine learning lifecycle is broken down into sequential steps. Run these in order to train the models and promote the best one to Production.

```powershell
# 1. build train/val/test splits and data drift reference fingerprints
.\tasks.ps1 prepare

# 2. train all three model architectures and log metrics to MLflow
.\tasks.ps1 train

# 3. evaluate the Staging model against the test set and promote to Production
.\tasks.ps1 evaluate

# 4. generate global and local SHAP explainability plots (saved to reports/shap_plots)
.\tasks.ps1 explain
```

To view the training curves, parameters, and model artifacts, boot up the MLflow UI:
```powershell
.\tasks.ps1 mlflow-ui
# browse to [http://127.0.0.1:5000](http://127.0.0.1:5000)
```

## 5. Production Serving (Docker)

Do not serve the model using the local MLflow command in production. Instead, build and run the isolated Linux container.

```powershell
# build the serving image
.\tasks.ps1 docker-build

# spin up the container API
.\tasks.ps1 docker-run
```
The inference server will now be listening on port `8080`. 

**Testing the Endpoint:**
In a second terminal, send a test payload to the active container to verify predictions:
```powershell
python -m churn.inference.request
```

## 6. Operations & Monitoring

**Data Drift:**
To check new incoming data against the training baseline for data drift:
```powershell
.\tasks.ps1 drift <path-to-new-data.parquet>
```

**Rollbacks:**
If a deployment degrades, list the archived models and roll back to a stable version:
```powershell
.\tasks.ps1 rollback-list
python -m churn.models.rollback --version <version_number>
```