from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import sklearn
import xgboost
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from churn import config
from churn.features.build_features import build_preprocessor

logger = logging.getLogger(__name__)


# Model factory: param name str is One of ``"logreg"``, ``"random_forest"``, ``"xgboost"``
def get_model(name: str) -> tuple[Any, dict[str, Any]]:
    
    if name == "logreg":
        params = {
            "C": 1.0,
            "max_iter": 1000,
            "class_weight": "balanced",
            "random_state": config.RANDOM_STATE,
        }
        return LogisticRegression(**params), params

    if name == "random_forest":
        params = {
            "n_estimators": 300,
            "max_depth": 12,
            "min_samples_leaf": 2,
            "class_weight": "balanced",
            "n_jobs": -1,
            "random_state": config.RANDOM_STATE,
        }
        return RandomForestClassifier(**params), params

    if name == "xgboost":
        # scale_pos_weight handles class imbalance: ratio of negatives to
        # positives in the training set (~83/17 -> ~4.94).
        params = {
            "n_estimators": 400,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "scale_pos_weight": 4.94,
            "eval_metric": "logloss",
            "n_jobs": -1,
            "random_state": config.RANDOM_STATE,
        }
        return XGBClassifier(**params), params

    raise ValueError(f"Unknown model name: {name!r}")


# Reproducibility helpers
def get_git_commit() -> str:
    """Return the current git commit hash, or 'unknown' if not in a git repo."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=config.PROJECT_ROOT,
            stderr=subprocess.DEVNULL,
        )
        return out.decode("utf-8").strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def load_fingerprints() -> dict[str, str]:
    """Load the dataset fingerprints written by ``make_dataset.py``."""
    fp_path = config.PROCESSED_DIR / "fingerprints.json"
    if not fp_path.exists():
        return {}
    with fp_path.open() as fh:
        return json.load(fh)


def load_splits() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    if not config.TRAIN_FILE.exists():
        raise FileNotFoundError(
            f"{config.TRAIN_FILE} not found. " "Run `python -m churn.data.make_dataset` first."
        )
    train_df = pd.read_parquet(config.TRAIN_FILE)
    val_df = pd.read_parquet(config.VAL_FILE)

    x_train = train_df[config.ALL_FEATURES]
    y_train = train_df[config.TARGET]
    x_val = val_df[config.ALL_FEATURES]
    y_val = val_df[config.TARGET]

    return x_train, y_train, x_val, y_val


def compute_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> dict[str, float]:
    """Compute the metrics tracked by the KPI table."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


def train_one(
    model_name: str,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
) -> tuple[str, float]:
    """Train one model inside a nested MLflow run.

    Returns
    -------
    tuple
        ``(model_version, val_f1)`` — the registered model version (as a
        string) and the validation F1 used to select the best run.
    """
    estimator, params = get_model(model_name)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", estimator),
        ]
    )

    with mlflow.start_run(run_name=model_name, nested=True) as run:
        mlflow.set_tag("model_family", model_name)
        mlflow.log_params(params)
        mlflow.log_param("n_train_rows", len(x_train))
        mlflow.log_param("n_val_rows", len(x_val))

        logger.info("Training %s ...", model_name)
        pipeline.fit(x_train, y_train)

        y_pred = pipeline.predict(x_val)
        y_proba = pipeline.predict_proba(x_val)[:, 1]

        metrics = compute_metrics(y_val, y_pred, y_proba)
        mlflow.log_metrics(metrics)
        logger.info(
            "%s val metrics — acc=%.3f prec=%.3f rec=%.3f f1=%.3f auc=%.3f",
            model_name,
            metrics["accuracy"],
            metrics["precision"],
            metrics["recall"],
            metrics["f1"],
            metrics["roc_auc"],
        )

        signature = infer_signature(x_val, y_pred)
        input_example = x_val.head(5)

        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            registered_model_name=config.REGISTERED_MODEL_NAME,
            signature=signature,
            input_example=input_example,
        )

        # Find the version that was just registered for this run
        client = MlflowClient()
        versions = client.search_model_versions(f"name='{config.REGISTERED_MODEL_NAME}'")
        version_for_run = next(v.version for v in versions if v.run_id == run.info.run_id)

        return version_for_run, metrics["f1"]


def promote_best_to_staging(results: dict[str, tuple[str, float]]) -> None:
    """Transition the registered version with the best val F1 to ``Staging``.

    Any previous ``Staging`` version is archived so we always have exactly one.
    """
    best_model = max(results, key=lambda name: results[name][1])
    best_version, best_f1 = results[best_model]

    client = MlflowClient()
    for v in client.search_model_versions(f"name='{config.REGISTERED_MODEL_NAME}'"):
        if v.current_stage == "Staging" and v.version != best_version:
            client.transition_model_version_stage(
                name=config.REGISTERED_MODEL_NAME,
                version=v.version,
                stage="Archived",
            )

    client.transition_model_version_stage(
        name=config.REGISTERED_MODEL_NAME,
        version=best_version,
        stage="Staging",
    )
    logger.info(
        "Promoted %s (version %s, val F1=%.3f) to Staging",
        best_model,
        best_version,
        best_f1,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=config.MODELS_TO_TRAIN + ["all"],
        default="all",
        help="Which model(s) to train. Defaults to all three.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)

    x_train, y_train, x_val, y_val = load_splits()
    fingerprints = load_fingerprints()
    git_commit = get_git_commit()

    models_to_run = config.MODELS_TO_TRAIN if args.model == "all" else [args.model]

    results: dict[str, tuple[str, float]] = {}
    with mlflow.start_run(run_name="train_all"):
        # Reproducibility tags / params on the parent run
        mlflow.set_tag("git.commit", git_commit)
        mlflow.set_tag("dataset.train_sha256", fingerprints.get("train", "unknown"))
        mlflow.set_tag("dataset.val_sha256", fingerprints.get("val", "unknown"))
        mlflow.set_tag("dataset.test_sha256", fingerprints.get("test", "unknown"))
        mlflow.log_param("sklearn_version", sklearn.__version__)
        mlflow.log_param("xgboost_version", xgboost.__version__)
        mlflow.log_param("random_state", config.RANDOM_STATE)
        mlflow.log_param("n_models", len(models_to_run))
        mlflow.log_param("models", ",".join(models_to_run))

        for name in models_to_run:
            version, f1 = train_one(name, x_train, y_train, x_val, y_val)
            results[name] = (version, f1)

    if args.model == "all":
        promote_best_to_staging(results)

    return 0


if __name__ == "__main__":
    sys.exit(main())
