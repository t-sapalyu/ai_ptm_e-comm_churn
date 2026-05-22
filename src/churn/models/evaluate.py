"""Evaluate the Staging model on the held-out test set and promote to Production.

Run with::

    python -m churn.models.evaluate

The script loads the model currently in the ``Staging`` stage of the
``churn_classifier`` registry, scores the test set, writes a classification
report and ROC curve PNG to ``reports/``, and — if the test F1 meets the
configured threshold — transitions the model to ``Production``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # must come before pyplot import; non-interactive backend for CI

import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from churn import config  # noqa: E402

logger = logging.getLogger(__name__)


def load_test_set() -> tuple[pd.DataFrame, pd.Series]:
    """Load the held-out test split."""
    if not config.TEST_FILE.exists():
        raise FileNotFoundError(
            f"{config.TEST_FILE} not found. Run `python -m churn.data.make_dataset` first."
        )
    test_df = pd.read_parquet(config.TEST_FILE)
    return test_df[config.ALL_FEATURES], test_df[config.TARGET]


def load_staging_model() -> tuple[object, str]:
    """Load the Staging model and return both the model and its version string."""
    client = MlflowClient()
    versions = client.get_latest_versions(config.REGISTERED_MODEL_NAME, stages=["Staging"])
    if not versions:
        raise RuntimeError(
            f"No '{config.REGISTERED_MODEL_NAME}' model in Staging. "
            "Run `python -m churn.models.train` first."
        )
    version = versions[0].version
    uri = f"models:/{config.REGISTERED_MODEL_NAME}/{version}"
    logger.info("Loading model from %s", uri)
    model = mlflow.sklearn.load_model(uri)
    return model, version


def save_artifacts(y_true: pd.Series, y_pred, y_proba, output_dir: Path) -> dict[str, str]:
    """Write the classification report, confusion matrix, and ROC curve.

    Returns a dict of artifact name -> file path (as strings) for MLflow logging.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, str] = {}

    # 1. Classification report as JSON
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    report_path = output_dir / "classification_report.json"
    with report_path.open("w") as fh:
        json.dump(report, fh, indent=2)
    artifacts["classification_report"] = str(report_path)

    # 2. Confusion matrix
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax, cmap="Blues")
    ax.set_title("Confusion matrix — test set")
    fig.tight_layout()
    cm_path = output_dir / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=120)
    plt.close(fig)
    artifacts["confusion_matrix"] = str(cm_path)

    # 3. ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.lineplot(x=fpr, y=tpr, ax=ax, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve — test set")
    ax.legend(loc="lower right")
    fig.tight_layout()
    roc_path = output_dir / "roc_curve.png"
    fig.savefig(roc_path, dpi=120)
    plt.close(fig)
    artifacts["roc_curve"] = str(roc_path)

    return artifacts


def maybe_promote_to_production(version: str, f1: float, threshold: float) -> bool:
    """Transition the Staging version to Production if F1 ≥ threshold.

    Returns True if a transition was performed.
    """
    if f1 < threshold:
        logger.info(
            "Test F1=%.3f is below promotion threshold %.3f — staying in Staging.",
            f1,
            threshold,
        )
        return False

    client = MlflowClient()
    # Archive any existing Production version first
    for v in client.search_model_versions(f"name='{config.REGISTERED_MODEL_NAME}'"):
        if v.current_stage == "Production" and v.version != version:
            client.transition_model_version_stage(
                name=config.REGISTERED_MODEL_NAME,
                version=v.version,
                stage="Archived",
            )

    client.transition_model_version_stage(
        name=config.REGISTERED_MODEL_NAME,
        version=version,
        stage="Production",
    )
    logger.info(
        "Promoted version %s of %s to Production (test F1=%.3f).",
        version,
        config.REGISTERED_MODEL_NAME,
        f1,
    )
    return True


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--threshold",
        type=float,
        default=config.PROMOTION_F1_THRESHOLD,
        help="Minimum test F1 required to promote Staging -> Production.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)

    x_test, y_test = load_test_set()
    model, version = load_staging_model()

    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)[:, 1]

    metrics = {
        "test_accuracy": accuracy_score(y_test, y_pred),
        "test_precision": precision_score(y_test, y_pred, zero_division=0),
        "test_recall": recall_score(y_test, y_pred, zero_division=0),
        "test_f1": f1_score(y_test, y_pred, zero_division=0),
        "test_roc_auc": roc_auc_score(y_test, y_proba),
    }

    logger.info("Test metrics: %s", {k: round(v, 4) for k, v in metrics.items()})

    artifacts = save_artifacts(y_test, y_pred, y_proba, config.REPORTS_DIR)

    # Log the evaluation as its own MLflow run for traceability
    with mlflow.start_run(run_name=f"evaluate_v{version}"):
        mlflow.set_tag("evaluated_model_version", version)
        mlflow.log_metrics(metrics)
        for art_path in artifacts.values():
            mlflow.log_artifact(art_path)

    maybe_promote_to_production(version, metrics["test_f1"], args.threshold)
    return 0


if __name__ == "__main__":
    sys.exit(main())
