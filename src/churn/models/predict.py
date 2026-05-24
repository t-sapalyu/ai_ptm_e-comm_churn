"""Batch inference using the Production model.
Run with::
python -m churn.models.predict --input customers.csv --output preds.csv

"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient

from churn import config

logger = logging.getLogger(__name__)


def load_production_model() -> object:
    """Load the model currently in the Production stage."""
    client = MlflowClient()
    versions = client.get_latest_versions(config.REGISTERED_MODEL_NAME, stages=["Production"])
    if not versions:
        raise RuntimeError(
            f"No '{config.REGISTERED_MODEL_NAME}' model in Production. "
            "Run `python -m churn.models.evaluate` first."
        )
    uri = f"models:/{config.REGISTERED_MODEL_NAME}/Production"
    logger.info("Loading %s", uri)
    return mlflow.sklearn.load_model(uri)


def predict(input_path: Path, output_path: Path) -> None:
    """Score the rows in ``input_path`` and write the result to ``output_path``.

    The input CSV must include the customer ID column plus every feature
    column listed in :data:`churn.config.ALL_FEATURES`.
    """
    df = pd.read_csv(input_path)

    missing_features = [c for c in config.ALL_FEATURES if c not in df.columns]
    if missing_features:
        raise ValueError(
            f"Input CSV is missing required feature columns: {missing_features}. "
            f"Expected at least: {config.ALL_FEATURES}"
        )
    if config.ID_COL not in df.columns:
        raise ValueError(
            f"Input CSV is missing the customer identifier '{config.ID_COL}'. "
            "Predictions without an ID cannot be joined back to the CRM."
        )

    model = load_production_model()

    # The model sees only the feature columns
    x = df[config.ALL_FEATURES]

    # The output keeps everything — ID, features, predictions
    output = df.copy()
    output["churn_prediction"] = model.predict(x)
    output["churn_probability"] = model.predict_proba(x)[:, 1]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    logger.info("Wrote %d predictions to %s", len(output), output_path)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Input CSV.")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)

    predict(args.input, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
