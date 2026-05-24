"""Rollback / inspect the model registry.

List candidates::

    python -m churn.models.rollback --list

Promote a specific version back to Production::

    python -m churn.models.rollback --version 3
"""

from __future__ import annotations

import argparse
import logging
import sys

import mlflow
from mlflow.tracking import MlflowClient

from churn import config

logger = logging.getLogger(__name__)


def list_versions() -> None:
    """Print every registered model version with its stage and metrics."""
    client = MlflowClient()
    versions = client.search_model_versions(
        f"name='{config.REGISTERED_MODEL_NAME}'"
    )

    if not versions:
        print(
            f"No versions registered yet for "
            f"'{config.REGISTERED_MODEL_NAME}'."
        )
        return

    # Sort newest first
    versions = sorted(versions, key=lambda v: int(v.version), reverse=True)

    print(f"{'Version':<8}{'Stage':<14}{'Run ID':<34}{'Val F1':<10}")
    print("-" * 66)
    for v in versions:
        run = client.get_run(v.run_id)
        val_f1 = run.data.metrics.get("f1", float("nan"))
        print(
            f"{v.version:<8}{v.current_stage:<14}"
            f"{v.run_id:<34}{val_f1:<10.3f}"
        )


def promote(version: str) -> None:
    """Transition given version to Production, archiving the current one."""
    client = MlflowClient()

    target = client.get_model_version(
        name=config.REGISTERED_MODEL_NAME,
        version=version,
    )
    if target.current_stage == "Production":
        logger.info(
            "Version %s is already in Production. Nothing to do.", version
        )
        return

    # Archive whatever sits in Production today (if anything)
    for v in client.search_model_versions(
        f"name='{config.REGISTERED_MODEL_NAME}'"
    ):
        if v.current_stage == "Production":
            client.transition_model_version_stage(
                name=config.REGISTERED_MODEL_NAME,
                version=v.version,
                stage="Archived",
            )
            logger.info("Archived previous Production version %s", v.version)

    client.transition_model_version_stage(
        name=config.REGISTERED_MODEL_NAME,
        version=version,
        stage="Production",
    )
    logger.info("Promoted version %s to Production.", version)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--list", action="store_true", help="List all versions."
    )
    group.add_argument("--version", help="Promote this version to Production.")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)

    if args.list:
        list_versions()
    else:
        promote(args.version)
    return 0


if __name__ == "__main__":
    sys.exit(main())
