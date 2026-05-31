"""Build the train / validation / test splits from the raw Excel file."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from churn import config
from churn.data.splitting import stratified_random_split

logger = logging.getLogger(__name__)


def load_raw(
    path: Path = config.RAW_DATA_FILE,
    sheet: str = config.RAW_SHEET_NAME,
) -> pd.DataFrame:
    """Read the raw Excel file into a DataFrame."""
    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {path}. "
            "Please place E_Commerce_Dataset.xlsx in data/raw/ before running."
        )
    logger.info("Reading raw data from %s (sheet=%s)", path, sheet)
    return pd.read_excel(path, sheet_name=sheet)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply minimal cleaning that should happen before splitting."""
    required = [config.ID_COL, config.TARGET] + config.ALL_FEATURES
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Raw dataset is missing required columns: {missing}")

    # drop duplicate rows to avoid train/test leakage
    df = df.drop_duplicates(subset=config.ALL_FEATURES).reset_index(drop=True)
    return df


def compute_fingerprint(df: pd.DataFrame) -> str:
    """Compute a stable SHA-256 fingerprint of a DataFrame.

    The hash is taken over the CSV serialisation of the frame so it is stable
    across pandas versions. Used as an MLflow tag on every training run, which
    lets us trace any model back to the exact data that produced it."""
    payload = df.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict[str, str]:
    """Write the three splits as parquet files + a reference snapshot.

    Returns
    -------
    dict
        Mapping of split name to its SHA-256 fingerprint.
    """
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    train_df.to_parquet(config.TRAIN_FILE, index=False)
    val_df.to_parquet(config.VAL_FILE, index=False)
    test_df.to_parquet(config.TEST_FILE, index=False)

    # Drift monitor uses the training distribution as the reference baseline
    train_df.to_parquet(config.REFERENCE_FILE, index=False)

    fingerprints = {
        "train": compute_fingerprint(train_df),
        "val": compute_fingerprint(val_df),
        "test": compute_fingerprint(test_df),
    }

    # Persist fingerprints next to the data so train.py can pick them up
    fp_path = config.PROCESSED_DIR / "fingerprints.json"
    with fp_path.open("w") as fh:
        json.dump(fingerprints, fh, indent=2)

    logger.info(
        "Wrote splits: train=%d rows, val=%d rows, test=%d rows. "
        "Fingerprints in %s",
        len(train_df),
        len(val_df),
        len(test_df),
        fp_path,
    )
    return fingerprints


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-file",
        type=Path,
        default=config.RAW_DATA_FILE,
        help="Path to the raw Excel file.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    df = load_raw(args.raw_file)
    df = clean(df)
    train_df, val_df, test_df = stratified_random_split(df)
    save_splits(train_df, val_df, test_df)

    # Sanity check: positive-class ratio should be roughly preserved
    for name, frame in [
        ("train", train_df),
        ("val", val_df),
        ("test", test_df),
    ]:
        ratio = frame[config.TARGET].mean()
        logger.info("Churn ratio in %s: %.3f", name, ratio)

    return 0


if __name__ == "__main__":
    sys.exit(main())
