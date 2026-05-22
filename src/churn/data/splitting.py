from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from churn import config


def stratified_random_split(
    df: pd.DataFrame,
    target: str = config.TARGET,
    train_size: float = config.TRAIN_SIZE,
    val_size: float = config.VAL_SIZE,
    random_state: int = config.RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified 70 / 15 / 15 split on the churn target.

    Used because the provided dataset has no calendar timestamp;
    """
    # First split: train vs. (val + test)
    train_df, holdout_df = train_test_split(
        df,
        train_size=train_size,
        random_state=random_state,
        stratify=df[target],
    )

    # Second split: val vs. test, keeping the val_size:test_size ratio
    val_fraction_of_holdout = val_size / (1.0 - train_size)
    val_df, test_df = train_test_split(
        holdout_df,
        train_size=val_fraction_of_holdout,
        random_state=random_state,
        stratify=holdout_df[target],
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )
