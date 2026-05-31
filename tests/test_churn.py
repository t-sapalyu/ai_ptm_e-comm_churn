"""Smoke tests for the churn package (no trained model or MLflow needed)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from churn import config
from churn.data.splitting import stratified_random_split
from churn.features.build_features import build_preprocessor
from churn.monitoring.drift import classify_psi, compute_psi


def _toy_frame(n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    data: dict[str, object] = {
        c: rng.normal(size=n) for c in config.NUMERIC_WITH_NAN
    }
    data.update({c: rng.integers(0, 5, size=n) for c in config.NUMERIC_NO_NAN})
    data.update(
        {c: rng.choice(["a", "b"], size=n) for c in config.CATEGORICAL}
    )
    data[config.ID_COL] = np.arange(n)
    data[config.TARGET] = rng.integers(0, 2, size=n)
    return pd.DataFrame(data)


def test_split_sizes_and_stratification() -> None:
    df = _toy_frame()
    train, val, test = stratified_random_split(df)
    assert len(train) + len(val) + len(test) == len(df)
    assert len(train) > len(val)
    overall = df[config.TARGET].mean()
    assert abs(train[config.TARGET].mean() - overall) < 0.05


def test_preprocessor_builds() -> None:
    names = [t[0] for t in build_preprocessor().transformers]
    assert "num_with_nan" in names
    assert "cat" in names


def test_psi_is_zero_for_identical_distribution() -> None:
    s = pd.Series(np.random.default_rng(1).normal(size=500))
    assert compute_psi(s, s.copy()) < 1e-6


def test_psi_detects_a_shift() -> None:
    rng = np.random.default_rng(2)
    ref = pd.Series(rng.normal(0, 1, 500))
    cur = pd.Series(rng.normal(3, 1, 500))
    assert compute_psi(ref, cur) > 0.25


def test_classify_psi_bands() -> None:
    assert classify_psi(0.05) == "none"
    assert classify_psi(0.15) == "moderate"
    assert classify_psi(0.30) == "significant"


def test_build_display_frame_restores_original_values() -> None:
    from churn.explain.shap_explain import build_display_frame

    raw = _toy_frame(50)[config.ALL_FEATURES]
    transformed = pd.DataFrame(
        np.zeros((50, len(config.ALL_FEATURES))),
        columns=config.ALL_FEATURES,
    )
    display = build_display_frame(raw, transformed)
    col = config.NUMERIC_NO_NAN[0]
    assert np.allclose(display[col].to_numpy(), raw[col].to_numpy())
