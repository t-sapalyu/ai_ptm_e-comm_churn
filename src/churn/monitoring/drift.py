"""Population Stability Index (PSI) and drift utilities.

PSI quantifies how much a distribution has shifted between a reference
(usually the training data) and a current sample. Industry thresholds:

* PSI < 0.10         — no significant shift
* 0.10 ≤ PSI < 0.25  — moderate shift, investigate
* PSI ≥ 0.25         — significant shift, act on it

See section 5.2 of ``IMPLEMENTATION_PLAN.md``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from churn import config


def compute_psi(
    reference: pd.Series,
    current: pd.Series,
    bins: int = 10,
    epsilon: float = 1e-6,
) -> float:
    """Compute the Population Stability Index between two 1-D distributions.

    Numeric columns are binned by quantile on the reference series.
    Categorical / object columns are bucketed by their distinct values.

    Parameters
    ----------
    reference : pandas.Series
        Reference (baseline) values, typically from the training set.
    current : pandas.Series
        Current values, typically the latest production sample.
    bins : int
        Number of quantile bins for numeric columns. Ignored for
        categoricals.
    epsilon : float
        Small constant added to bin proportions to avoid ``log(0)``.

    Returns
    -------
    float
        The PSI value. Larger means more drift; never negative.
    """
    ref = reference.dropna()
    cur = current.dropna()
    if ref.empty or cur.empty:
        return float("nan")

    is_numeric = pd.api.types.is_numeric_dtype(
        ref
    ) and pd.api.types.is_numeric_dtype(cur)

    if is_numeric:
        # Quantile edges from the reference distribution
        quantiles = np.linspace(0, 1, bins + 1)
        edges = np.unique(np.quantile(ref, quantiles))
        if len(edges) < 2:
            # All reference values are identical — fall back to a single bin
            return 0.0
        # Make sure the outer edges include any extreme values in `current`
        edges[0] = min(edges[0], cur.min())
        edges[-1] = max(edges[-1], cur.max())

        ref_counts, _ = np.histogram(ref, bins=edges)
        cur_counts, _ = np.histogram(cur, bins=edges)
    else:
        # Categorical: use the union of categories observed in either sample
        categories = sorted(set(ref.unique()) | set(cur.unique()))
        ref_counts = np.array([(ref == c).sum() for c in categories])
        cur_counts = np.array([(cur == c).sum() for c in categories])

    ref_props = ref_counts / max(ref_counts.sum(), 1)
    cur_props = cur_counts / max(cur_counts.sum(), 1)

    # Avoid log(0) without distorting non-zero bins
    ref_props = np.where(ref_props == 0, epsilon, ref_props)
    cur_props = np.where(cur_props == 0, epsilon, cur_props)

    psi = np.sum((cur_props - ref_props) * np.log(cur_props / ref_props))
    return float(psi)


def classify_psi(psi: float) -> str:
    """Map a PSI value to one of ``"none"``, ``"moderate"``."""
    if np.isnan(psi):
        return "unknown"
    if psi < config.PSI_NO_SHIFT:
        return "none"
    if psi < config.PSI_MODERATE_SHIFT:
        return "moderate"
    return "significant"


def feature_drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    features: list[str] | None = None,
    bins: int = 10,
) -> pd.DataFrame:
    """Compute PSI for every feature and return a tidy summary.

    Parameters
    ----------
    reference_df : pandas.DataFrame
        Baseline data, usually the training set.
    current_df : pandas.DataFrame
        Latest production sample.
    features : list of str, optional
        Subset of columns to evaluate. Defaults to
        :data:`config.ALL_FEATURES`.
    bins : int
        Number of bins for numeric features.

    Returns
    -------
    pandas.DataFrame
        One row per feature with columns ``feature``, ``psi``, ``severity``.
        Sorted by descending PSI.
    """
    cols = features if features is not None else config.ALL_FEATURES
    records = []
    for col in cols:
        if col not in reference_df.columns or col not in current_df.columns:
            continue
        psi = compute_psi(reference_df[col], current_df[col], bins=bins)
        records.append(
            {
                "feature": col,
                "psi": psi,
                "severity": classify_psi(psi),
            }
        )

    out = pd.DataFrame.from_records(records)
    if not out.empty:
        out = out.sort_values("psi", ascending=False).reset_index(drop=True)
    return out
