"""SHAP explanations for the RetailGenius churn pipeline (Part 3 - XAI).

This module loads a registered *tree-based* churn pipeline from the MLflow
Model Registry and explains its predictions with SHAP. Because the model is
a scikit-learn :class:`~sklearn.pipeline.Pipeline`
(``preprocessor`` + ``classifier``), ``TreeExplainer`` cannot be applied to
the pipeline directly. We therefore:

1. transform the raw features with the fitted ``preprocessor`` step,
2. build the ``TreeExplainer`` on the bare ``classifier`` step, and
3. plot the Shapley values against the *original* (un-scaled) feature
   values so the explanations stay readable for a business audience.

Running it::

    python -m churn.explain.shap_explain --model xgboost

All figures are written to ``outputs/xai/`` so they are committed alongside
the code (they are *not* under the git-ignored ``reports/`` tree).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend, safe for CI / servers

import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from churn import config  # noqa: E402

logger = logging.getLogger(__name__)

# Output directory for all XAI artifacts (kept out of git-ignored reports/).
XAI_OUTPUT_DIR: Path = config.PROJECT_ROOT / "outputs" / "xai"

# TreeExplainer only supports tree ensembles, so logreg is excluded.
TREE_FAMILIES: tuple[str, ...] = ("xgboost", "random_forest")


# Helpers


def _save(filename: str) -> None:
    """Save the current matplotlib figure to the XAI output dir and close."""
    XAI_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = XAI_OUTPUT_DIR / filename
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    logger.info("Saved %s", path)


def _positive_class(shap_values: shap.Explanation) -> shap.Explanation:
    """Return the churn (positive) class slice of a SHAP Explanation.

    RandomForest's ``TreeExplainer`` yields a 3-D array shaped
    ``(n_samples, n_features, n_classes)``; we keep class index 1 (churn).
    XGBoost yields a 2-D array already, which is returned unchanged.
    """
    if shap_values.values.ndim == 3:
        return shap_values[..., 1]
    return shap_values


def _with_display(
    explanation: shap.Explanation, display: pd.DataFrame
) -> shap.Explanation:
    """Re-wrap an Explanation so its plotted feature values are readable.

    The Shapley values are computed on the *transformed* matrix, but the
    feature values shown next to them should be the *original* ones (real
    tenure, cashback, ... instead of standardised numbers). This keeps the
    SHAP values intact and only swaps the ``data`` used for display.
    """
    return shap.Explanation(
        values=explanation.values,
        base_values=explanation.base_values,
        data=display.to_numpy(),
        feature_names=list(display.columns),
    )


# Model and data loading


def load_tree_pipeline(
    preferred: str = "xgboost",
) -> tuple[Pipeline, str, str]:
    """Load a registered tree-based churn pipeline from MLflow.

    Parameters
    ----------
    preferred : str
        Preferred model family (``"xgboost"`` or ``"random_forest"``). If no
        registered version carries that family tag, the other tree family is
        used as a fallback.

    Returns
    -------
    tuple
        ``(pipeline, family, version)`` — the fitted sklearn pipeline, the
        model-family tag, and the registry version string.
    """
    client = MlflowClient()
    versions = client.search_model_versions(
        f"name='{config.REGISTERED_MODEL_NAME}'"
    )
    if not versions:
        raise RuntimeError(
            f"No versions registered for "
            f"'{config.REGISTERED_MODEL_NAME}'. Run "
            "`python -m churn.models.train` first."
        )

    # Map each registered version to the model_family tag of its run.
    candidates: dict[str, str] = {}
    for v in versions:
        run = client.get_run(v.run_id)
        family = run.data.tags.get("model_family", "")
        if family in TREE_FAMILIES:
            # Keep the highest version number per family.
            prev = candidates.get(family)
            if prev is None or int(v.version) > int(prev):
                candidates[family] = v.version

    if not candidates:
        raise RuntimeError(
            "No tree-based (xgboost / random_forest) version is registered; "
            "SHAP TreeExplainer needs a tree model."
        )

    order = [preferred] + [f for f in TREE_FAMILIES if f != preferred]
    family = next(f for f in order if f in candidates)
    version = candidates[family]

    uri = f"models:/{config.REGISTERED_MODEL_NAME}/{version}"
    logger.info("Loading %s pipeline from %s", family, uri)
    pipeline = mlflow.sklearn.load_model(uri)
    return pipeline, family, version


def load_test_features(sample_size: int) -> pd.DataFrame:
    """Load the test-split features, optionally subsampled.

    ``sample_size <= 0`` (the default) explains every row in the test set.
    """
    if not config.TEST_FILE.exists():
        raise FileNotFoundError(
            f"{config.TEST_FILE} not found. Run "
            "`python -m churn.data.make_dataset` first."
        )
    test_df = pd.read_parquet(config.TEST_FILE)
    x = test_df[config.ALL_FEATURES]
    if 0 < sample_size < len(x):
        x = x.sample(sample_size, random_state=config.RANDOM_STATE)
    return x.reset_index(drop=True)


def transform_features(
    pipeline: Pipeline, x: pd.DataFrame
) -> tuple[object, pd.DataFrame]:
    """Split the pipeline and return the classifier + transformed frame.

    Returns
    -------
    tuple
        ``(classifier, x_transformed)`` where ``x_transformed`` is a
        DataFrame whose columns are the post-preprocessing feature names, so
        SHAP plots are labelled with the one-hot / scaled feature names.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    matrix = preprocessor.transform(x)
    feature_names = list(preprocessor.get_feature_names_out())
    x_transformed = pd.DataFrame(matrix, columns=feature_names)
    logger.info(
        "Transformed %d rows into %d features",
        x_transformed.shape[0],
        x_transformed.shape[1],
    )
    return classifier, x_transformed


def build_display_frame(
    x_raw: pd.DataFrame, x_transformed: pd.DataFrame
) -> pd.DataFrame:
    """Build readable display values aligned to the transformed columns.

    Numeric columns are replaced by their *original* (un-scaled) values so a
    waterfall reads ``Tenure = 4`` instead of ``Tenure = -0.73``. One-hot
    columns keep their 0/1 encoding, which is already interpretable
    (``MaritalStatus_Single = 1``). Missing numeric values are shown as the
    column median, matching what the model's imputer used.
    """
    display = x_transformed.copy()
    numeric_cols = config.NUMERIC_WITH_NAN + config.NUMERIC_NO_NAN
    for col in numeric_cols:
        if col in display.columns and col in x_raw.columns:
            values = x_raw[col].to_numpy(dtype="float64")
            median = np.nanmedian(values)
            display[col] = np.where(np.isnan(values), median, values)
    return display


def build_explainer(
    classifier: object, x_transformed: pd.DataFrame
) -> tuple[shap.TreeExplainer, shap.Explanation]:
    """Build a SHAP ``TreeExplainer`` and compute Shapley values."""
    logger.info("Building TreeExplainer and computing SHAP values ...")
    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer(x_transformed)
    logger.info("SHAP values shape: %s", shap_values.values.shape)
    return explainer, shap_values


# Visualisations


def plot_single_point_bar(sv_churn: shap.Explanation, index: int) -> None:
    """Bar chart explaining the churn prediction of one customer."""
    logger.info("Single-point bar (index=%d)", index)
    shap.plots.bar(sv_churn[index], show=False)
    plt.title(f"SHAP contributions - customer #{index} (churn)")
    _save("01_single_point_bar.png")


def plot_global_bar(sv_churn: shap.Explanation) -> None:
    """Global mean-|SHAP| bar chart across all explained customers."""
    logger.info("Global mean-|SHAP| bar")
    shap.plots.bar(sv_churn, show=False)
    plt.title("Mean |SHAP| over all customers (churn)")
    _save("02_global_bar.png")


def plot_summary_per_class(
    shap_values: shap.Explanation, x_display: pd.DataFrame
) -> None:
    """Summary (beeswarm) plot for each class over all customers.

    Handles both the 3-D RandomForest output and the 2-D XGBoost output. For
    XGBoost (a single margin output) the no-churn view is the additive
    inverse of the churn SHAP values, which is labelled as such.
    """
    vals = shap_values.values
    if vals.ndim == 3:
        class0, class1 = vals[:, :, 0], vals[:, :, 1]
        c0_title = "SHAP summary - no-churn class (0)"
    else:
        class0, class1 = -vals, vals
        c0_title = "SHAP summary - no-churn class (negated margin)"

    logger.info("Summary plot - no-churn class")
    shap.summary_plot(class0, x_display, show=False)
    plt.title(c0_title)
    _save("03a_summary_class0.png")

    logger.info("Summary plot - churn class")
    shap.summary_plot(class1, x_display, show=False)
    plt.title("SHAP summary - churn class (1)")
    _save("03b_summary_class1.png")


def plot_waterfall(sv_churn: shap.Explanation, index: int) -> None:
    """Waterfall plot for a single customer."""
    logger.info("Waterfall (index=%d)", index)
    shap.plots.waterfall(sv_churn[index], show=False)
    plt.title(f"Why customer #{index} is scored for churn")
    _save("04_waterfall.png")


def plot_force(
    explainer: shap.TreeExplainer,
    sv_churn: shap.Explanation,
    x_display: pd.DataFrame,
    index: int,
) -> None:
    """Interactive force plot for a single customer, saved as HTML."""
    logger.info("Force plot (index=%d)", index)
    ev = explainer.expected_value
    if isinstance(ev, (list, np.ndarray)):
        ev = ev[1]  # churn class base value for RandomForest
    force = shap.force_plot(
        ev,
        sv_churn.values[index],
        x_display.iloc[index],
        matplotlib=False,
    )
    XAI_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    html_path = XAI_OUTPUT_DIR / "05_force_plot.html"
    shap.save_html(str(html_path), force)
    logger.info("Saved %s", html_path)


def plot_mean_shap(
    sv_churn: shap.Explanation, feature_names: list[str]
) -> None:
    """Horizontal bar chart of mean-|SHAP| per feature (churn class)."""
    logger.info("Mean-|SHAP| importance bar")
    mean_shap = np.abs(sv_churn.values).mean(axis=0)
    order = np.argsort(mean_shap)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(
        np.array(feature_names)[order],
        mean_shap[order],
        color="steelblue",
    )
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Mean SHAP feature importance (churn class)")
    fig.tight_layout()
    _save("06_mean_shap.png")


def plot_beeswarm(sv_churn: shap.Explanation) -> None:
    """Beeswarm plot of SHAP value distributions across all customers."""
    logger.info("Beeswarm")
    shap.plots.beeswarm(sv_churn, show=False)
    plt.title("SHAP value distribution per feature (churn)")
    _save("07_beeswarm.png")


def plot_dependence(
    sv_churn: shap.Explanation, x_display: pd.DataFrame
) -> None:
    """Dependence plots for the two most important features."""
    logger.info("Dependence plots (top-2 features)")
    mean_shap = np.abs(sv_churn.values).mean(axis=0)
    top = np.argsort(mean_shap)[::-1]
    first = x_display.columns[top[0]]
    second = x_display.columns[top[1]]

    shap.dependence_plot(
        first,
        sv_churn.values,
        x_display,
        interaction_index=second,
        show=False,
    )
    _save("08_dependence_top1.png")

    shap.dependence_plot(
        second,
        sv_churn.values,
        x_display,
        interaction_index=first,
        show=False,
    )
    _save("09_dependence_top2.png")


# CLI


def pick_index(pipeline: Pipeline, x_raw: pd.DataFrame, index: int) -> int:
    """Choose the row for single-point plots.

    ``index < 0`` (the default) auto-selects the highest-risk customer in
    the sample, which makes the waterfall / force plots illustrative instead
    of explaining an obvious non-churner.
    """
    if index >= 0:
        return index
    proba = pipeline.predict_proba(x_raw)[:, 1]
    chosen = int(np.argmax(proba))
    logger.info(
        "Auto-selected customer #%d (churn probability %.3f)",
        chosen,
        proba[chosen],
    )
    return chosen


def run(model: str, sample_size: int, index: int) -> None:
    """Generate the full SHAP plot suite for the churn model."""
    pipeline, family, version = load_tree_pipeline(preferred=model)
    logger.info("Explaining %s (registry version %s)", family, version)

    x_raw = load_test_features(sample_size)
    index = pick_index(pipeline, x_raw, index)

    classifier, x_transformed = transform_features(pipeline, x_raw)
    explainer, shap_values = build_explainer(classifier, x_transformed)
    x_display = build_display_frame(x_raw, x_transformed)

    sv_churn = _positive_class(shap_values)
    sv_disp = _with_display(sv_churn, x_display)

    plot_single_point_bar(sv_disp, index)
    plot_global_bar(sv_disp)
    plot_summary_per_class(shap_values, x_display)
    plot_waterfall(sv_disp, index)
    plot_force(explainer, sv_churn, x_display, index)
    plot_mean_shap(sv_churn, list(x_display.columns))
    plot_beeswarm(sv_disp)
    plot_dependence(sv_churn, x_display)

    logger.info("All XAI outputs written to %s", XAI_OUTPUT_DIR)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=list(TREE_FAMILIES),
        default="xgboost",
        help="Preferred tree family to explain (default: xgboost).",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=0,
        help="Rows from the test set to explain (0 = all, the default).",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=-1,
        help="Row for single-point plots (-1 = auto-pick highest risk).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)

    run(args.model, args.sample_size, args.index)
    return 0


if __name__ == "__main__":
    sys.exit(main())
