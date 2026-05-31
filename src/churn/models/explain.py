"""Generate Explainable AI (XAI) outputs using SHAP.

This script loads the production model, unpacks the scikit-learn
pipeline to access the raw tree classifier, and generates both
global and local SHAP explainability plots[cite: 14, 15, 17].
"""

import logging
import sys

import matplotlib

# Headless backend to prevent display crashes in Docker/CI
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402

from churn import config  # noqa: E402
from churn.models.evaluate import load_test_set  # noqa: E402
from churn.models.predict import load_production_model  # noqa: E402

logger = logging.getLogger(__name__)


def prepare_shap_data() -> tuple[object, pd.DataFrame]:
    """Load model and transform test data for SHAP processing.

    SHAP's TreeExplainer requires the raw algorithm, not a pipeline.
    This function unpacks the model and manually transforms the test
    set to match the expected numerical input matrix.

    Returns
    -------
    tuple[object, pd.DataFrame]
        The raw classifier and the transformed test data as a DataFrame
        with proper feature names.
    """
    logger.info("Loading test data and production model...")
    x_test, _ = load_test_set()
    model = load_production_model()

    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    x_test_transformed = preprocessor.transform(x_test)
    feature_names = preprocessor.get_feature_names_out()

    x_test_df = pd.DataFrame(
        x_test_transformed, columns=feature_names
    )
    return classifier, x_test_df


def generate_global_plots(
    explainer: shap.TreeExplainer,
    shap_values: shap.Explanation,
    x_test_df: pd.DataFrame,
) -> None:
    """Generate and save global explainability plots.

    Creates a summary plot [cite: 20], beeswarm plot[cite: 28],
    mean SHAP bar plot [cite: 27], and a dependence plot[cite: 29].

    Parameters
    ----------
    explainer : shap.TreeExplainer
        The fitted SHAP explainer object.
    shap_values : shap.Explanation
        Computed SHAP values for the test dataset.
    x_test_df : pd.DataFrame
        The transformed test dataset.
    """
    out_dir = config.REPORTS_DIR / "shap_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Summary plot
    plt.figure()
    shap.summary_plot(shap_values, x_test_df, show=False)
    plt.savefig(
        out_dir / "summary_plot.png", bbox_inches="tight"
    )
    plt.close()

    # 2. Beeswarm plot
    plt.figure()
    shap.plots.beeswarm(shap_values, show=False)
    plt.savefig(
        out_dir / "beeswarm_plot.png", bbox_inches="tight"
    )
    plt.close()

    # 3. Mean SHAP plot
    plt.figure()
    shap.plots.bar(shap_values, show=False)
    plt.savefig(
        out_dir / "mean_shap_plot.png", bbox_inches="tight"
    )
    plt.close()

    # 4. Dependence plot (uses the first feature dynamically)
    plt.figure()
    first_feature = x_test_df.columns[0]
    raw_shap_values = explainer.shap_values(x_test_df)
    shap.dependence_plot(
        first_feature, raw_shap_values, x_test_df, show=False
    )
    plt.savefig(
        out_dir / "dependence_plot.png", bbox_inches="tight"
    )
    plt.close()
    logger.info("Global SHAP plots generated.")


def generate_local_plots(
    explainer: shap.TreeExplainer,
    shap_values: shap.Explanation,
    x_test_df: pd.DataFrame,
    point_idx: int = 0,
) -> None:
    """Generate and save local explainability plots.

    Creates a waterfall plot [cite: 23] and a force plot [cite: 25]
    for a specific data point to explain an individual prediction.

    Parameters
    ----------
    explainer : shap.TreeExplainer
        The fitted SHAP explainer object.
    shap_values : shap.Explanation
        Computed SHAP values for the test dataset.
    x_test_df : pd.DataFrame
        The transformed test dataset.
    point_idx : int, optional
        The index of the data point to explain, by default 0.
    """
    out_dir = config.REPORTS_DIR / "shap_plots"

    # 1. Waterfall plot
    plt.figure()
    shap.plots.waterfall(shap_values[point_idx], show=False)
    plt.savefig(
        out_dir / f"waterfall_plot_{point_idx}.png",
        bbox_inches="tight",
    )
    plt.close()

    # 2. Force plot
    raw_shap_values = explainer.shap_values(x_test_df)
    shap.force_plot(
        explainer.expected_value,
        raw_shap_values[point_idx],
        x_test_df.iloc[point_idx],
        matplotlib=True,
        show=False,
    )
    plt.savefig(
        out_dir / f"force_plot_{point_idx}.png",
        bbox_inches="tight",
    )
    plt.close()
    logger.info("Local SHAP plots generated for index %d", point_idx)


def main() -> int:
    """CLI entry point for generating XAI reports."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    try:
        classifier, x_test_df = prepare_shap_data()

        logger.info("Initializing TreeExplainer...")
        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer(x_test_df)

        generate_global_plots(explainer, shap_values, x_test_df)
        generate_local_plots(explainer, shap_values, x_test_df)

        logger.info("All XAI artifacts saved to reports/shap_plots")
        return 0
    except Exception as e:
        logger.error("Failed to generate XAI plots: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
