from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn import config


def build_preprocessor() -> ColumnTransformer:
    """Return the ColumnTransformer used as the first stage of every model pipeline."""
    numeric_with_nan_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    numeric_no_nan_pipeline = Pipeline(steps=[("scaler", StandardScaler())])

    # sparse_output=False keeps the dense output compatible with all downstream
    # estimators (XGBoost works fine with sparse, but logreg + RF are happier dense)
    categorical_pipeline = Pipeline(
        steps=[
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num_with_nan", numeric_with_nan_pipeline, config.NUMERIC_WITH_NAN),
            ("num_no_nan", numeric_no_nan_pipeline, config.NUMERIC_NO_NAN),
            ("cat", categorical_pipeline, config.CATEGORICAL),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor
