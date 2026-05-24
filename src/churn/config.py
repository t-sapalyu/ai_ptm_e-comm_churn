from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
INTERIM_DIR: Path = DATA_DIR / "interim"
PROCESSED_DIR: Path = DATA_DIR / "processed"

REPORTS_DIR: Path = PROJECT_ROOT / "reports"
MLRUNS_DIR: Path = PROJECT_ROOT / "mlruns"

RAW_DATA_FILE: Path = RAW_DIR / "E_Commerce_Dataset.xlsx"
RAW_SHEET_NAME: str = "E Comm"

TRAIN_FILE: Path = PROCESSED_DIR / "train.parquet"
VAL_FILE: Path = PROCESSED_DIR / "val.parquet"
TEST_FILE: Path = PROCESSED_DIR / "test.parquet"
REFERENCE_FILE: Path = PROCESSED_DIR / "reference.parquet"


# Target and columns
TARGET: str = "Churn"

# Customer identifier. KEPT in split files and prediction outputs so we can
# join back to the CRM, but NOT in ALL_FEATURES.
ID_COL: str = "CustomerID"

# Numeric columns that contain NaN in the raw data
NUMERIC_WITH_NAN: list[str] = [
    "Tenure",
    "WarehouseToHome",
    "HourSpendOnApp",
    "OrderAmountHikeFromlastYear",
    "CouponUsed",
    "OrderCount",
    "DaySinceLastOrder",
]

# Numeric columns that are complete in the raw data
NUMERIC_NO_NAN: list[str] = [
    "CityTier",
    "NumberOfDeviceRegistered",
    "SatisfactionScore",
    "NumberOfAddress",
    "Complain",
    "CashbackAmount",
]

CATEGORICAL: list[str] = [
    "PreferredLoginDevice",
    "PreferredPaymentMode",
    "Gender",
    "PreferedOrderCat",
    "MaritalStatus",
]

# Columns the model is allowed to see. CustomerID is intentionally absent.
ALL_FEATURES: list[str] = NUMERIC_WITH_NAN + NUMERIC_NO_NAN + CATEGORICAL

# Splits
RANDOM_STATE: int = 42
TRAIN_SIZE: float = 0.70
VAL_SIZE: float = 0.15  # the remaining 0.15 goes to test


# MLflow
MLFLOW_TRACKING_URI: str = os.environ.get("MLFLOW_TRACKING_URI", MLRUNS_DIR.as_uri())
MLFLOW_EXPERIMENT_NAME: str = "retailgenius-churn"
REGISTERED_MODEL_NAME: str = "churn_classifier"

# Models to train. Keys are the names used in MLflow run tags and CLI flags.
MODELS_TO_TRAIN: list[str] = ["logreg", "random_forest", "xgboost"]

# Test F1 threshold for promoting Staging -> Production
PROMOTION_F1_THRESHOLD: float = 0.70


# Monitoring
# PSI thresholds per the industry convention
PSI_NO_SHIFT: float = 0.10
PSI_MODERATE_SHIFT: float = 0.25
