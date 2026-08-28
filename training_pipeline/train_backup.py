"""
Production-grade time-series training pipeline
for Pearls AQI Predictor.

Trains separate models for:
    - 24-hour PM2.5 forecast
    - 48-hour PM2.5 forecast
    - 72-hour PM2.5 forecast

Design principles:
    - Uses production model_features.csv
    - Chronological train/test split
    - No random shuffling
    - No future leakage
    - Separate model per forecast horizon
    - Reproducible training
    - Model metadata and metrics saved
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline


# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_features.csv"
)

MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports" / "model_evaluation"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# CONFIGURATION
# ============================================================================

TEST_SIZE = 0.20
RANDOM_STATE = 42

FORECAST_TARGETS = {
    "24h": "target_pm2_5_24h",
    "48h": "target_pm2_5_48h",
    "72h": "target_pm2_5_72h",
}

EXCLUDED_COLUMNS = {
    "timestamp",
    "city",
    "target_pm2_5_24h",
    "target_pm2_5_48h",
    "target_pm2_5_72h",
}


# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(name)s | "
        "%(message)s"
    ),
)

logger = logging.getLogger(__name__)


# ============================================================================
# DATA LOADING
# ============================================================================

def load_dataset() -> pd.DataFrame:
    """Load and validate the production feature dataset."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Feature dataset not found: {DATA_PATH}"
        )

    logger.info(
        "Loading production feature dataset: %s",
        DATA_PATH,
    )

    df = pd.read_csv(DATA_PATH)

    if df.empty:
        raise ValueError("Feature dataset is empty.")

    if "timestamp" not in df.columns:
        raise ValueError(
            "Feature dataset must contain 'timestamp'."
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
        errors="coerce",
    )

    df = (
        df.dropna(subset=["timestamp"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    logger.info(
        "Dataset shape: %s",
        df.shape,
    )

    logger.info(
        "Dataset period: %s -> %s",
        df["timestamp"].min(),
        df["timestamp"].max(),
    )

    return df


# ============================================================================
# FEATURE SELECTION
# ============================================================================

def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return numeric model features excluding targets and metadata."""

    feature_columns = [
        column
        for column in df.columns
        if column not in EXCLUDED_COLUMNS
        and pd.api.types.is_numeric_dtype(df[column])
    ]

    if not feature_columns:
        raise ValueError(
            "No numeric feature columns available."
        )

    logger.info(
        "Feature count: %d",
        len(feature_columns),
    )

    return feature_columns


# ============================================================================
# MODEL BUILDERS
# ============================================================================

def build_random_forest() -> Pipeline:
    """Build production Random Forest pipeline."""

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=300,
                    max_depth=None,
                    min_samples_leaf=2,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def build_ridge() -> Pipeline:
    """Build Ridge regression pipeline."""

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "model",
                Ridge(alpha=1.0),
            ),
        ]
    )


# ============================================================================
# METRICS
# ============================================================================

def evaluate_model(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    """Calculate regression metrics."""

    predictions = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        predictions,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions,
        )
    )

    r2 = r2_score(
        y_test,
        predictions,
    )

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
    }


# ============================================================================
# TRAIN ONE HORIZON
# ============================================================================

def train_horizon(
    df: pd.DataFrame,
    feature_columns: list[str],
    horizon_name: str,
    target_column: str,
) -> dict:
    """Train and evaluate models for one forecast horizon."""

    logger.info(
        "=" * 72
    )

    logger.info(
        "Training forecast horizon: %s",
        horizon_name,
    )

    # Only rows with an actual future target are usable.
    dataset = df.dropna(
        subset=[target_column]
    ).copy()

    if len(dataset) < 100:
        raise ValueError(
            f"Insufficient samples for {horizon_name}: "
            f"{len(dataset)}"
        )

    X = dataset[feature_columns]
    y = dataset[target_column]

    # ------------------------------------------------------------------------
    # Chronological split
    # ------------------------------------------------------------------------

    split_index = int(
        len(dataset) * (1 - TEST_SIZE)
    )

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    logger.info(
        "%s samples: %d",
        horizon_name,
        len(dataset),
    )

    logger.info(
        "Training samples: %d",
        len(X_train),
    )

    logger.info(
        "Testing samples: %d",
        len(X_test),
    )

    logger.info(
        "Train period: %s -> %s",
        dataset["timestamp"].iloc[0],
        dataset["timestamp"].iloc[split_index - 1],
    )

    logger.info(
        "Test period: %s -> %s",
        dataset["timestamp"].iloc[split_index],
        dataset["timestamp"].iloc[-1],
    )

    # ------------------------------------------------------------------------
    # Random Forest
    # ------------------------------------------------------------------------

    rf_model = build_random_forest()

    rf_model.fit(
        X_train,
        y_train,
    )

    rf_metrics = evaluate_model(
        rf_model,
        X_test,
        y_test,
    )

    logger.info(
        "Random Forest | MAE=%.4f | RMSE=%.4f | R2=%.4f",
        rf_metrics["mae"],
        rf_metrics["rmse"],
        rf_metrics["r2"],
    )

    # ------------------------------------------------------------------------
    # Ridge
    # ------------------------------------------------------------------------

    ridge_model = build_ridge()

    ridge_model.fit(
        X_train,
        y_train,
    )

    ridge_metrics = evaluate_model(
        ridge_model,
        X_test,
        y_test,
    )

    logger.info(
        "Ridge | MAE=%.4f | RMSE=%.4f | R2=%.4f",
        ridge_metrics["mae"],
        ridge_metrics["rmse"],
        ridge_metrics["r2"],
    )

    # ------------------------------------------------------------------------
    # Select best model by RMSE
    # ------------------------------------------------------------------------

    if rf_metrics["rmse"] <= ridge_metrics["rmse"]:
        best_model_name = "random_forest"
        best_model = rf_model
        best_metrics = rf_metrics
    else:
        best_model_name = "ridge"
        best_model = ridge_model
        best_metrics = ridge_metrics

    # ------------------------------------------------------------------------
    # Save models
    # ------------------------------------------------------------------------

    rf_path = (
        MODEL_DIR
        / f"random_forest_{horizon_name}.joblib"
    )

    ridge_path = (
        MODEL_DIR
        / f"ridge_{horizon_name}.joblib"
    )

    best_path = (
        MODEL_DIR
        / f"best_model_{horizon_name}.joblib"
    )

    joblib.dump(
        rf_model,
        rf_path,
    )

    joblib.dump(
        ridge_model,
        ridge_path,
    )

    joblib.dump(
        best_model,
        best_path,
    )

    return {
        "horizon": horizon_name,
        "target": target_column,
        "samples": len(dataset),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "random_forest": rf_metrics,
        "ridge": ridge_metrics,
        "best_model": best_model_name,
        "best_metrics": best_metrics,
        "random_forest_path": str(rf_path),
        "ridge_path": str(ridge_path),
        "best_model_path": str(best_path),
    }


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    """Execute complete model training pipeline."""

    logger.info("=" * 72)
    logger.info("PEARLS AQI PREDICTOR")
    logger.info("PRODUCTION MODEL TRAINING")
    logger.info("=" * 72)

    df = load_dataset()

    feature_columns = get_feature_columns(df)

    results = {}

    for horizon_name, target_column in FORECAST_TARGETS.items():

        results[horizon_name] = train_horizon(
            df=df,
            feature_columns=feature_columns,
            horizon_name=horizon_name,
            target_column=target_column,
        )

    # ------------------------------------------------------------------------
    # Save evaluation report
    # ------------------------------------------------------------------------

    report = {
        "dataset": str(DATA_PATH),
        "dataset_rows": int(len(df)),
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "forecast_targets": FORECAST_TARGETS,
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "results": results,
    }

    report_path = (
        REPORT_DIR
        / "training_report.json"
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=4,
        ),
        encoding="utf-8",
    )

    logger.info(
        "Training report saved: %s",
        report_path,
    )

    logger.info("=" * 72)
    logger.info(
        "MODEL TRAINING COMPLETED SUCCESSFULLY."
    )
    logger.info("=" * 72)


if __name__ == "__main__":
    main()