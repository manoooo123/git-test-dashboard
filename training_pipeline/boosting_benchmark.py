"""
Production benchmark for Pearls AQI Predictor.

Compares:
    1. Persistence baseline
    2. Random Forest
    3. HistGradientBoosting

Evaluation is chronological and leakage-safe.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


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

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "model_evaluation"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================================
# CONFIGURATION
# ============================================================================

TARGETS = {
    "24h": "target_pm2_5_24h",
    "48h": "target_pm2_5_48h",
    "72h": "target_pm2_5_72h",
}

PERSISTENCE_FEATURES = {
    "24h": "pm2_5_lag_24h",
    "48h": "pm2_5_lag_48h",
    "72h": "pm2_5_lag_72h",
}

TEST_SIZE = 0.20
RANDOM_STATE = 42


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
# DATA
# ============================================================================

def load_data() -> pd.DataFrame:
    """Load and validate production feature data."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    if df.empty:
        raise ValueError(
            "Production feature dataset is empty."
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

    return df


# ============================================================================
# METRICS
# ============================================================================

def calculate_metrics(
    y_true: pd.Series,
    predictions: np.ndarray,
) -> dict[str, float]:
    """Calculate regression metrics."""

    return {
        "mae": float(
            mean_absolute_error(
                y_true,
                predictions,
            )
        ),
        "rmse": float(
            np.sqrt(
                mean_squared_error(
                    y_true,
                    predictions,
                )
            )
        ),
        "r2": float(
            r2_score(
                y_true,
                predictions,
            )
        ),
    }


# ============================================================================
# MODEL
# ============================================================================

def build_random_forest() -> RandomForestRegressor:
    """Build production Random Forest."""

    return RandomForestRegressor(
        n_estimators=400,
        max_depth=18,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def build_hist_gradient_boosting() -> (
    HistGradientBoostingRegressor
):
    """Build HistGradientBoosting model."""

    return HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_iter=400,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=RANDOM_STATE,
    )


# ============================================================================
# HORIZON EVALUATION
# ============================================================================

def evaluate_horizon(
    df: pd.DataFrame,
    horizon: str,
    target_column: str,
) -> dict:
    """Evaluate all candidate models for one forecast horizon."""

    logger.info("=" * 72)
    logger.info(
        "Benchmarking forecast horizon: %s",
        horizon,
    )

    persistence_column = (
        PERSISTENCE_FEATURES[horizon]
    )

    required = [
        "timestamp",
        target_column,
        persistence_column,
    ]

    dataset = (
        df[required + [
            column
            for column in df.columns
            if column not in required
            and pd.api.types.is_numeric_dtype(
                df[column]
            )
        ]]
        .dropna(
            subset=[
                target_column,
            ]
        )
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    # ------------------------------------------------------------------------
    # Chronological split
    # ------------------------------------------------------------------------

    split_index = int(
        len(dataset) * (1 - TEST_SIZE)
    )

    train = dataset.iloc[:split_index]
    test = dataset.iloc[split_index:]

    y_train = train[target_column]
    y_test = test[target_column]

    excluded = {
        "timestamp",
        "city",
        "target_pm2_5_24h",
        "target_pm2_5_48h",
        "target_pm2_5_72h",
    }

    feature_columns = [
        column
        for column in dataset.columns
        if column not in excluded
        and pd.api.types.is_numeric_dtype(
            dataset[column]
        )
    ]

    X_train = train[feature_columns]
    X_test = test[feature_columns]

    logger.info(
        "Samples: %d",
        len(dataset),
    )

    logger.info(
        "Training samples: %d",
        len(train),
    )

    logger.info(
        "Testing samples: %d",
        len(test),
    )

    # ------------------------------------------------------------------------
    # Persistence baseline
    # ------------------------------------------------------------------------

    persistence_predictions = test[
        persistence_column
    ]

    persistence_mask = (
        persistence_predictions.notna()
        & y_test.notna()
    )

    persistence_metrics = calculate_metrics(
        y_test[persistence_mask],
        persistence_predictions[persistence_mask]
        .to_numpy(),
    )

    logger.info(
        "Persistence | MAE=%.4f | RMSE=%.4f | R2=%.4f",
        persistence_metrics["mae"],
        persistence_metrics["rmse"],
        persistence_metrics["r2"],
    )

    # ------------------------------------------------------------------------
    # Random Forest
    # ------------------------------------------------------------------------

    rf_model = build_random_forest()

    rf_model.fit(
        X_train,
        y_train,
    )

    rf_predictions = rf_model.predict(
        X_test
    )

    rf_metrics = calculate_metrics(
        y_test,
        rf_predictions,
    )

    logger.info(
        "Random Forest | MAE=%.4f | RMSE=%.4f | R2=%.4f",
        rf_metrics["mae"],
        rf_metrics["rmse"],
        rf_metrics["r2"],
    )

    # ------------------------------------------------------------------------
    # HistGradientBoosting
    # ------------------------------------------------------------------------

    hgb_model = build_hist_gradient_boosting()

    hgb_model.fit(
        X_train,
        y_train,
    )

    hgb_predictions = hgb_model.predict(
        X_test
    )

    hgb_metrics = calculate_metrics(
        y_test,
        hgb_predictions,
    )

    logger.info(
        "HistGradientBoosting | MAE=%.4f | RMSE=%.4f | R2=%.4f",
        hgb_metrics["mae"],
        hgb_metrics["rmse"],
        hgb_metrics["r2"],
    )

    # ------------------------------------------------------------------------
    # Model selection
    # ------------------------------------------------------------------------

    candidates = {
        "persistence": persistence_metrics,
        "random_forest": rf_metrics,
        "hist_gradient_boosting": hgb_metrics,
    }

    best_model_name = min(
        candidates,
        key=lambda name: candidates[name]["mae"],
    )

    logger.info(
        "BEST MODEL | %s | MAE=%.4f",
        best_model_name,
        candidates[best_model_name]["mae"],
    )

    return {
        "horizon": horizon,
        "target": target_column,
        "samples": len(dataset),
        "train_samples": len(train),
        "test_samples": len(test),
        "persistence": persistence_metrics,
        "random_forest": rf_metrics,
        "hist_gradient_boosting": hgb_metrics,
        "best_model": best_model_name,
        "best_metrics": candidates[
            best_model_name
        ],
    }


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    """Run complete model benchmark."""

    logger.info("=" * 72)
    logger.info("PEARLS AQI PREDICTOR")
    logger.info("ADVANCED MODEL BENCHMARK")
    logger.info("=" * 72)

    df = load_data()

    results = {}

    for horizon, target in TARGETS.items():

        results[horizon] = evaluate_horizon(
            df=df,
            horizon=horizon,
            target_column=target,
        )

    report = {
        "dataset": str(DATA_PATH),
        "dataset_rows": int(len(df)),
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "models": [
            "persistence",
            "random_forest",
            "hist_gradient_boosting",
        ],
        "results": results,
    }

    report_path = (
        REPORT_DIR
        / "advanced_model_benchmark.json"
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=4,
        ),
        encoding="utf-8",
    )

    logger.info(
        "Benchmark report saved: %s",
        report_path,
    )

    logger.info("=" * 72)
    logger.info(
        "ADVANCED MODEL BENCHMARK COMPLETED."
    )
    logger.info("=" * 72)


if __name__ == "__main__":
    main()