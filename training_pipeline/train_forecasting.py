"""
Production-grade time-series forecasting pipeline.

Pearls AQI Predictor
--------------------

Responsibilities:
- Load engineered features.
- Prevent target leakage.
- Use chronological train/validation/test splits.
- Train multiple forecasting models.
- Evaluate 24h, 48h and 72h horizons.
- Persist metrics and trained models.
- Produce a machine-readable training report.

No random train/test split is used.
"""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    ExtraTreesRegressor,
    RandomForestRegressor,
    HistGradientBoostingRegressor,
)
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

FEATURE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_features.parquet"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "forecasting"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "model_evaluation"
)

METRICS_FILE = (
    REPORT_DIR
    / "forecasting_metrics.json"
)

LEADERBOARD_FILE = (
    REPORT_DIR
    / "model_leaderboard.csv"
)


# ============================================================================
# CONFIGURATION
# ============================================================================

TARGETS = {
    "24h": "target_pm2_5_24h",
    "48h": "target_pm2_5_48h",
    "72h": "target_pm2_5_72h",
}

TEST_RATIO = 0.15
VALIDATION_RATIO = 0.15

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
# MODEL SPECIFICATION
# ============================================================================

@dataclass(frozen=True)
class ModelSpec:
    """Configuration for a forecasting model."""

    name: str
    estimator: object


def build_models() -> list[ModelSpec]:
    """Build the candidate regression models."""

    return [
        ModelSpec(
            name="ridge",
            estimator=Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="median"
                        ),
                    ),
                    (
                        "model",
                        Ridge(
                            alpha=10.0
                        ),
                    ),
                ]
            ),
        ),
        ModelSpec(
            name="random_forest",
            estimator=Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="median"
                        ),
                    ),
                    (
                        "model",
                        RandomForestRegressor(
                            n_estimators=400,
                            max_depth=18,
                            min_samples_leaf=2,
                            max_features="sqrt",
                            n_jobs=-1,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
        ),
        ModelSpec(
            name="extra_trees",
            estimator=Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="median"
                        ),
                    ),
                    (
                        "model",
                        ExtraTreesRegressor(
                            n_estimators=400,
                            max_depth=20,
                            min_samples_leaf=2,
                            max_features=1.0,
                            n_jobs=-1,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
        ),
        ModelSpec(
            name="hist_gradient_boosting",
            estimator=Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="median"
                        ),
                    ),
                    (
                        "model",
                        HistGradientBoostingRegressor(
                            max_iter=300,
                            learning_rate=0.05,
                            max_leaf_nodes=31,
                            l2_regularization=1.0,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
        ),
    ]


# ============================================================================
# LOAD FEATURES
# ============================================================================

def load_features() -> pd.DataFrame:
    """Load the engineered feature dataset."""

    logger.info(
        "Loading engineered features: %s",
        FEATURE_FILE,
    )

    if not FEATURE_FILE.exists():
        raise FileNotFoundError(
            f"Feature dataset not found: {FEATURE_FILE}"
        )

    df = pd.read_parquet(FEATURE_FILE)

    if df.empty:
        raise ValueError(
            "Feature dataset is empty."
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
    )

    df = (
        df.sort_values("timestamp")
        .reset_index(drop=True)
    )

    logger.info(
        "Features loaded: rows=%d | columns=%d",
        len(df),
        len(df.columns),
    )

    return df


# ============================================================================
# FEATURE SELECTION
# ============================================================================

def identify_features(
    df: pd.DataFrame,
) -> list[str]:
    """
    Identify model input columns.

    Targets and identifiers are explicitly excluded.
    """

    excluded = {
        "timestamp",
        "city",
        "target_pm2_5_24h",
        "target_pm2_5_48h",
        "target_pm2_5_72h",
    }

    features = [
        column
        for column in df.columns
        if column not in excluded
        and pd.api.types.is_numeric_dtype(
            df[column]
        )
    ]

    if not features:
        raise ValueError(
            "No numeric model features found."
        )

    logger.info(
        "Model features: %d",
        len(features),
    )

    return features


# ============================================================================
# CHRONOLOGICAL SPLIT
# ============================================================================

def chronological_split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data chronologically.

    Train -> earliest observations
    Validation -> following observations
    Test -> latest observations
    """

    n = len(df)

    test_size = int(
        n * TEST_RATIO
    )

    validation_size = int(
        n * VALIDATION_RATIO
    )

    train_end = (
        n
        - validation_size
        - test_size
    )

    validation_end = (
        n
        - test_size
    )

    train = df.iloc[:train_end].copy()

    validation = df.iloc[
        train_end:validation_end
    ].copy()

    test = df.iloc[
        validation_end:
    ].copy()

    logger.info(
        "Chronological split:"
    )

    logger.info(
        "Train: %d rows | %s → %s",
        len(train),
        train["timestamp"].min(),
        train["timestamp"].max(),
    )

    logger.info(
        "Validation: %d rows | %s → %s",
        len(validation),
        validation["timestamp"].min(),
        validation["timestamp"].max(),
    )

    logger.info(
        "Test: %d rows | %s → %s",
        len(test),
        test["timestamp"].min(),
        test["timestamp"].max(),
    )

    if not (
        train["timestamp"].max()
        < validation["timestamp"].min()
        < test["timestamp"].min()
    ):
        raise ValueError(
            "Chronological split integrity check failed."
        )

    return train, validation, test


# ============================================================================
# METRICS
# ============================================================================

def calculate_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Calculate regression metrics."""

    mae = mean_absolute_error(
        y_true,
        y_pred,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred,
        )
    )

    r2 = r2_score(
        y_true,
        y_pred,
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
    horizon_name: str,
    target_column: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[dict, dict]:
    """Train and evaluate all candidate models for one horizon."""

    logger.info("=" * 72)
    logger.info(
        "FORECAST HORIZON: %s",
        horizon_name,
    )
    logger.info(
        "TARGET: %s",
        target_column,
    )
    logger.info("=" * 72)

    train = train.dropna(
        subset=[target_column]
    )

    validation = validation.dropna(
        subset=[target_column]
    )

    test = test.dropna(
        subset=[target_column]
    )

    X_train = train[feature_columns]
    y_train = train[target_column]

    X_validation = validation[
        feature_columns
    ]
    y_validation = validation[
        target_column
    ]

    X_test = test[feature_columns]
    y_test = test[target_column]

    models = build_models()

    horizon_results = []

    trained_models = {}

    for specification in models:

        logger.info(
            "Training model: %s",
            specification.name,
        )

        model = specification.estimator

        model.fit(
            X_train,
            y_train,
        )

        validation_prediction = model.predict(
            X_validation
        )

        test_prediction = model.predict(
            X_test
        )

        validation_metrics = calculate_metrics(
            y_validation,
            validation_prediction,
        )

        test_metrics = calculate_metrics(
            y_test,
            test_prediction,
        )

        logger.info(
            "%s | validation MAE=%.4f | "
            "validation RMSE=%.4f | "
            "test MAE=%.4f | "
            "test RMSE=%.4f | "
            "test R2=%.4f",
            specification.name,
            validation_metrics["mae"],
            validation_metrics["rmse"],
            test_metrics["mae"],
            test_metrics["rmse"],
            test_metrics["r2"],
        )

        horizon_results.append(
            {
                "horizon": horizon_name,
                "model": specification.name,
                "validation_mae": (
                    validation_metrics["mae"]
                ),
                "validation_rmse": (
                    validation_metrics["rmse"]
                ),
                "validation_r2": (
                    validation_metrics["r2"]
                ),
                "test_mae": (
                    test_metrics["mae"]
                ),
                "test_rmse": (
                    test_metrics["rmse"]
                ),
                "test_r2": (
                    test_metrics["r2"]
                ),
            }
        )

        trained_models[
            specification.name
        ] = model

    results_df = pd.DataFrame(
        horizon_results
    )

    # Select model using validation MAE.
    best_row = (
        results_df
        .sort_values("validation_mae")
        .iloc[0]
    )

    best_model_name = best_row["model"]

    logger.info(
        "Selected model for %s: %s",
        horizon_name,
        best_model_name,
    )

    best_model = trained_models[
        best_model_name
    ]

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        MODEL_DIR
        / f"best_model_{horizon_name}.pkl"
    )

    with model_path.open(
        "wb"
    ) as file:

        pickle.dump(
            best_model,
            file,
        )

    logger.info(
        "Saved model: %s",
        model_path,
    )

    selection = {
        "horizon": horizon_name,
        "target": target_column,
        "selected_model": best_model_name,
        "model_path": str(model_path),
        "validation_mae": float(
            best_row["validation_mae"]
        ),
        "test_mae": float(
            best_row["test_mae"]
        ),
        "test_rmse": float(
            best_row["test_rmse"]
        ),
        "test_r2": float(
            best_row["test_r2"]
        ),
    }

    return (
        horizon_results,
        selection,
    )


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    """Execute the complete forecasting training pipeline."""

    logger.info("=" * 72)
    logger.info("PEARLS AQI PREDICTOR")
    logger.info("PRODUCTION FORECASTING TRAINING")
    logger.info("=" * 72)

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_features()

    feature_columns = identify_features(
        df
    )

    train, validation, test = chronological_split(
        df
    )

    all_results = []
    selections = []

    for horizon_name, target_column in TARGETS.items():

        results, selection = train_horizon(
            horizon_name=horizon_name,
            target_column=target_column,
            train=train,
            validation=validation,
            test=test,
            feature_columns=feature_columns,
        )

        all_results.extend(results)
        selections.append(selection)

    leaderboard = pd.DataFrame(
        all_results
    )

    leaderboard = leaderboard.sort_values(
        [
            "horizon",
            "validation_mae",
        ]
    )

    leaderboard.to_csv(
        LEADERBOARD_FILE,
        index=False,
    )

    metrics = {
        "project": "Pearls AQI Predictor",
        "training_type": "time_series_forecasting",
        "feature_count": len(
            feature_columns
        ),
        "dataset_rows": len(df),
        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "targets": TARGETS,
        "selected_models": selections,
        "leaderboard": all_results,
    }

    METRICS_FILE.write_text(
        json.dumps(
            metrics,
            indent=4,
        ),
        encoding="utf-8",
    )

    logger.info("=" * 72)
    logger.info(
        "FORECASTING TRAINING COMPLETED."
    )
    logger.info(
        "Leaderboard: %s",
        LEADERBOARD_FILE,
    )
    logger.info(
        "Metrics: %s",
        METRICS_FILE,
    )
    logger.info("=" * 72)


if __name__ == "__main__":
    main()