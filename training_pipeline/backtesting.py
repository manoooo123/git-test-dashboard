"""
Pearls AQI Predictor
====================

Leakage-safe time-series backtesting.

Current target:
    PM2.5 forecasting

Forecast horizons:
    24h
    48h
    72h

Baselines:
    1. Persistence
    2. 24-hour lag
    3. 168-hour lag

Models:
    4. Random Forest
    5. Extra Trees

Important
---------
All baselines use only information available at prediction time.

The 24-hour baseline directly uses the already validated
pm2_5_lag_24h feature.

No random shuffling.
No synthetic data.
No future observations.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    ExtraTreesRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline


# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FEATURE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_features.parquet"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "model_evaluation"
)

RESULTS_FILE = (
    OUTPUT_DIR
    / "backtesting_results.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "backtesting_summary.json"
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

N_ESTIMATORS = 300

MIN_TRAIN_ROWS = 3000


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
# LOAD FEATURES
# ============================================================================

def load_features() -> pd.DataFrame:
    """Load the validated time-aware feature dataset."""

    if not FEATURE_FILE.exists():
        raise FileNotFoundError(
            f"Feature dataset not found: {FEATURE_FILE}"
        )

    logger.info(
        "Loading feature dataset: %s",
        FEATURE_FILE,
    )

    df = pd.read_parquet(
        FEATURE_FILE
    )

    if df.empty:
        raise ValueError(
            "Feature dataset is empty."
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
        errors="coerce",
    )

    if df["timestamp"].isna().any():
        raise ValueError(
            "Invalid timestamps detected."
        )

    df = (
        df
        .sort_values("timestamp")
        .drop_duplicates("timestamp")
        .reset_index(drop=True)
    )

    required_baseline_columns = {
        "pm2_5",
        "pm2_5_lag_24h",
    }

    missing = (
        required_baseline_columns
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            "Required baseline columns missing: "
            + ", ".join(sorted(missing))
        )

    logger.info(
        "Loaded feature dataset: %d rows | %d columns",
        len(df),
        len(df.columns),
    )

    return df


# ============================================================================
# FEATURE SELECTION
# ============================================================================

def get_feature_columns(
    df: pd.DataFrame,
) -> list[str]:
    """Identify numeric model input columns."""

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
            "No numeric feature columns found."
        )

    return features


# ============================================================================
# CHRONOLOGICAL SPLIT
# ============================================================================

def split_dataset(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data chronologically."""

    n_rows = len(df)

    test_size = int(
        n_rows * TEST_RATIO
    )

    validation_size = int(
        n_rows * VALIDATION_RATIO
    )

    train_end = (
        n_rows
        - validation_size
        - test_size
    )

    validation_end = (
        n_rows
        - test_size
    )

    train = df.iloc[
        :train_end
    ].copy()

    validation = df.iloc[
        train_end:validation_end
    ].copy()

    test = df.iloc[
        validation_end:
    ].copy()

    if len(train) < MIN_TRAIN_ROWS:
        raise ValueError(
            "Training dataset is too small."
        )

    if not (
        train["timestamp"].max()
        < validation["timestamp"].min()
        < test["timestamp"].min()
    ):
        raise ValueError(
            "Chronological ordering validation failed."
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

    return train, validation, test


# ============================================================================
# METRICS
# ============================================================================

def calculate_metrics(
    y_true: pd.Series,
    y_pred: pd.Series | np.ndarray,
) -> dict[str, float]:
    """Calculate regression metrics."""

    return {
        "mae": float(
            mean_absolute_error(
                y_true,
                y_pred,
            )
        ),
        "rmse": float(
            np.sqrt(
                mean_squared_error(
                    y_true,
                    y_pred,
                )
            )
        ),
        "r2": float(
            r2_score(
                y_true,
                y_pred,
            )
        ),
    }


# ============================================================================
# BASELINE FROM EXISTING SAFE FEATURES
# ============================================================================

def evaluate_feature_baseline(
    *,
    horizon: str,
    target_column: str,
    test: pd.DataFrame,
    baseline_column: str,
    baseline_name: str,
) -> dict | None:
    """
    Evaluate a baseline using a feature that only contains
    information available at prediction time.
    """

    evaluation = test[
        [
            "timestamp",
            target_column,
            baseline_column,
        ]
    ].dropna()

    if evaluation.empty:
        logger.warning(
            "%s | no valid rows",
            baseline_name,
        )
        return None

    result = calculate_metrics(
        evaluation[target_column],
        evaluation[baseline_column],
    )

    logger.info(
        "%s | test MAE=%.4f | "
        "test RMSE=%.4f | "
        "test R2=%.4f | n=%d",
        baseline_name,
        result["mae"],
        result["rmse"],
        result["r2"],
        len(evaluation),
    )

    return {
        "horizon": horizon,
        "model": baseline_name,
        "evaluation_set": "test",
        "evaluation_rows": int(
            len(evaluation)
        ),
        **result,
    }


# ============================================================================
# BUILD ML MODELS
# ============================================================================

def build_models() -> dict[str, Pipeline]:
    """Build candidate ML regressors."""

    return {
        "random_forest": Pipeline(
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
                        n_estimators=N_ESTIMATORS,
                        max_depth=18,
                        min_samples_leaf=2,
                        max_features="sqrt",
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),

        "extra_trees": Pipeline(
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
                        n_estimators=N_ESTIMATORS,
                        max_depth=20,
                        min_samples_leaf=2,
                        max_features=1.0,
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


# ============================================================================
# EVALUATE HORIZON
# ============================================================================

def evaluate_horizon(
    *,
    horizon: str,
    target_column: str,
    baseline_column: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: list[str],
) -> list[dict]:
    """Evaluate baselines and ML models for one horizon."""

    logger.info("=" * 72)
    logger.info(
        "FORECAST HORIZON: %s",
        horizon,
    )
    logger.info(
        "TARGET: %s",
        target_column,
    )
    logger.info("=" * 72)

    train_target = train.dropna(
        subset=[target_column]
    )

    validation_target = validation.dropna(
        subset=[target_column]
    )

    test_target = test.dropna(
        subset=[target_column]
    )

    results: list[dict] = []

    # ------------------------------------------------------------------------
    # Persistence baseline
    # ------------------------------------------------------------------------

    persistence_result = (
        evaluate_feature_baseline(
            horizon=horizon,
            target_column=target_column,
            test=test_target,
            baseline_column="pm2_5",
            baseline_name="persistence",
        )
    )

    if persistence_result:
        results.append(
            persistence_result
        )

    # ------------------------------------------------------------------------
    # 24-hour historical baseline
    # ------------------------------------------------------------------------

    lag24_result = (
        evaluate_feature_baseline(
            horizon=horizon,
            target_column=target_column,
            test=test_target,
            baseline_column="pm2_5_lag_24h",
            baseline_name="lag_24h",
        )
    )

    if lag24_result:
        results.append(
            lag24_result
        )

    # ------------------------------------------------------------------------
    # ML MODELS
    # ------------------------------------------------------------------------

    models = build_models()

    X_train = train_target[
        feature_columns
    ]

    y_train = train_target[
        target_column
    ]

    X_validation = validation_target[
        feature_columns
    ]

    y_validation = validation_target[
        target_column
    ]

    X_test = test_target[
        feature_columns
    ]

    y_test = test_target[
        target_column
    ]

    for model_name, model in models.items():

        logger.info(
            "Training %s",
            model_name,
        )

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

        validation_metrics = (
            calculate_metrics(
                y_validation,
                validation_prediction,
            )
        )

        test_metrics = (
            calculate_metrics(
                y_test,
                test_prediction,
            )
        )

        logger.info(
            "%s | validation MAE=%.4f | "
            "test MAE=%.4f | "
            "test RMSE=%.4f | "
            "test R2=%.4f",
            model_name,
            validation_metrics["mae"],
            test_metrics["mae"],
            test_metrics["rmse"],
            test_metrics["r2"],
        )

        results.append(
            {
                "horizon": horizon,
                "model": model_name,
                "evaluation_set": "validation",
                "evaluation_rows": int(
                    len(y_validation)
                ),
                **validation_metrics,
            }
        )

        results.append(
            {
                "horizon": horizon,
                "model": model_name,
                "evaluation_set": "test",
                "evaluation_rows": int(
                    len(y_test)
                ),
                **test_metrics,
            }
        )

    return results


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    """Execute complete backtesting."""

    logger.info("=" * 72)
    logger.info(
        "PEARLS AQI PREDICTOR"
    )
    logger.info(
        "LEAKAGE-SAFE BASELINE + BACKTESTING"
    )
    logger.info("=" * 72)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_features()

    feature_columns = get_feature_columns(
        df
    )

    train, validation, test = split_dataset(
        df
    )

    baseline_map = {
        "24h": "pm2_5_lag_24h",
        "48h": "pm2_5_lag_24h",
        "72h": "pm2_5_lag_24h",
    }

    all_results: list[dict] = []

    for horizon, target_column in (
        TARGETS.items()
    ):

        results = evaluate_horizon(
            horizon=horizon,
            target_column=target_column,
            baseline_column=baseline_map[horizon],
            train=train,
            validation=validation,
            test=test,
            feature_columns=feature_columns,
        )

        all_results.extend(
            results
        )

    results_df = pd.DataFrame(
        all_results
    )

    if results_df.empty:
        raise RuntimeError(
            "No backtesting results were produced."
        )

    results_df.sort_values(
        [
            "horizon",
            "evaluation_set",
            "mae",
        ],
        inplace=True,
    )

    results_df.to_csv(
        RESULTS_FILE,
        index=False,
    )

    test_results = results_df[
        results_df["evaluation_set"]
        == "test"
    ].copy()

    leaderboard = (
        test_results
        .sort_values(
            [
                "horizon",
                "mae",
            ]
        )
        .groupby(
            "horizon",
            as_index=False,
        )
        .first()
    )

    summary = {
        "project": "Pearls AQI Predictor",
        "evaluation_method": (
            "chronological holdout"
        ),
        "baseline_method": (
            "validated time-aware features"
        ),
        "best_test_models": (
            leaderboard[
                [
                    "horizon",
                    "model",
                    "mae",
                    "rmse",
                    "r2",
                    "evaluation_rows",
                ]
            ]
            .to_dict(
                orient="records"
            )
        ),
        "all_results": all_results,
    }

    SUMMARY_FILE.write_text(
        json.dumps(
            summary,
            indent=4,
        ),
        encoding="utf-8",
    )

    logger.info("=" * 72)
    logger.info(
        "BACKTESTING COMPLETED SUCCESSFULLY."
    )
    logger.info(
        "Results: %s",
        RESULTS_FILE,
    )
    logger.info(
        "Summary: %s",
        SUMMARY_FILE,
    )
    logger.info("=" * 72)

    print(
        "\nLEAKAGE-SAFE TEST LEADERBOARD"
    )

    print(
        leaderboard[
            [
                "horizon",
                "model",
                "mae",
                "rmse",
                "r2",
                "evaluation_rows",
            ]
        ].to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()