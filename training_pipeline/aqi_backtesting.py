"""
Pearls AQI Predictor
====================

Production-grade daily AQI forecasting backtesting.

Forecast horizons:
    Day +1
    Day +2
    Day +3

Baselines:
    - Persistence
    - 7-day seasonal lag

Candidate models:
    - Ridge
    - Random Forest
    - Extra Trees
    - HistGradientBoosting

Evaluation:
    - Chronological train/validation/test split
    - No random shuffling
    - No future target leakage
    - Validation-based model selection
    - Untouched chronological test evaluation

Important:
    Current target is calculated Punjab-aligned AQI,
    not an official EPA Punjab AQI observation.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
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
    / "aqi_model_features.parquet"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "model_evaluation"
)

RESULTS_FILE = (
    OUTPUT_DIR
    / "aqi_backtesting_results.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "aqi_backtesting_summary.json"
)


# ============================================================================
# CONFIGURATION
# ============================================================================

TARGETS = {
    "day_1": "target_aqi_day_1",
    "day_2": "target_aqi_day_2",
    "day_3": "target_aqi_day_3",
}

TEST_RATIO = 0.20
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
# LOAD DATA
# ============================================================================

def load_features() -> pd.DataFrame:
    """Load the daily AQI forecasting feature dataset."""

    if not FEATURE_FILE.exists():
        raise FileNotFoundError(
            f"AQI feature dataset not found: {FEATURE_FILE}"
        )

    logger.info(
        "Loading AQI feature dataset: %s",
        FEATURE_FILE,
    )

    df = pd.read_parquet(
        FEATURE_FILE
    )

    if df.empty:
        raise ValueError(
            "AQI feature dataset is empty."
        )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    ).dt.normalize()

    if df["date"].isna().any():
        raise ValueError(
            "Invalid dates found in AQI feature dataset."
        )

    df = (
        df
        .sort_values("date")
        .drop_duplicates("date")
        .reset_index(drop=True)
    )

    logger.info(
        "Loaded rows=%d | columns=%d",
        len(df),
        len(df.columns),
    )

    logger.info(
        "Date range: %s → %s",
        df["date"].min(),
        df["date"].max(),
    )

    return df


# ============================================================================
# FEATURE SELECTION
# ============================================================================

def get_feature_columns(
    df: pd.DataFrame,
) -> list[str]:
    """
    Select numeric predictor columns.

    Current AQI is a valid predictor because it is known at the
    forecasting origin.

    Future target columns are excluded.
    """

    excluded = {
        "date",
        "city",
        "aqi_category",
        "aqi_color",
        "methodology",
        "source",
        "target_aqi_day_1",
        "target_aqi_day_2",
        "target_aqi_day_3",
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
            "No numeric AQI forecasting features found."
        )

    logger.info(
        "AQI model feature count: %d",
        len(features),
    )

    return features


# ============================================================================
# CHRONOLOGICAL SPLIT
# ============================================================================

def chronological_split(
    df: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Create chronological train/validation/test partitions.

    No shuffling is used.
    """

    n_rows = len(df)

    test_size = max(
        1,
        int(
            n_rows * TEST_RATIO
        ),
    )

    validation_size = max(
        1,
        int(
            n_rows * VALIDATION_RATIO
        ),
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

    if train_end <= 0:
        raise ValueError(
            "Dataset is too small for chronological splitting."
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

    if not (
        train["date"].max()
        < validation["date"].min()
        < test["date"].min()
    ):
        raise ValueError(
            "Chronological split integrity check failed."
        )

    logger.info(
        "Train: %d | %s → %s",
        len(train),
        train["date"].min(),
        train["date"].max(),
    )

    logger.info(
        "Validation: %d | %s → %s",
        len(validation),
        validation["date"].min(),
        validation["date"].max(),
    )

    logger.info(
        "Test: %d | %s → %s",
        len(test),
        test["date"].min(),
        test["date"].max(),
    )

    return (
        train,
        validation,
        test,
    )


# ============================================================================
# METRICS
# ============================================================================

def calculate_metrics(
    y_true: pd.Series,
    y_pred: pd.Series | np.ndarray,
) -> dict[str, float]:
    """Calculate forecasting metrics."""

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
# BASELINES
# ============================================================================

def evaluate_persistence(
    *,
    target_column: str,
    test: pd.DataFrame,
    horizon: str,
) -> dict | None:
    """
    Persistence baseline.

    Predict future AQI using the current day's known AQI.
    """

    evaluation = test[
        [
            "date",
            "aqi",
            target_column,
        ]
    ].dropna()

    if evaluation.empty:
        return None

    metrics = calculate_metrics(
        evaluation[target_column],
        evaluation["aqi"],
    )

    logger.info(
        "persistence | %s | "
        "MAE=%.4f | RMSE=%.4f | R2=%.4f | n=%d",
        horizon,
        metrics["mae"],
        metrics["rmse"],
        metrics["r2"],
        len(evaluation),
    )

    return {
        "horizon": horizon,
        "model": "persistence",
        "evaluation_set": "test",
        "evaluation_rows": int(
            len(evaluation)
        ),
        **metrics,
    }


def evaluate_seasonal_7day(
    *,
    target_column: str,
    test: pd.DataFrame,
    horizon: str,
) -> dict | None:
    """
    Seven-day seasonal baseline.

    Uses the explicitly engineered aqi_lag_7d feature.

    This is leakage-safe because the value comes from the past.
    """

    if "aqi_lag_7d" not in test.columns:
        raise ValueError(
            "aqi_lag_7d is required for the seasonal baseline."
        )

    evaluation = test[
        [
            "date",
            "aqi_lag_7d",
            target_column,
        ]
    ].dropna()

    if evaluation.empty:
        return None

    metrics = calculate_metrics(
        evaluation[target_column],
        evaluation["aqi_lag_7d"],
    )

    logger.info(
        "seasonal_7d | %s | "
        "MAE=%.4f | RMSE=%.4f | R2=%.4f | n=%d",
        horizon,
        metrics["mae"],
        metrics["rmse"],
        metrics["r2"],
        len(evaluation),
    )

    return {
        "horizon": horizon,
        "model": "seasonal_7d",
        "evaluation_set": "test",
        "evaluation_rows": int(
            len(evaluation)
        ),
        **metrics,
    }


# ============================================================================
# MODEL FACTORY
# ============================================================================

def build_models() -> dict[str, Pipeline]:
    """Create candidate regression models."""

    return {
        "ridge": Pipeline(
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
                        n_estimators=400,
                        max_depth=12,
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
                        n_estimators=400,
                        max_depth=14,
                        min_samples_leaf=2,
                        max_features="sqrt",
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),

        "hist_gradient_boosting": Pipeline(
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
                        max_iter=250,
                        learning_rate=0.05,
                        max_leaf_nodes=15,
                        l2_regularization=1.0,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


# ============================================================================
# HORIZON EVALUATION
# ============================================================================

def evaluate_horizon(
    *,
    horizon: str,
    target_column: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: list[str],
) -> list[dict]:
    """Evaluate baselines and ML candidates for one horizon."""

    logger.info("=" * 72)
    logger.info(
        "AQI FORECAST HORIZON: %s",
        horizon,
    )
    logger.info(
        "TARGET: %s",
        target_column,
    )
    logger.info("=" * 72)

    train_data = train.dropna(
        subset=[target_column]
    )

    validation_data = validation.dropna(
        subset=[target_column]
    )

    test_data = test.dropna(
        subset=[target_column]
    )

    results: list[dict] = []

    # ------------------------------------------------------------------------
    # TEST BASELINES
    # ------------------------------------------------------------------------

    persistence = evaluate_persistence(
        target_column=target_column,
        test=test_data,
        horizon=horizon,
    )

    if persistence:
        results.append(
            persistence
        )

    seasonal = evaluate_seasonal_7day(
        target_column=target_column,
        test=test_data,
        horizon=horizon,
    )

    if seasonal:
        results.append(
            seasonal
        )

    # ------------------------------------------------------------------------
    # PREPARE ML DATA
    # ------------------------------------------------------------------------

    X_train = train_data[
        feature_columns
    ]

    y_train = train_data[
        target_column
    ]

    X_validation = validation_data[
        feature_columns
    ]

    y_validation = validation_data[
        target_column
    ]

    X_test = test_data[
        feature_columns
    ]

    y_test = test_data[
        target_column
    ]

    # ------------------------------------------------------------------------
    # TRAIN MODELS
    # ------------------------------------------------------------------------

    models = build_models()

    for model_name, model in models.items():

        logger.info(
            "Training %s | %s",
            model_name,
            horizon,
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
            "%s | %s | "
            "validation MAE=%.4f | "
            "test MAE=%.4f | "
            "test RMSE=%.4f | "
            "test R2=%.4f",
            model_name,
            horizon,
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
    """Run complete daily AQI backtesting."""

    logger.info("=" * 72)
    logger.info(
        "PEARLS AQI PREDICTOR"
    )
    logger.info(
        "DAILY AQI BACKTESTING"
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

    train, validation, test = (
        chronological_split(
            df
        )
    )

    all_results: list[dict] = []

    for horizon, target_column in (
        TARGETS.items()
    ):

        horizon_results = (
            evaluate_horizon(
                horizon=horizon,
                target_column=target_column,
                train=train,
                validation=validation,
                test=test,
                feature_columns=feature_columns,
            )
        )

        all_results.extend(
            horizon_results
        )

    results_df = pd.DataFrame(
        all_results
    )

    if results_df.empty:
        raise RuntimeError(
            "No AQI backtesting results produced."
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

    # ------------------------------------------------------------------------
    # Select best ML model using validation MAE.
    # Baselines are not selected as production ML models.
    # ------------------------------------------------------------------------

    validation_models = results_df[
        (
            results_df[
                "evaluation_set"
            ]
            == "validation"
        )
        & (
            ~results_df[
                "model"
            ].isin(
                [
                    "persistence",
                    "seasonal_7d",
                ]
            )
        )
    ].copy()

    selections = []

    for horizon in TARGETS:

        candidates = validation_models[
            validation_models["horizon"]
            == horizon
        ]

        if candidates.empty:
            continue

        best = (
            candidates
            .sort_values("mae")
            .iloc[0]
        )

        test_candidate = results_df[
            (
                results_df["horizon"]
                == horizon
            )
            & (
                results_df["model"]
                == best["model"]
            )
            & (
                results_df["evaluation_set"]
                == "test"
            )
        ]

        if test_candidate.empty:
            continue

        test_row = test_candidate.iloc[0]

        selections.append(
            {
                "horizon": horizon,
                "selected_model": best["model"],
                "validation_mae": float(
                    best["mae"]
                ),
                "test_mae": float(
                    test_row["mae"]
                ),
                "test_rmse": float(
                    test_row["rmse"]
                ),
                "test_r2": float(
                    test_row["r2"]
                ),
            }
        )

    # ------------------------------------------------------------------------
    # Compare against persistence baseline.
    # ------------------------------------------------------------------------

    baseline_comparisons = []

    for horizon in TARGETS:

        selected = next(
            (
                item
                for item in selections
                if item["horizon"]
                == horizon
            ),
            None,
        )

        baseline_row = results_df[
            (
                results_df["horizon"]
                == horizon
            )
            & (
                results_df["model"]
                == "persistence"
            )
            & (
                results_df["evaluation_set"]
                == "test"
            )
        ]

        if (
            selected is None
            or baseline_row.empty
        ):
            continue

        baseline_mae = float(
            baseline_row.iloc[0]["mae"]
        )

        improvement = (
            (
                baseline_mae
                - selected["test_mae"]
            )
            / baseline_mae
            * 100
        )

        baseline_comparisons.append(
            {
                "horizon": horizon,
                "selected_model": (
                    selected[
                        "selected_model"
                    ]
                ),
                "baseline_mae": baseline_mae,
                "model_mae": selected[
                    "test_mae"
                ],
                "mae_improvement_percent": float(
                    improvement
                ),
                "model_beats_baseline": bool(
                    selected[
                        "test_mae"
                    ]
                    < baseline_mae
                ),
            }
        )

    summary = {
        "project": "Pearls AQI Predictor",

        "target": (
            "calculated Punjab-aligned AQI"
        ),

        "evaluation": (
            "chronological train/validation/test"
        ),

        "selected_models": selections,

        "baseline_comparison": (
            baseline_comparisons
        ),

        "all_results": (
            all_results
        ),
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
        "DAILY AQI BACKTESTING COMPLETED."
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
        "\nTEST RESULTS"
    )

    print(
        results_df[
            results_df[
                "evaluation_set"
            ]
            == "test"
        ][
            [
                "horizon",
                "model",
                "mae",
                "rmse",
                "r2",
                "evaluation_rows",
            ]
        ]
        .sort_values(
            [
                "horizon",
                "mae",
            ]
        )
        .to_string(
            index=False
        )
    )

    print(
        "\nMODEL VS PERSISTENCE"
    )

    print(
        pd.DataFrame(
            baseline_comparisons
        ).to_string(
            index=False
        )
    )


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()