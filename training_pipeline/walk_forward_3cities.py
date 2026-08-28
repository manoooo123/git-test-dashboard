"""
3-CITY WALK-FORWARD VALIDATION
PEARLS AQI PREDICTOR

Validates 24h, 48h and 72h AQI forecasting using
strict chronological walk-forward validation.

Cities:
    Lahore
    Islamabad
    Faisalabad
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline


# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_features_3cities.csv"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "model_evaluation"
    / "3cities"
)

RESULTS_PATH = REPORT_DIR / "walk_forward_3cities_results.csv"
SUMMARY_PATH = REPORT_DIR / "walk_forward_3cities_summary.json"

RANDOM_STATE = 42

CITIES = [
    "Lahore",
    "Islamabad",
    "Faisalabad",
]

FORECAST_TARGETS = {
    "24h": "target_24h",
    "48h": "target_48h",
    "72h": "target_72h",
}

EXCLUDED_COLUMNS = {
    "city",
    "hour",
    "target_24h",
    "target_48h",
    "target_72h",
    "is_missing_hour",
}


# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================================
# DATA LOADING
# ============================================================================

def load_dataset() -> pd.DataFrame:
    """Load and validate the 3-city feature dataset."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    required_columns = {
        "city",
        "hour",
        "target_24h",
        "target_48h",
        "target_72h",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    df["hour"] = pd.to_datetime(
        df["hour"],
        errors="coerce",
    )

    if df["hour"].isna().any():
        raise ValueError(
            "Invalid datetime values found in 'hour'."
        )

    df = df[
        df["city"].isin(CITIES)
    ].copy()

    df = df.sort_values(
        ["city", "hour"]
    ).reset_index(drop=True)

    logger.info(
        "Dataset shape: %s",
        df.shape,
    )

    logger.info(
        "Cities:\n%s",
        df["city"].value_counts().to_string(),
    )

    return df


# ============================================================================
# FEATURES
# ============================================================================

def get_feature_columns(
    df: pd.DataFrame,
) -> list[str]:
    """Return numeric predictor columns."""

    numeric_columns = df.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    feature_columns = [
        column
        for column in numeric_columns
        if column not in EXCLUDED_COLUMNS
    ]

    if not feature_columns:
        raise ValueError(
            "No numeric feature columns found."
        )

    logger.info(
        "Feature count: %d",
        len(feature_columns),
    )

    return feature_columns


# ============================================================================
# METRICS
# ============================================================================

def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Calculate regression metrics."""

    return {
        "MAE": float(
            mean_absolute_error(
                y_true,
                y_pred,
            )
        ),
        "RMSE": float(
            np.sqrt(
                mean_squared_error(
                    y_true,
                    y_pred,
                )
            )
        ),
        "R2": float(
            r2_score(
                y_true,
                y_pred,
            )
        ),
    }


# ============================================================================
# MODELS
# ============================================================================

def create_ridge() -> Pipeline:
    """Create Ridge regression pipeline."""

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "model",
                Ridge(alpha=10.0),
            ),
        ]
    )


def create_random_forest() -> Pipeline:
    """Create Random Forest pipeline."""

    return Pipeline(
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
                    n_estimators=200,
                    min_samples_leaf=2,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


# ============================================================================
# WALK-FORWARD VALIDATION
# ============================================================================

def evaluate_city_horizon(
    city_df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    horizon_name: str,
    city: str,
) -> list[dict]:
    """
    Perform chronological walk-forward validation.

    The first 70% is used as initial training history.
    The remaining 30% is evaluated in sequential blocks.
    """

    city_df = city_df.sort_values(
        "hour"
    ).reset_index(drop=True)

    usable = city_df[
        feature_columns + [target_column, "hour"]
    ].dropna(
        subset=[target_column]
    ).reset_index(drop=True)

    if len(usable) < 100:
        logger.warning(
            "%s | %s | insufficient samples: %d",
            city,
            horizon_name,
            len(usable),
        )
        return []

    initial_train_size = int(
        len(usable) * 0.70
    )

    validation_size = max(
        24,
        int(len(usable) * 0.10),
    )

    rows = []

    fold = 1
    train_end = initial_train_size

    while train_end + validation_size <= len(usable):

        validation_end = min(
            train_end + validation_size,
            len(usable),
        )

        train_df = usable.iloc[
            :train_end
        ]

        validation_df = usable.iloc[
            train_end:validation_end
        ]

        if len(validation_df) == 0:
            break

        X_train = train_df[
            feature_columns
        ]

        y_train = train_df[
            target_column
        ]

        X_validation = validation_df[
            feature_columns
        ]

        y_validation = validation_df[
            target_column
        ]

        # ------------------------------------------------------------
        # ------------------------------------------------------------
        # Persistence baseline
        # ------------------------------------------------------------

        baseline_prediction = validation_df['pm25_mean'].to_numpy()
        baseline_actual = y_validation.to_numpy()
        baseline_mask = np.isfinite(baseline_prediction) & np.isfinite(baseline_actual)

        if baseline_mask.any():
            baseline_metrics = calculate_metrics(
                baseline_actual[baseline_mask],
                baseline_prediction[baseline_mask],
            )
        else:
            baseline_metrics = {
                'MAE': float('nan'),
                'RMSE': float('nan'),
                'R2': float('nan'),
            }

        # ------------------------------------------------------------
        # Ridge
        # ------------------------------------------------------------

        ridge = create_ridge()

        ridge.fit(
            X_train,
            y_train,
        )

        ridge_prediction = ridge.predict(
            X_validation
        )

        ridge_metrics = calculate_metrics(
            y_validation.to_numpy(),
            ridge_prediction,
        )

        # ------------------------------------------------------------
        # Random Forest
        # ------------------------------------------------------------

        random_forest = create_random_forest()

        random_forest.fit(
            X_train,
            y_train,
        )

        rf_prediction = random_forest.predict(
            X_validation
        )

        rf_metrics = calculate_metrics(
            y_validation.to_numpy(),
            rf_prediction,
        )

        logger.info(
            "%s | %s | Fold %d | "
            "train=%d | validation=%d",
            city,
            horizon_name,
            fold,
            len(train_df),
            len(validation_df),
        )

        logger.info(
            "%s | %s | Fold %d | "
            "Baseline MAE=%.2f | "
            "Ridge MAE=%.2f | "
            "RF MAE=%.2f",
            city,
            horizon_name,
            fold,
            baseline_metrics["MAE"],
            ridge_metrics["MAE"],
            rf_metrics["MAE"],
        )

        for model_name, metrics in [
            (
                "persistence",
                baseline_metrics,
            ),
            (
                "ridge",
                ridge_metrics,
            ),
            (
                "random_forest",
                rf_metrics,
            ),
        ]:
            rows.append(
                {
                    "city": city,
                    "horizon": horizon_name,
                    "fold": fold,
                    "model": model_name,
                    "train_samples": len(train_df),
                    "validation_samples": len(
                        validation_df
                    ),
                    "train_start": str(
                        train_df["hour"].min()
                    ),
                    "train_end": str(
                        train_df["hour"].max()
                    ),
                    "validation_start": str(
                        validation_df["hour"].min()
                    ),
                    "validation_end": str(
                        validation_df["hour"].max()
                    ),
                    "MAE": metrics["MAE"],
                    "RMSE": metrics["RMSE"],
                    "R2": metrics["R2"],
                }
            )

        train_end = validation_end
        fold += 1

    return rows


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    """Run complete 3-city walk-forward validation."""

    logger.info("=" * 72)
    logger.info(
        "PEARLS AQI PREDICTOR"
    )
    logger.info(
        "3-CITY WALK-FORWARD VALIDATION"
    )
    logger.info("=" * 72)

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_dataset()

    feature_columns = get_feature_columns(
        df
    )

    all_results = []

    for horizon_name, target_column in (
        FORECAST_TARGETS.items()
    ):

        logger.info("=" * 72)
        logger.info(
            "HORIZON: %s",
            horizon_name,
        )
        logger.info("=" * 72)

        for city in CITIES:

            city_df = df[
                df["city"] == city
            ].copy()

            city_results = evaluate_city_horizon(
                city_df=city_df,
                feature_columns=feature_columns,
                target_column=target_column,
                horizon_name=horizon_name,
                city=city,
            )

            all_results.extend(
                city_results
            )

    results_df = pd.DataFrame(
        all_results
    )

    results_df.to_csv(
        RESULTS_PATH,
        index=False,
    )

    logger.info(
        "Results saved: %s",
        RESULTS_PATH,
    )

    # ------------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------------

    if results_df.empty:
        raise RuntimeError(
            "Walk-forward validation produced no results."
        )

    summary = (
        results_df
        .groupby(
            [
                "city",
                "horizon",
                "model",
            ]
        )
        .agg(
            mean_mae=("MAE", "mean"),
            median_mae=("MAE", "median"),
            mean_rmse=("RMSE", "mean"),
            mean_r2=("R2", "mean"),
            folds=("fold", "nunique"),
        )
        .reset_index()
    )

    summary_records = summary.to_dict(
        orient="records"
    )

    SUMMARY_PATH.write_text(
        json.dumps(
            {
                "dataset": str(DATA_PATH),
                "feature_count": len(
                    feature_columns
                ),
                "feature_columns": feature_columns,
                "results": summary_records,
            },
            indent=4,
        ),
        encoding="utf-8",
    )

    logger.info(
        "Summary saved: %s",
        SUMMARY_PATH,
    )

    print()
    print("=" * 90)
    print("3-CITY WALK-FORWARD SUMMARY")
    print("=" * 90)

    print(
        summary.to_string(
            index=False
        )
    )

    print("=" * 90)
    logger.info(
        "3-CITY WALK-FORWARD VALIDATION COMPLETED"
    )
    logger.info("=" * 72)


if __name__ == "__main__":
    main()


