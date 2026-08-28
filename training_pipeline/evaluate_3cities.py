"""
3-City Model Evaluation
Pearls AQI Predictor

Evaluates the already-trained 3-city models separately for:
    - Lahore
    - Islamabad
    - Faisalabad

Forecast horizons:
    - 24h
    - 48h
    - 72h

Uses the same city-wise chronological split methodology
used during training.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
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
    / "model_features_3cities.csv"
)

MODEL_DIR = PROJECT_ROOT / "models" / "3cities"

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "model_evaluation"
    / "3cities"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================================
# CONFIGURATION
# ============================================================================

TEST_RATIO = 0.20

TARGETS = {
    "24h": "target_24h",
    "48h": "target_48h",
    "72h": "target_72h",
}

CITIES = [
    "Lahore",
    "Islamabad",
    "Faisalabad",
]

EXCLUDED_COLUMNS = {
    "city",
    "hour",
    "coverage_quality",
    "is_missing_hour",
    "target_24h",
    "target_48h",
    "target_72h",
}


# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================================
# DATA LOADING
# ============================================================================


def load_dataset() -> pd.DataFrame:
    """Load and validate the 3-city feature dataset."""

    logger.info("Loading dataset: %s", DATA_PATH)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(
        DATA_PATH,
        parse_dates=["hour"],
    )

    df["hour"] = pd.to_datetime(
        df["hour"],
        utc=True,
        errors="coerce",
    )

    df = df.dropna(
        subset=["hour", "city"]
    ).copy()

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
# FEATURE SELECTION
# ============================================================================


def get_feature_columns(
    df: pd.DataFrame,
) -> list[str]:
    """Return numeric model feature columns."""

    features = [
        column
        for column in df.columns
        if column not in EXCLUDED_COLUMNS
    ]

    if not features:
        raise ValueError(
            "No feature columns found."
        )

    return features


# ============================================================================
# METRICS
# ============================================================================


def calculate_metrics(
    y_true: pd.Series,
    predictions: np.ndarray,
) -> dict[str, float]:
    """Calculate regression metrics."""

    y_true_array = np.asarray(
        y_true,
        dtype=float,
    )

    predictions_array = np.asarray(
        predictions,
        dtype=float,
    )

    return {
        "mae": float(
            mean_absolute_error(
                y_true_array,
                predictions_array,
            )
        ),
        "rmse": float(
            np.sqrt(
                mean_squared_error(
                    y_true_array,
                    predictions_array,
                )
            )
        ),
        "r2": float(
            r2_score(
                y_true_array,
                predictions_array,
            )
        ),
    }


# ============================================================================
# CITY-WISE SPLIT
# ============================================================================


def create_city_split(
    city_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create chronological 80/20 split for one city.

    The latest 20% is used as test data.
    """

    city_df = city_df.sort_values(
        "hour"
    ).reset_index(drop=True)

    split_index = int(
        len(city_df) * (1 - TEST_RATIO)
    )

    train = city_df.iloc[
        :split_index
    ].copy()

    test = city_df.iloc[
        split_index:
    ].copy()

    return train, test


# ============================================================================
# EVALUATION
# ============================================================================


def evaluate_horizon(
    df: pd.DataFrame,
    feature_columns: list[str],
    horizon_name: str,
    target_column: str,
) -> list[dict]:
    """Evaluate one forecast horizon city by city."""

    logger.info("")
    logger.info("=" * 72)
    logger.info(
        "EVALUATING %s FORECAST",
        horizon_name,
    )
    logger.info("=" * 72)

    model_path = (
        MODEL_DIR
        / f"best_model_{horizon_name}.joblib"
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    logger.info(
        "Loading model: %s",
        model_path.name,
    )

    model = joblib.load(model_path)

    results = []

    for city in CITIES:

        city_df = df[
            df["city"] == city
        ].copy()

        city_df = city_df.dropna(
            subset=[target_column]
        ).reset_index(drop=True)

        if city_df.empty:
            logger.warning(
                "%s | no usable samples",
                city,
            )
            continue

        train, test = create_city_split(
            city_df
        )

        logger.info(
            "%s | total=%d | train=%d | test=%d",
            city,
            len(city_df),
            len(train),
            len(test),
        )

        logger.info(
            "%s | test: %s -> %s",
            city,
            test["hour"].min(),
            test["hour"].max(),
        )

        X_test = test[
            feature_columns
        ].copy()

        y_test = test[
            target_column
        ].copy()

        # Numeric conversion
        for column in feature_columns:
            X_test[column] = pd.to_numeric(
                X_test[column],
                errors="coerce",
            )

        X_test = X_test.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        # Same median-imputation strategy as training
        train_features = train[
            feature_columns
        ].copy()

        for column in feature_columns:
            train_features[column] = pd.to_numeric(
                train_features[column],
                errors="coerce",
            )

        train_features = train_features.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        medians = train_features.median()

        X_test = X_test.fillna(
            medians
        )

        predictions = model.predict(
            X_test
        )

        metrics = calculate_metrics(
            y_test,
            predictions,
        )

        logger.info(
            "%s | MAE=%.4f | RMSE=%.4f | R2=%.4f",
            city,
            metrics["mae"],
            metrics["rmse"],
            metrics["r2"],
        )

        results.append(
            {
                "city": city,
                "horizon": horizon_name,
                "target": target_column,
                "total_samples": int(len(city_df)),
                "train_samples": int(len(train)),
                "test_samples": int(len(test)),
                "test_start": str(
                    test["hour"].min()
                ),
                "test_end": str(
                    test["hour"].max()
                ),
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "r2": metrics["r2"],
            }
        )

    return results


# ============================================================================
# MAIN
# ============================================================================


def main() -> None:
    """Run complete 3-city evaluation."""

    logger.info("=" * 72)
    logger.info(
        "PEARLS AQI PREDICTOR"
    )
    logger.info(
        "3-CITY MODEL EVALUATION"
    )
    logger.info("=" * 72)

    df = load_dataset()

    feature_columns = get_feature_columns(
        df
    )

    logger.info(
        "Feature count: %d",
        len(feature_columns),
    )

    logger.info(
        "Features:\n%s",
        feature_columns,
    )

    all_results = []

    for horizon_name, target_column in TARGETS.items():

        horizon_results = evaluate_horizon(
            df=df,
            feature_columns=feature_columns,
            horizon_name=horizon_name,
            target_column=target_column,
        )

        all_results.extend(
            horizon_results
        )

    if not all_results:
        raise RuntimeError(
            "No evaluation results generated."
        )

    results_df = pd.DataFrame(
        all_results
    )

    results_df = results_df.sort_values(
        ["horizon", "city"]
    )

    # ------------------------------------------------------------------------
    # Save CSV
    # ------------------------------------------------------------------------

    csv_path = (
        REPORT_DIR
        / "city_wise_evaluation_3cities.csv"
    )

    results_df.to_csv(
        csv_path,
        index=False,
    )

    # ------------------------------------------------------------------------
    # Save JSON
    # ------------------------------------------------------------------------

    json_path = (
        REPORT_DIR
        / "city_wise_evaluation_3cities.json"
    )

    report = {
        "dataset": str(DATA_PATH),
        "model_directory": str(MODEL_DIR),
        "test_ratio": TEST_RATIO,
        "cities": CITIES,
        "results": all_results,
    }

    json_path.write_text(
        json.dumps(
            report,
            indent=4,
        ),
        encoding="utf-8",
    )

    # ------------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------------

    logger.info("")
    logger.info("=" * 72)
    logger.info(
        "CITY-WISE EVALUATION SUMMARY"
    )
    logger.info("=" * 72)

    print()
    print(
        results_df[
            [
                "city",
                "horizon",
                "test_samples",
                "mae",
                "rmse",
                "r2",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    logger.info("")
    logger.info(
        "CSV saved: %s",
        csv_path,
    )

    logger.info(
        "JSON saved: %s",
        json_path,
    )

    logger.info("=" * 72)
    logger.info(
        "3-CITY EVALUATION COMPLETED SUCCESSFULLY"
    )
    logger.info("=" * 72)


if __name__ == "__main__":
    main()


