"""
Pearls AQI Predictor
====================

Walk-forward validation for daily AQI forecasting.

Uses:
    data/processed/model_features_3cities.csv

Forecast horizons:
    Day +1
    Day +2
    Day +3

Evaluation:
    - Expanding-window walk-forward validation
    - Persistence baseline
    - Ridge
    - Random Forest
    - Extra Trees
    - HistGradientBoosting

No random shuffle.
No future target leakage.
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


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_features_3cities.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "model_evaluation"
)

RESULTS_FILE = (
    OUTPUT_DIR
    / "aqi_walk_forward_results.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "aqi_walk_forward_summary.json"
)


TARGETS = {
    "day_1": "target_24h",
    "day_2": "target_48h",
    "day_3": "target_72h",
}

FEATURE_EXCLUDE = {
    "hour",
    "city",
    "target_24h",
    "target_48h",
    "target_72h",
}

RANDOM_STATE = 42

MIN_TRAIN_SIZE = 180

VALIDATION_BLOCK_SIZE = 20

STEP_SIZE = 20


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


def load_dataset() -> pd.DataFrame:
    """Load and validate reduced AQI features."""

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Reduced AQI feature file not found: {INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE
    )

    if df.empty:
        raise ValueError(
            "Reduced AQI dataset is empty."
        )

    df["hour"] = pd.to_datetime(
        df["hour"],
        errors="coerce",
    ).dt.normalize()

    if df["hour"].isna().any():
        raise ValueError(
            "Invalid dates detected."
        )

    df = (
        df
        .sort_values("hour")
        .drop_duplicates("hour")
        .reset_index(drop=True)
    )

    logger.info(
        "Loaded AQI dataset: rows=%d | columns=%d",
        len(df),
        len(df.columns),
    )

    return df


def get_features(
    df: pd.DataFrame,
) -> list[str]:
    """Return numeric predictor columns."""

    features = [
        column
        for column in df.columns
        if column not in FEATURE_EXCLUDE
        and pd.api.types.is_numeric_dtype(
            df[column]
        )
    ]

    if not features:
        raise ValueError(
            "No numeric predictors found."
        )

    logger.info(
        "Predictor count: %d",
        len(features),
    )

    return features


def build_models() -> dict[str, Pipeline]:
    """Build candidate forecasting models."""

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
                        n_estimators=300,
                        max_depth=8,
                        min_samples_leaf=3,
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
                        n_estimators=300,
                        max_depth=8,
                        min_samples_leaf=3,
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
                        max_iter=200,
                        learning_rate=0.05,
                        max_leaf_nodes=12,
                        min_samples_leaf=8,
                        l2_regularization=2.0,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def calculate_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
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


def walk_forward_evaluate(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    horizon_name: str,
) -> list[dict]:
    """
    Perform expanding-window walk-forward evaluation.

    Each validation block occurs strictly after its training block.
    """

    usable = df[
        feature_columns
        + [
            "hour",
            target_column,
        ]
    ].copy()

    usable = usable.dropna(
        subset=[
            target_column
        ]
    ).reset_index(drop=True)

    if len(usable) < (
        MIN_TRAIN_SIZE
        + VALIDATION_BLOCK_SIZE
    ):
        raise ValueError(
            f"Not enough observations for {horizon_name}."
        )

    results: list[dict] = []

    models = build_models()

    fold_number = 0

    train_end = MIN_TRAIN_SIZE

    while (
        train_end
        + VALIDATION_BLOCK_SIZE
        <= len(usable)
    ):

        fold_number += 1

        train = usable.iloc[
            :train_end
        ]

        validation = usable.iloc[
            train_end:
            train_end
            + VALIDATION_BLOCK_SIZE
        ]

        logger.info(
            "%s | Fold %d | train=%d | validation=%d | "
            "%s â†’ %s",
            horizon_name,
            fold_number,
            len(train),
            len(validation),
            validation["hour"].min(),
            validation["hour"].max(),
        )

        X_train = train[
            feature_columns
        ]

        y_train = train[
            target_column
        ]

        X_validation = validation[
            feature_columns
        ]

        y_validation = validation[
            target_column
        ]

        # --------------------------------------------------------------
        # Persistence baseline
        # --------------------------------------------------------------

        baseline_prediction = validation[
            "aqi"
        ].to_numpy()

        baseline_metrics = calculate_metrics(
            y_validation,
            baseline_prediction,
        )

        results.append(
            {
                "horizon": horizon_name,
                "fold": fold_number,
                "model": "persistence",
                "evaluation_rows": len(
                    validation
                ),
                **baseline_metrics,
            }
        )

        # --------------------------------------------------------------
        # Machine learning
        # --------------------------------------------------------------

        for model_name, model in models.items():

            model.fit(
                X_train,
                y_train,
            )

            prediction = model.predict(
                X_validation
            )

            metrics = calculate_metrics(
                y_validation,
                prediction,
            )

            results.append(
                {
                    "horizon": horizon_name,
                    "fold": fold_number,
                    "model": model_name,
                    "evaluation_rows": len(
                        validation
                    ),
                    **metrics,
                }
            )

        train_end += STEP_SIZE

    return results


def create_summary(
    results: pd.DataFrame,
) -> dict:
    """Create aggregate walk-forward summary."""

    summary = {}

    for horizon in results[
        "horizon"
    ].unique():

        horizon_data = results[
            results["horizon"] == horizon
        ]

        model_summary = (
            horizon_data
            .groupby("model")
            .agg(
                mean_mae=("mae", "mean"),
                median_mae=("mae", "median"),
                mean_rmse=("rmse", "mean"),
                mean_r2=("r2", "mean"),
                folds=("fold", "nunique"),
            )
            .reset_index()
            .sort_values("mean_mae")
        )

        summary[horizon] = (
            model_summary
            .to_dict(
                orient="records"
            )
        )

    return summary


def main() -> None:
    """Run complete walk-forward AQI evaluation."""

    logger.info("=" * 72)
    logger.info(
        "PEARLS AQI PREDICTOR"
    )
    logger.info(
        "WALK-FORWARD AQI VALIDATION"
    )
    logger.info("=" * 72)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_dataset()

    feature_columns = get_features(
        df
    )

    all_results: list[dict] = []

    for horizon_name, target_column in (
        TARGETS.items()
    ):

        logger.info("=" * 72)
        logger.info(
            "HORIZON: %s",
            horizon_name,
        )

        results = walk_forward_evaluate(
            df=df,
            feature_columns=feature_columns,
            target_column=target_column,
            horizon_name=horizon_name,
        )

        all_results.extend(
            results
        )

    results_df = pd.DataFrame(
        all_results
    )

    if results_df.empty:
        raise RuntimeError(
            "No walk-forward results generated."
        )

    results_df.to_csv(
        RESULTS_FILE,
        index=False,
    )

    summary = create_summary(
        results_df
    )

    SUMMARY_FILE.write_text(
        json.dumps(
            summary,
            indent=4,
        ),
        encoding="utf-8",
    )

    logger.info(
        "Results saved: %s",
        RESULTS_FILE,
    )

    logger.info(
        "Summary saved: %s",
        SUMMARY_FILE,
    )

    print(
        "\nWALK-FORWARD SUMMARY"
    )

    print(
        results_df
        .groupby(
            [
                "horizon",
                "model",
            ]
        )
        .agg(
            mean_mae=("mae", "mean"),
            median_mae=("mae", "median"),
            mean_rmse=("rmse", "mean"),
            mean_r2=("r2", "mean"),
            folds=("fold", "nunique"),
        )
        .reset_index()
        .sort_values(
            [
                "horizon",
                "mean_mae",
            ]
        )
        .to_string(index=False)
    )

    logger.info("=" * 72)
    logger.info(
        "WALK-FORWARD VALIDATION COMPLETED."
    )
    logger.info("=" * 72)


if __name__ == "__main__":
    main()





