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
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "aqi_reduced_features.parquet"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "model_evaluation"
)

RESULTS_FILE = OUTPUT_DIR / "aqi_hybrid_results.csv"
SUMMARY_FILE = OUTPUT_DIR / "aqi_hybrid_summary.json"

TARGETS = {
    "day_1": "target_aqi_day_1",
    "day_2": "target_aqi_day_2",
    "day_3": "target_aqi_day_3",
}

MIN_TRAIN_SIZE = 180
VALIDATION_BLOCK_SIZE = 20
STEP_SIZE = 20
RANDOM_STATE = 42


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)


def load_dataset() -> pd.DataFrame:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input dataset not found: {INPUT_FILE}"
        )

    df = pd.read_parquet(INPUT_FILE)

    if df.empty:
        raise ValueError("AQI dataset is empty.")

    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    df = df.loc[
        :,
        ~df.columns.duplicated(keep="first"),
    ].copy()

    required = {
        "date",
        "aqi",
        "target_aqi_day_1",
        "target_aqi_day_2",
        "target_aqi_day_3",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            "Missing columns: "
            + ", ".join(sorted(missing))
        )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    ).dt.normalize()

    df["aqi"] = pd.to_numeric(
        df["aqi"],
        errors="coerce",
    )

    for target in TARGETS.values():
        df[target] = pd.to_numeric(
            df[target],
            errors="coerce",
        )

    df = (
        df.dropna(subset=["date", "aqi"])
        .sort_values("date")
        .drop_duplicates("date")
        .reset_index(drop=True)
    )

    if list(df.columns).count("aqi") != 1:
        raise ValueError("AQI column is not unique.")

    logger.info(
        "Loaded dataset: rows=%d | columns=%d",
        len(df),
        len(df.columns),
    )

    logger.info(
        "Date range: %s -> %s",
        df["date"].min(),
        df["date"].max(),
    )

    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {
        "date",
        "city",
        "target_aqi_day_1",
        "target_aqi_day_2",
        "target_aqi_day_3",
    }

    features = []

    for column in df.columns:
        if column in excluded:
            continue

        if not pd.api.types.is_numeric_dtype(df[column]):
            continue

        if column not in features:
            features.append(column)

    if "aqi" not in features:
        raise ValueError(
            "Current AQI must be a predictor."
        )

    logger.info(
        "Predictor count: %d",
        len(features),
    )

    return features


def build_models() -> dict[str, Pipeline]:
    return {
        "ridge_residual": Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(strategy="median"),
                ),
                (
                    "model",
                    Ridge(alpha=10.0),
                ),
            ]
        ),
        "random_forest_residual": Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(strategy="median"),
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
        "extra_trees_residual": Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(strategy="median"),
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
        "hist_gradient_residual": Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(strategy="median"),
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


def metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[float, float]:

    y_true = np.asarray(
        y_true,
        dtype=float,
    ).reshape(-1)

    y_pred = np.asarray(
        y_pred,
        dtype=float,
    ).reshape(-1)

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

    return float(mae), float(rmse)


def evaluate_horizon(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    horizon: str,
) -> list[dict]:

    columns = list(
        dict.fromkeys(
            [
                "date",
                "aqi",
                target_column,
                *feature_columns,
            ]
        )
    )

    data = (
        df[columns]
        .dropna(
            subset=[
                "aqi",
                target_column,
            ]
        )
        .sort_values("date")
        .reset_index(drop=True)
    )

    minimum = (
        MIN_TRAIN_SIZE
        + VALIDATION_BLOCK_SIZE
    )

    if len(data) < minimum:
        raise ValueError(
            f"Not enough observations for {horizon}."
        )

    models = build_models()

    results = []

    train_end = MIN_TRAIN_SIZE
    fold = 0

    while (
        train_end + VALIDATION_BLOCK_SIZE
        <= len(data)
    ):

        fold += 1

        train = (
            data.iloc[:train_end]
            .copy()
            .reset_index(drop=True)
        )

        validation = (
            data.iloc[
                train_end:
                train_end + VALIDATION_BLOCK_SIZE
            ]
            .copy()
            .reset_index(drop=True)
        )

        logger.info(
            "%s | Fold %d | train=%d | validation=%d | %s -> %s",
            horizon,
            fold,
            len(train),
            len(validation),
            validation["date"].min(),
            validation["date"].max(),
        )

        X_train = train[feature_columns].copy()
        X_validation = validation[feature_columns].copy()

        X_train = X_train.loc[
            :,
            ~X_train.columns.duplicated(keep="first"),
        ].copy()

        X_validation = X_validation.loc[
            :,
            ~X_validation.columns.duplicated(keep="first"),
        ].copy()

        X_validation = X_validation[
            X_train.columns
        ]

        current_train = train[
            "aqi"
        ].to_numpy(
            dtype=float
        )

        future_train = train[
            target_column
        ].to_numpy(
            dtype=float
        )

        current_validation = validation[
            "aqi"
        ].to_numpy(
            dtype=float
        )

        future_validation = validation[
            target_column
        ].to_numpy(
            dtype=float
        )

        residual_train = (
            future_train - current_train
        )

        baseline_prediction = (
            current_validation.copy()
        )

        baseline_mae, baseline_rmse = metrics(
            future_validation,
            baseline_prediction,
        )

        results.append(
            {
                "horizon": horizon,
                "fold": fold,
                "model": "persistence",
                "evaluation_rows": len(validation),
                "mae": baseline_mae,
                "rmse": baseline_rmse,
            }
        )

        for model_name, model in models.items():

            model.fit(
                X_train,
                residual_train,
            )

            predicted_change = model.predict(
                X_validation
            )

            prediction = (
                current_validation
                + predicted_change
            )

            prediction = np.maximum(
                prediction,
                0.0,
            )

            model_mae, model_rmse = metrics(
                future_validation,
                prediction,
            )

            results.append(
                {
                    "horizon": horizon,
                    "fold": fold,
                    "model": model_name,
                    "evaluation_rows": len(validation),
                    "mae": model_mae,
                    "rmse": model_rmse,
                }
            )

        train_end += STEP_SIZE

    return results


def create_summary(
    results_df: pd.DataFrame,
) -> dict:

    summary = {}

    for horizon in results_df["horizon"].unique():

        current = results_df[
            results_df["horizon"] == horizon
        ]

        aggregate = (
            current
            .groupby("model")
            .agg(
                mean_mae=("mae", "mean"),
                median_mae=("mae", "median"),
                mean_rmse=("rmse", "mean"),
                folds=("fold", "nunique"),
            )
            .reset_index()
        )

        baseline = aggregate[
            aggregate["model"] == "persistence"
        ]

        if baseline.empty:
            summary[horizon] = aggregate.to_dict(
                orient="records"
            )
            continue

        baseline_mae = float(
            baseline.iloc[0]["mean_mae"]
        )

        aggregate[
            "mae_improvement_percent"
        ] = (
            (
                baseline_mae
                - aggregate["mean_mae"]
            )
            / baseline_mae
            * 100.0
        )

        aggregate[
            "beats_persistence"
        ] = (
            aggregate["mean_mae"]
            < baseline_mae
        )

        aggregate = aggregate.sort_values(
            "mean_mae"
        )

        summary[horizon] = aggregate.to_dict(
            orient="records"
        )

    return summary


def main() -> None:

    logger.info("=" * 72)
    logger.info("PEARLS AQI PREDICTOR")
    logger.info("HYBRID AQI FORECASTING")
    logger.info("=" * 72)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_dataset()

    feature_columns = get_feature_columns(
        df
    )

    all_results = []

    for horizon, target_column in TARGETS.items():

        logger.info(
            "Starting horizon: %s",
            horizon,
        )

        horizon_results = evaluate_horizon(
            df=df,
            feature_columns=feature_columns,
            target_column=target_column,
            horizon=horizon,
        )

        all_results.extend(
            horizon_results
        )

    results_df = pd.DataFrame(
        all_results
    )

    if results_df.empty:
        raise RuntimeError(
            "No forecasting results generated."
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

    print(
        "\nHYBRID WALK-FORWARD SUMMARY"
    )

    for horizon, rows in summary.items():

        print(
            f"\n{horizon.upper()}"
        )

        print(
            pd.DataFrame(rows).to_string(
                index=False
            )
        )

    logger.info(
        "Results saved: %s",
        RESULTS_FILE,
    )

    logger.info(
        "Summary saved: %s",
        SUMMARY_FILE,
    )

    logger.info(
        "HYBRID AQI FORECASTING COMPLETED."
    )


if __name__ == "__main__":
    main()
