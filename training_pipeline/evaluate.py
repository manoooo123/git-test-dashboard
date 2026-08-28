from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = PROJECT_ROOT / "data" / "processed" / "model_features.csv"
MODELS_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports" / "model_evaluation"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS = [24, 48, 72]

TEST_RATIO = 0.20

TARGET_TEMPLATE = "target_pm2_5_{h}h"

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
)

logger = logging.getLogger(__name__)


# ============================================================
# HELPERS
# ============================================================

def load_dataset() -> pd.DataFrame:
    """Load production feature dataset."""

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Feature dataset not found:\n{DATA_FILE}"
        )

    logger.info("Loading feature dataset: %s", DATA_FILE)

    df = pd.read_csv(DATA_FILE)

    if "timestamp" not in df.columns:
        raise ValueError(
            "Required column 'timestamp' is missing."
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
        errors="coerce",
    )

    df = df.dropna(
        subset=["timestamp"]
    ).sort_values(
        "timestamp"
    ).reset_index(drop=True)

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


def get_feature_columns(
    df: pd.DataFrame,
    target_column: str,
) -> list[str]:
    """
    Detect model input columns.

    Excludes identifiers, timestamp and all future targets.
    """

    excluded = {
        "timestamp",
        "city",
        "location_id",
        "sensor_id",
        "target_pm2_5_24h",
        "target_pm2_5_48h",
        "target_pm2_5_72h",
    }

    features = [
        column
        for column in df.columns
        if column not in excluded
        and column != target_column
    ]

    if not features:
        raise ValueError(
            "No feature columns found."
        )

    return features


def prepare_data(
    df: pd.DataFrame,
    target_column: str,
    test_ratio: float = TEST_RATIO,
):
    """
    Prepare time-aware train/test data.

    The test set is always the latest portion of the dataset.
    """

    data = df[
        ["timestamp", target_column]
        + get_feature_columns(df, target_column)
    ].copy()

    data = data.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    data = data.dropna(
        subset=[target_column]
    ).reset_index(drop=True)

    feature_columns = get_feature_columns(
        data,
        target_column,
    )

    # Numeric conversion
    for column in feature_columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    data[feature_columns] = data[
        feature_columns
    ].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # Median imputation for remaining missing features
    data[feature_columns] = data[
        feature_columns
    ].fillna(
        data[feature_columns].median()
    )

    split_index = int(
        len(data) * (1 - test_ratio)
    )

    train = data.iloc[:split_index].copy()
    test = data.iloc[split_index:].copy()

    X_train = train[feature_columns]
    y_train = train[target_column]

    X_test = test[feature_columns]
    y_test = test[target_column]

    return (
        X_train,
        y_train,
        X_test,
        y_test,
        train,
        test,
        feature_columns,
    )


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """Calculate regression metrics."""

    predictions = model.predict(X_test)

    predictions = np.asarray(
        predictions,
        dtype=float,
    )

    y_true = np.asarray(
        y_test,
        dtype=float,
    )

    mae = mean_absolute_error(
        y_true,
        predictions,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            predictions,
        )
    )

    r2 = r2_score(
        y_true,
        predictions,
    )

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
    }


# ============================================================
# MAIN EVALUATION
# ============================================================

def main() -> None:

    logger.info("=" * 72)
    logger.info("PEARLS AQI PREDICTOR")
    logger.info("PRODUCTION MODEL EVALUATION")
    logger.info("=" * 72)

    df = load_dataset()

    all_results = []

    for horizon in HORIZONS:

        logger.info("-" * 72)
        logger.info(
            "Evaluating forecast horizon: %sh",
            horizon,
        )

        target_column = TARGET_TEMPLATE.format(
            h=horizon
        )

        if target_column not in df.columns:
            logger.error(
                "Target missing: %s",
                target_column,
            )
            continue

        (
            X_train,
            y_train,
            X_test,
            y_test,
            train,
            test,
            feature_columns,
        ) = prepare_data(
            df,
            target_column,
        )

        logger.info(
            "%sh valid samples: %d",
            horizon,
            len(X_train) + len(X_test),
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
            train["timestamp"].min(),
            train["timestamp"].max(),
        )

        logger.info(
            "Test period: %s -> %s",
            test["timestamp"].min(),
            test["timestamp"].max(),
        )

        model_files = {
            "ridge": MODELS_DIR / f"ridge_{horizon}h.joblib",
            "random_forest": MODELS_DIR
            / f"random_forest_{horizon}h.joblib",
            "best_model": MODELS_DIR
            / f"best_model_{horizon}h.joblib",
        }

        for model_name, model_path in model_files.items():

            if not model_path.exists():
                logger.warning(
                    "Model not found: %s",
                    model_path,
                )
                continue

            logger.info(
                "Loading %s: %s",
                model_name,
                model_path.name,
            )

            model = joblib.load(
                model_path
            )

            metrics = evaluate_model(
                model,
                X_test,
                y_test,
            )

            logger.info(
                "%s | MAE=%.4f | RMSE=%.4f | R2=%.4f",
                model_name,
                metrics["mae"],
                metrics["rmse"],
                metrics["r2"],
            )

            all_results.append(
                {
                    "horizon_hours": horizon,
                    "model": model_name,
                    "mae": metrics["mae"],
                    "rmse": metrics["rmse"],
                    "r2": metrics["r2"],
                    "training_samples": len(X_train),
                    "testing_samples": len(X_test),
                    "feature_count": len(feature_columns),
                    "train_start": str(
                        train["timestamp"].min()
                    ),
                    "train_end": str(
                        train["timestamp"].max()
                    ),
                    "test_start": str(
                        test["timestamp"].min()
                    ),
                    "test_end": str(
                        test["timestamp"].max()
                    ),
                }
            )

    if not all_results:
        raise RuntimeError(
            "No models were successfully evaluated."
        )

    results_df = pd.DataFrame(
        all_results
    )

    results_df = results_df.sort_values(
        ["horizon_hours", "r2"],
        ascending=[True, False],
    )

    output_csv = (
        REPORT_DIR
        / "production_evaluation.csv"
    )

    results_df.to_csv(
        output_csv,
        index=False,
    )

    # Best model per horizon
    best_models = (
        results_df.sort_values(
            "r2",
            ascending=False,
        )
        .groupby(
            "horizon_hours",
            as_index=False,
        )
        .first()
    )

    best_output = (
        REPORT_DIR
        / "best_models.json"
    )

    with open(
        best_output,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            best_models.to_dict(
                orient="records"
            ),
            file,
            indent=4,
        )

    logger.info("=" * 72)
    logger.info(
        "BEST MODEL RESULTS"
    )
    logger.info("=" * 72)

    for _, row in best_models.iterrows():

        logger.info(
            "%sh | %s | MAE=%.4f | RMSE=%.4f | R2=%.4f",
            int(row["horizon_hours"]),
            row["model"],
            row["mae"],
            row["rmse"],
            row["r2"],
        )

    logger.info(
        "Evaluation CSV saved: %s",
        output_csv,
    )

    logger.info(
        "Best-model report saved: %s",
        best_output,
    )

    logger.info("=" * 72)
    logger.info(
        "PRODUCTION MODEL EVALUATION COMPLETED."
    )
    logger.info("=" * 72)


if __name__ == "__main__":
    main()