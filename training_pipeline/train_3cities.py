"""
3-City AQI Forecasting Training Pipeline.

Cities:
    - Lahore
    - Islamabad
    - Faisalabad

Forecast horizons:
    - 24 hours
    - 48 hours
    - 72 hours

Uses:
    - Chronological train/test split
    - Random Forest
    - Ridge Regression
    - MAE
    - RMSE
    - R2
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
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
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
    / "model_features_3cities.csv"
)

MODEL_DIR = PROJECT_ROOT / "models" / "3cities"
REPORT_DIR = PROJECT_ROOT / "reports" / "model_evaluation" / "3cities"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# CONFIGURATION
# ============================================================================

TEST_SIZE = 0.20
RANDOM_STATE = 42

FORECAST_TARGETS = {
    "24h": "target_24h",
    "48h": "target_48h",
    "72h": "target_72h",
}

EXCLUDED_COLUMNS = {
    "city",
    "hour",
    "coverage_quality",
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

    df = df.sort_values(
        ["city", "hour"]
    ).reset_index(drop=True)

    logger.info("Dataset shape: %s", df.shape)

    logger.info(
        "Cities:\n%s",
        df["city"].value_counts().to_string(),
    )

    return df


# ============================================================================
# FEATURE SELECTION
# ============================================================================

def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return numeric model features."""

    numeric_columns = df.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    feature_columns = [
        column
        for column in numeric_columns
        if column not in EXCLUDED_COLUMNS
    ]

    if not feature_columns:
        raise ValueError("No numeric feature columns found.")

    logger.info(
        "Feature count: %d",
        len(feature_columns),
    )

    return feature_columns


# ============================================================================
# METRICS
# ============================================================================

def calculate_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Calculate regression metrics."""

    return {
        "MAE": float(
            mean_absolute_error(y_true, y_pred)
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
            r2_score(y_true, y_pred)
        ),
    }


# ============================================================================
# MODEL CREATION
# ============================================================================

def create_random_forest() -> Pipeline:
    """Create Random Forest pipeline."""

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=200,
                    max_depth=16,
                    min_samples_leaf=2,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def create_ridge() -> Pipeline:
    """Create Ridge Regression pipeline."""

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                Ridge(alpha=10.0),
            ),
        ]
    )


def create_deep_learning_mlp() -> Pipeline:
    """Create Deep Learning (Multi-Layer Perceptron) Neural Network pipeline."""

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                MLPRegressor(
                    hidden_layer_sizes=(128, 64),
                    activation="relu",
                    max_iter=150,
                    early_stopping=True,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )



# ============================================================================
# TIME-BASED SPLIT
# ============================================================================

def time_split(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split each city chronologically using the last 20% as test."""

    train_parts = []
    test_parts = []

    for city, city_df in data.groupby("city"):
        city_df = city_df.sort_values("hour").reset_index(drop=True)

        split_index = int(
            len(city_df) * (1.0 - TEST_SIZE)
        )

        train_part = city_df.iloc[:split_index]
        test_part = city_df.iloc[split_index:]

        train_parts.append(train_part)
        test_parts.append(test_part)

        logger.info(
            "%s | total=%d | train=%d | test=%d",
            city,
            len(city_df),
            len(train_part),
            len(test_part),
        )

        logger.info(
            "%s | train: %s -> %s",
            city,
            train_part["hour"].min(),
            train_part["hour"].max(),
        )

        logger.info(
            "%s | test:  %s -> %s",
            city,
            test_part["hour"].min(),
            test_part["hour"].max(),
        )

    train_df = pd.concat(
        train_parts,
        ignore_index=True,
    )

    test_df = pd.concat(
        test_parts,
        ignore_index=True,
    )

    return train_df, test_df


# ============================================================================
# CITY-WISE EVALUATION
# ============================================================================

def evaluate_by_city(
    model: Pipeline,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
) -> dict[str, dict[str, float]]:
    """Calculate metrics separately for each city."""

    results = {}

    for city, city_df in test_df.groupby("city"):

        X_city = city_df[feature_columns]
        y_city = city_df[target_column]

        predictions = model.predict(X_city)

        results[city] = calculate_metrics(
            y_city,
            predictions,
        )

    return results


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

    logger.info("")
    logger.info("=" * 72)
    logger.info("FORECAST HORIZON: %s", horizon_name)
    logger.info("=" * 72)

    dataset = df.dropna(
        subset=[target_column]
    ).copy()

    logger.info(
        "Usable samples: %d",
        len(dataset),
    )

    train_df, test_df = time_split(dataset)

    X_train = train_df[feature_columns]
    y_train = train_df[target_column]

    X_test = test_df[feature_columns]
    y_test = test_df[target_column]

    logger.info(
        "Training samples: %d",
        len(X_train),
    )

    logger.info(
        "Testing samples: %d",
        len(X_test),
    )

    # ------------------------------------------------------------------------
    # Random Forest
    # ------------------------------------------------------------------------

    logger.info("Training Random Forest...")

    rf_model = create_random_forest()

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

    rf_city_metrics = evaluate_by_city(
        rf_model,
        test_df,
        feature_columns,
        target_column,
    )

    logger.info(
        "Random Forest | MAE=%.4f | RMSE=%.4f | R2=%.4f",
        rf_metrics["MAE"],
        rf_metrics["RMSE"],
        rf_metrics["R2"],
    )

    # ------------------------------------------------------------------------
    # Ridge
    # ------------------------------------------------------------------------

    logger.info("Training Ridge Regression...")

    ridge_model = create_ridge()

    ridge_model.fit(
        X_train,
        y_train,
    )

    ridge_predictions = ridge_model.predict(
        X_test
    )

    ridge_metrics = calculate_metrics(
        y_test,
        ridge_predictions,
    )

    ridge_city_metrics = evaluate_by_city(
        ridge_model,
        test_df,
        feature_columns,
        target_column,
    )

    logger.info(
        "Ridge | MAE=%.4f | RMSE=%.4f | R2=%.4f",
        ridge_metrics["MAE"],
        ridge_metrics["RMSE"],
        ridge_metrics["R2"],
    )

    # ------------------------------------------------------------------------
    # Deep Learning (MLP)
    # ------------------------------------------------------------------------

    logger.info("Training Deep Learning (MLP Neural Network)...")

    mlp_model = create_deep_learning_mlp()

    mlp_model.fit(
        X_train,
        y_train,
    )

    mlp_predictions = mlp_model.predict(
        X_test
    )

    mlp_metrics = calculate_metrics(
        y_test,
        mlp_predictions,
    )

    mlp_city_metrics = evaluate_by_city(
        mlp_model,
        test_df,
        feature_columns,
        target_column,
    )

    logger.info(
        "Deep Learning (MLP) | MAE=%.4f | RMSE=%.4f | R2=%.4f",
        mlp_metrics["MAE"],
        mlp_metrics["RMSE"],
        mlp_metrics["R2"],
    )

    # ------------------------------------------------------------------------
    # Select best model
    # ------------------------------------------------------------------------

    models_dict = {
        "RandomForest": (rf_model, rf_metrics, rf_city_metrics),
        "Ridge": (ridge_model, ridge_metrics, ridge_city_metrics),
        "DeepLearning_MLP": (mlp_model, mlp_metrics, mlp_city_metrics),
    }

    best_model_name = min(models_dict.keys(), key=lambda k: models_dict[k][1]["MAE"])
    best_model, best_metrics, best_city_metrics = models_dict[best_model_name]

    logger.info(
        "BEST MODEL FOR %s: %s (MAE=%.4f)",
        horizon_name,
        best_model_name,
        best_metrics["MAE"],
    )

    # ------------------------------------------------------------------------
    # Save models
    # ------------------------------------------------------------------------

    rf_path = MODEL_DIR / f"random_forest_{horizon_name}.joblib"
    ridge_path = MODEL_DIR / f"ridge_{horizon_name}.joblib"
    mlp_path = MODEL_DIR / f"deep_learning_{horizon_name}.joblib"
    best_path = MODEL_DIR / f"best_model_{horizon_name}.joblib"

    joblib.dump(rf_model, rf_path)
    joblib.dump(ridge_model, ridge_path)
    joblib.dump(mlp_model, mlp_path)
    joblib.dump(best_model, best_path)

    return {
        "horizon": horizon_name,
        "target": target_column,
        "samples": len(dataset),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "random_forest": rf_metrics,
        "random_forest_by_city": rf_city_metrics,
        "ridge": ridge_metrics,
        "ridge_by_city": ridge_city_metrics,
        "deep_learning_mlp": mlp_metrics,
        "deep_learning_mlp_by_city": mlp_city_metrics,
        "best_model": best_model_name,
        "best_metrics": best_metrics,
        "best_by_city": best_city_metrics,
        "random_forest_path": str(rf_path),
        "ridge_path": str(ridge_path),
        "deep_learning_mlp_path": str(mlp_path),
        "best_model_path": str(best_path),
    }


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    """Execute the complete 3-city training pipeline."""

    logger.info("=" * 72)
    logger.info("PEARLS AQI PREDICTOR")
    logger.info("3-CITY MODEL TRAINING")
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
    # Save report
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
        / "training_report_3cities.json"
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
    logger.info("3-CITY MODEL TRAINING COMPLETED")
    logger.info("=" * 72)


if __name__ == "__main__":
    main()