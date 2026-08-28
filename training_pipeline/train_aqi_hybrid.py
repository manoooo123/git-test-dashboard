from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "aqi_reduced_features.parquet"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "forecasting"
)

REGISTRY_FILE = (
    MODEL_DIR
    / "aqi_model_registry.json"
)

MODEL_VERSION = "aqi-hybrid-rf-v1.0.1"

TARGETS = {
    "day_1": "target_aqi_day_1",
    "day_2": "target_aqi_day_2",
    "day_3": "target_aqi_day_3",
}

RANDOM_STATE = 42


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)


def load_dataset() -> pd.DataFrame:
    """Load and clean the reduced AQI dataset."""

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_FILE}"
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
            "Missing required columns: "
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
        df
        .dropna(subset=["date", "aqi"])
        .sort_values("date")
        .drop_duplicates("date")
        .reset_index(drop=True)
    )

    if df.columns.duplicated().any():
        raise ValueError(
            "Duplicate column names remain."
        )

    if list(df.columns).count("aqi") != 1:
        raise ValueError(
            "AQI column is not unique."
        )

    logger.info(
        "Dataset rows: %d",
        len(df),
    )

    logger.info(
        "Date range: %s -> %s",
        df["date"].min(),
        df["date"].max(),
    )

    return df


def get_feature_columns(
    df: pd.DataFrame,
) -> list[str]:
    """Return unique numeric predictor columns."""

    excluded = {
        "date",
        "city",
        "target_aqi_day_1",
        "target_aqi_day_2",
        "target_aqi_day_3",
    }

    features: list[str] = []

    for column in df.columns:
        if column in excluded:
            continue

        if not pd.api.types.is_numeric_dtype(
            df[column]
        ):
            continue

        if column not in features:
            features.append(column)

    if "aqi" not in features:
        raise ValueError(
            "Current AQI is missing from predictors."
        )

    return features


def build_model() -> Pipeline:
    """Build the validated Random Forest residual model."""

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
                    n_estimators=500,
                    max_depth=8,
                    min_samples_leaf=3,
                    max_features="sqrt",
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def train_horizon(
    df: pd.DataFrame,
    feature_columns: list[str],
    horizon: str,
    target_column: str,
) -> dict:
    """Train one final hybrid residual model."""

    # IMPORTANT:
    # 'aqi' is already inside feature_columns.
    # Therefore we do NOT add 'aqi' separately here.
    selected_columns = list(
        dict.fromkeys(
            [
                "date",
                target_column,
                *feature_columns,
            ]
        )
    )

    training = (
        df[selected_columns]
        .copy()
        .dropna(
            subset=[
                "aqi",
                target_column,
            ]
        )
        .sort_values("date")
        .reset_index(drop=True)
    )

    if training.empty:
        raise ValueError(
            f"No training rows available for {horizon}."
        )

    X = training[
        feature_columns
    ].copy()

    X = X.loc[
        :,
        ~X.columns.duplicated(keep="first"),
    ].copy()

    y_current = training[
        "aqi"
    ].to_numpy(
        dtype=float
    ).reshape(-1)

    y_future = training[
        target_column
    ].to_numpy(
        dtype=float
    ).reshape(-1)

    if y_current.shape != y_future.shape:
        raise ValueError(
            f"Shape mismatch for {horizon}: "
            f"current={y_current.shape}, "
            f"future={y_future.shape}"
        )

    # Residual = future AQI - current AQI
    y_residual = (
        y_future - y_current
    )

    model = build_model()

    model.fit(
        X,
        y_residual,
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        MODEL_DIR
        / f"aqi_{horizon}_hybrid.pkl"
    )

    joblib.dump(
        model,
        model_path,
    )

    metadata = {
        "model_version": MODEL_VERSION,
        "horizon": horizon,
        "target_column": target_column,
        "strategy": "hybrid_residual_forecasting",
        "algorithm": "RandomForestRegressor",
        "training_rows": int(len(training)),
        "training_start": str(
            training["date"].min()
        ),
        "training_end": str(
            training["date"].max()
        ),
        "feature_count": len(feature_columns),
        "features": feature_columns,
        "random_state": RANDOM_STATE,
        "artifact": str(
            model_path.relative_to(
                PROJECT_ROOT
            )
        ),
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    logger.info(
        "Training completed: %s",
        horizon,
    )

    logger.info(
        "Training rows: %d",
        len(training),
    )

    logger.info(
        "Saved model: %s",
        model_path,
    )

    return metadata


def save_registry(
    models: list[dict],
) -> None:
    """Save model registry."""

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    registry = {
        "project": "Pearls AQI Predictor",
        "model_version": MODEL_VERSION,
        "strategy": "Hybrid residual AQI forecasting",
        "selection_basis": "Walk-forward validation",
        "source_dataset": str(
            INPUT_FILE.relative_to(
                PROJECT_ROOT
            )
        ),
        "models": models,
    }

    REGISTRY_FILE.write_text(
        json.dumps(
            registry,
            indent=4,
        ),
        encoding="utf-8",
    )

    logger.info(
        "Model registry saved: %s",
        REGISTRY_FILE,
    )


def main() -> None:
    """Train all final AQI forecasting models."""

    logger.info("=" * 72)
    logger.info("PEARLS AQI PREDICTOR")
    logger.info("FINAL HYBRID MODEL TRAINING")
    logger.info("=" * 72)

    df = load_dataset()

    feature_columns = get_feature_columns(
        df
    )

    logger.info(
        "Feature count: %d",
        len(feature_columns),
    )

    metadata: list[dict] = []

    for horizon, target_column in TARGETS.items():
        logger.info(
            "Training final model: %s",
            horizon,
        )

        model_metadata = train_horizon(
            df=df,
            feature_columns=feature_columns,
            horizon=horizon,
            target_column=target_column,
        )

        metadata.append(
            model_metadata
        )

    save_registry(
        metadata
    )

    logger.info("=" * 72)
    logger.info(
        "FINAL HYBRID MODEL TRAINING COMPLETED."
    )
    logger.info("=" * 72)

    print("\nFINAL MODEL ARTIFACTS")

    for item in metadata:
        print(
            f"{item['horizon']}: "
            f"{item['artifact']}"
        )

    print(
        f"\nRegistry: "
        f"{REGISTRY_FILE}"
    )


if __name__ == "__main__":
    main()
