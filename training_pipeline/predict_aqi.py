"""
Pearls AQI Predictor
====================

Production inference pipeline for 1-day, 2-day and 3-day AQI forecasts.

Loads the final hybrid residual models and generates forecasts from
the latest available daily AQI feature row.

No retraining is performed here.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FEATURE_FILE = (
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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "predictions"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "latest_aqi_forecast.json"
)


# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_FILES = {
    "day_1": MODEL_DIR / "aqi_day_1_hybrid.pkl",
    "day_2": MODEL_DIR / "aqi_day_2_hybrid.pkl",
    "day_3": MODEL_DIR / "aqi_day_3_hybrid.pkl",
}


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

def load_latest_features() -> pd.DataFrame:
    """Load the latest feature row."""

    if not FEATURE_FILE.exists():
        raise FileNotFoundError(
            f"Feature dataset not found: {FEATURE_FILE}"
        )

    df = pd.read_parquet(
        FEATURE_FILE
    )

    if df.empty:
        raise ValueError(
            "Feature dataset is empty."
        )

    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    df = df.loc[
        :,
        ~df.columns.duplicated(
            keep="first"
        ),
    ].copy()

    required = {
        "date",
        "city",
        "aqi",
    }

    missing = (
        required
        - set(df.columns)
    )

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

    df = (
        df
        .dropna(
            subset=[
                "date",
                "aqi",
            ]
        )
        .sort_values("date")
        .drop_duplicates(
            subset=["date"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    if df.empty:
        raise ValueError(
            "No valid daily AQI feature rows available."
        )

    latest = df.tail(1).copy()

    logger.info(
        "Latest feature date: %s",
        latest.iloc[0]["date"],
    )

    logger.info(
        "Latest AQI: %.2f",
        latest.iloc[0]["aqi"],
    )

    return latest


# ============================================================================
# LOAD REGISTRY
# ============================================================================

def load_registry() -> dict:
    """Load model registry metadata."""

    if not REGISTRY_FILE.exists():
        raise FileNotFoundError(
            f"Model registry not found: {REGISTRY_FILE}"
        )

    registry = json.loads(
        REGISTRY_FILE.read_text(
            encoding="utf-8"
        )
    )

    if "models" not in registry:
        raise ValueError(
            "Invalid model registry."
        )

    return registry


# ============================================================================
# LOAD MODELS
# ============================================================================

def load_models() -> dict[str, object]:
    """Load all final forecasting models."""

    models = {}

    for horizon, model_path in MODEL_FILES.items():

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}"
            )

        models[horizon] = joblib.load(
            model_path
        )

        logger.info(
            "Loaded model: %s",
            model_path.name,
        )

    return models


# ============================================================================
# FEATURE VALIDATION
# ============================================================================

def validate_features(
    latest_row: pd.DataFrame,
    registry: dict,
) -> list[str]:
    """Validate that runtime features match the training contract."""

    model_metadata = registry.get(
        "models",
        []
    )

    if not model_metadata:
        raise ValueError(
            "Model registry contains no model metadata."
        )

    reference_features = (
        model_metadata[0].get(
            "features",
            []
        )
    )

    if not reference_features:
        raise ValueError(
            "Feature list missing from registry."
        )

    missing = [
        feature
        for feature in reference_features
        if feature not in latest_row.columns
    ]

    if missing:
        raise ValueError(
            "Runtime feature mismatch. Missing: "
            + ", ".join(missing)
        )

    return reference_features


# ============================================================================
# AQI CATEGORY
# ============================================================================

def classify_aqi(
    aqi_value: float,
) -> str:
    """
    Classify AQI using the project's configured Punjab-aligned
    reporting categories.

    The target methodology remains documented separately.
    """

    if aqi_value <= 50:
        return "Good"

    if aqi_value <= 100:
        return "Satisfactory"

    if aqi_value <= 150:
        return "Moderate"

    if aqi_value <= 200:
        return "Unhealthy for Sensitive Groups"

    if aqi_value <= 300:
        return "Unhealthy"

    if aqi_value <= 400:
        return "Very Unhealthy"

    return "Hazardous"


# ============================================================================
# FORECAST
# ============================================================================

def generate_forecast(
    latest_row: pd.DataFrame,
    models: dict[str, object],
    feature_columns: list[str],
) -> dict:
    """Generate Day+1, Day+2 and Day+3 AQI forecasts."""

    X = latest_row[
        feature_columns
    ].copy()

    current_aqi = float(
        latest_row.iloc[0]["aqi"]
    )

    forecast_date = pd.Timestamp(
        latest_row.iloc[0]["date"]
    )

    city = str(
        latest_row.iloc[0]["city"]
    )

    forecasts = []

    for horizon in (
        "day_1",
        "day_2",
        "day_3",
    ):

        model = models[horizon]

        predicted_residual = model.predict(
            X
        )

        predicted_residual = float(
            np.asarray(
                predicted_residual,
                dtype=float,
            ).reshape(-1)[0]
        )

        predicted_aqi = (
            current_aqi
            + predicted_residual
        )

        predicted_aqi = max(
            0.0,
            predicted_aqi,
        )

        if horizon == "day_1":
            days_ahead = 1
        elif horizon == "day_2":
            days_ahead = 2
        else:
            days_ahead = 3

        target_date = (
            forecast_date
            + pd.Timedelta(
                days=days_ahead
            )
        )

        forecasts.append(
            {
                "horizon": horizon,
                "date": target_date.date().isoformat(),
                "predicted_aqi": round(
                    predicted_aqi,
                    2,
                ),
                "category": classify_aqi(
                    predicted_aqi
                ),
                "current_aqi": round(
                    current_aqi,
                    2,
                ),
                "predicted_change": round(
                    predicted_residual,
                    2,
                ),
            }
        )

    result = {
        "city": city,
        "forecast_generated_from": (
            forecast_date.date().isoformat()
        ),
        "current_aqi": round(
            current_aqi,
            2,
        ),
        "forecasts": forecasts,
    }

    return result


# ============================================================================
# SAVE
# ============================================================================

def save_forecast(
    result: dict,
) -> None:
    """Persist the latest forecast."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_FILE.write_text(
        json.dumps(
            result,
            indent=4,
        ),
        encoding="utf-8",
    )

    logger.info(
        "Forecast saved: %s",
        OUTPUT_FILE,
    )


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    """Execute production AQI inference."""

    logger.info("=" * 72)
    logger.info(
        "PEARLS AQI PREDICTOR"
    )
    logger.info(
        "PRODUCTION AQI INFERENCE"
    )
    logger.info("=" * 72)

    latest_row = load_latest_features()

    registry = load_registry()

    models = load_models()

    feature_columns = validate_features(
        latest_row,
        registry,
    )

    result = generate_forecast(
        latest_row=latest_row,
        models=models,
        feature_columns=feature_columns,
    )

    save_forecast(
        result
    )

    print(
        "\nLATEST AQI FORECAST"
    )

    print(
        f"City: {result['city']}"
    )

    print(
        f"Current AQI: {result['current_aqi']}"
    )

    for forecast in result["forecasts"]:
        print(
            f"{forecast['horizon']} | "
            f"{forecast['date']} | "
            f"AQI={forecast['predicted_aqi']} | "
            f"{forecast['category']}"
        )

    logger.info("=" * 72)
    logger.info(
        "PRODUCTION AQI INFERENCE COMPLETED."
    )
    logger.info("=" * 72)


if __name__ == "__main__":
    main()