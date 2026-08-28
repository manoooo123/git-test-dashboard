"""
Pearls AQI Predictor - 3-City / 3-Day Forecast Inference Engine.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from utils.db import log_prediction
from utils.feature_store import feature_store

# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_features_3cities.csv"
)

MODELS_DIR = PROJECT_ROOT / "models" / "3cities"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "predictions"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

HORIZONS = [24, 48, 72]

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


# ============================================================
# AQI CONVERSION
# ============================================================

def calculate_aqi_from_pm25(pm25: float) -> int:
    """Convert PM2.5 concentration to US EPA AQI."""
    if pd.isna(pm25) or pm25 < 0:
        return 0

    pm25 = float(pm25)
    breakpoints = [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500),
    ]

    pm25 = np.floor(pm25 * 10) / 10

    for (c_low, c_high, aqi_low, aqi_high) in breakpoints:
        if c_low <= pm25 <= c_high:
            aqi = ((aqi_high - aqi_low) / (c_high - c_low)) * (pm25 - c_low) + aqi_low
            return int(round(aqi))

    return 500


def get_aqi_category(aqi: int) -> str:
    """Return AQI health category."""
    if aqi <= 50:
        return "Good"
    if aqi <= 100:
        return "Moderate"
    if aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    if aqi <= 200:
        return "Unhealthy"
    if aqi <= 300:
        return "Very Unhealthy"
    return "Hazardous"


def get_health_message(aqi: int) -> str:
    """Return health recommendation."""
    if aqi <= 50:
        return "Air quality is good."
    if aqi <= 100:
        return "Air quality is acceptable. Sensitive people should monitor conditions."
    if aqi <= 150:
        return "Sensitive groups should reduce prolonged outdoor activity."
    if aqi <= 200:
        return "Everyone may experience health effects. Reduce prolonged outdoor activity."
    if aqi <= 300:
        return "Health alert. Everyone should avoid prolonged outdoor activity."
    return "Health emergency. Avoid outdoor exposure and follow local health guidance."


# ============================================================
# LOAD DATA & PREDICTION
# ============================================================

def load_data() -> pd.DataFrame:
    """Load feature dataset via Feature Store Manager."""
    df = feature_store.load_features()
    if df.empty:
        raise FileNotFoundError(f"Feature dataset not available: {DATA_FILE}")

    if "hour" in df.columns:
        df["timestamp"] = pd.to_datetime(df["hour"], utc=True, errors="coerce")
    elif "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    else:
        df["timestamp"] = pd.Timestamp.now(tz="UTC")

    df = df.dropna(subset=["timestamp"]).sort_values(["city", "timestamp"]).reset_index(drop=True)
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return numeric model features."""
    excluded = {
        "timestamp", "hour", "city", "coverage_quality",
        "target_24h", "target_48h", "target_72h",
        "target_pm2_5_24h", "target_pm2_5_48h", "target_pm2_5_72h"
    }
    return [col for col in df.columns if col not in excluded and np.issubdtype(df[col].dtype, np.number)]


def main() -> None:
    logger.info("=" * 72)
    logger.info("PEARLS AQI PREDICTOR | 3-CITY FORECAST INFERENCE")
    logger.info("=" * 72)

    df = load_data()
    cities = ["Lahore", "Islamabad", "Faisalabad"]
    all_predictions = []

    for city in cities:
        city_df = df[df["city"].str.lower() == city.lower()].copy()
        if city_df.empty:
            logger.warning("No feature data found for city: %s", city)
            continue

        city_df = city_df.sort_values("timestamp").reset_index(drop=True)
        feature_cols = get_feature_columns(city_df)

        latest_row = city_df.iloc[-1]
        latest_timestamp = latest_row["timestamp"]
        X_latest = pd.DataFrame([latest_row[feature_cols]]).fillna(0)

        city_aqis = {}

        for horizon in HORIZONS:
            model_path = MODELS_DIR / f"best_model_{horizon}h.joblib"
            if not model_path.exists():
                logger.warning(f"Model file missing: {model_path}")
                continue

            try:
                model = joblib.load(model_path)
                raw_prediction = model.predict(X_latest)[0]
                
                # SANITY CHECK 1: Detect NaN
                if pd.isna(raw_prediction):
                    logger.error(f"[{city}] Model returned NaN for +{horizon}h")
                    continue
                
                # SANITY CHECK 2: Detect Infinity
                if np.isinf(raw_prediction):
                    logger.error(f"[{city}] Model returned Inf for +{horizon}h")
                    continue
                
                # SANITY CHECK 3: Validate type and range
                predicted_pm25 = float(raw_prediction)
                predicted_pm25 = max(0.0, predicted_pm25)  # Ensure non-negative
                
                # SANITY CHECK 4: Warn on extreme values
                if predicted_pm25 > 1000:
                    logger.warning(f"[{city}] Unusually high PM2.5 for +{horizon}h: {predicted_pm25:.2f}")
                
                # Calculate AQI
                predicted_aqi = calculate_aqi_from_pm25(predicted_pm25)
                
                # SANITY CHECK 5: Validate AQI
                if predicted_aqi == 0 and predicted_pm25 > 0:
                    logger.error(f"[{city}] AQI calculation failed for PM2.5={predicted_pm25:.2f}")
                    continue
                
                category = get_aqi_category(predicted_aqi)
                health_message = get_health_message(predicted_aqi)
                forecast_time = latest_timestamp + pd.Timedelta(hours=horizon)

                result = {
                    "city": city,
                    "forecast_horizon_hours": horizon,
                    "forecast_timestamp": forecast_time.isoformat(),
                    "predicted_pm2_5": round(predicted_pm25, 2),
                    "predicted_aqi": predicted_aqi,
                    "aqi_category": category,
                    "health_message": health_message,
                    "model_version": "v2.0.0-3cities",
                }
                all_predictions.append(result)
                city_aqis[horizon] = predicted_aqi
                
                logger.info(f"[{city}] +{horizon}h: PM2.5={predicted_pm25:.2f} → AQI={predicted_aqi} ({category})")
                
            except Exception as e:
                logger.error(f"[{city}] Prediction failed for +{horizon}h: {e}")
                continue

        # Log prediction to SQLite DB
        log_prediction(
            city,
            city_aqis.get(24, 0),
            city_aqis.get(48, 0),
            city_aqis.get(72, 0),
            model_version="v2.0.0-3cities"
        )

    output = {
        "project": "Pearls AQI Predictor",
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "data_source": "OpenAQ v3 API + Open-Meteo Weather",
        "forecast": all_predictions,
    }

    output_file = OUTPUT_DIR / "aqi_forecast_3cities.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4)

    pd.DataFrame(all_predictions).to_csv(OUTPUT_DIR / "aqi_forecast_3cities.csv", index=False)
    logger.info(f"Successfully generated forecasts for {len(cities)} cities!")


if __name__ == "__main__":
    main()
