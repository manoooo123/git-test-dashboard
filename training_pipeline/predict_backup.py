from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

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

TARGET_COLUMNS = {
    24: "target_24h",
    48: "target_48h",
    72: "target_72h",
}

LOG_FORMAT = (
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
)

logger = logging.getLogger(__name__)


# ============================================================
# AQI CONVERSION
# ============================================================

def calculate_aqi_from_pm25(pm25: float) -> int:
    """
    Convert PM2.5 concentration to US EPA-style AQI.

    PM2.5 is assumed to be in µg/m³.
    """

    if pd.isna(pm25):
        return 0

    pm25 = float(pm25)

    if pm25 < 0:
        pm25 = 0.0

    # PM2.5 AQI breakpoints
    breakpoints = [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500),
    ]

    # EPA calculation uses truncated PM2.5.
    pm25 = np.floor(pm25 * 10) / 10

    for (
        concentration_low,
        concentration_high,
        aqi_low,
        aqi_high,
    ) in breakpoints:

        if (
            concentration_low
            <= pm25
            <= concentration_high
        ):
            aqi = (
                (
                    (aqi_high - aqi_low)
                    / (concentration_high - concentration_low)
                )
                * (pm25 - concentration_low)
                + aqi_low
            )

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
        return (
            "Air quality is acceptable. "
            "Sensitive people should monitor conditions."
        )

    if aqi <= 150:
        return (
            "Sensitive groups should reduce "
            "prolonged outdoor activity."
        )

    if aqi <= 200:
        return (
            "Everyone may experience health effects. "
            "Reduce prolonged outdoor activity."
        )

    if aqi <= 300:
        return (
            "Health alert. Everyone should avoid "
            "prolonged outdoor activity."
        )

    return (
        "Health emergency. Avoid outdoor exposure "
        "and follow local health guidance."
    )


# ============================================================
# LOAD DATA
# ============================================================

def load_data() -> pd.DataFrame:
    """Load production feature dataset."""

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}"
        )

    logger.info(
        "Loading feature dataset: %s",
        DATA_FILE,
    )

    df = pd.read_csv(DATA_FILE)

    if "hour" not in df.columns:
        raise ValueError(
            "hour column is missing."
        )

    df["hour"] = pd.to_datetime(
        df["hour"],
        utc=True,
        errors="coerce",
    )

    df = df.dropna(subset=["hour"]).sort_values(["city", "hour"]).reset_index(drop=True)

    logger.info(
        "Dataset rows: %d",
        len(df),
    )

    logger.info(
        "Latest timestamp: %s",
        df["hour"].max(),
    )

    return df


# ============================================================
# FEATURE PREPARATION
# ============================================================

def get_feature_columns(
    df: pd.DataFrame,
) -> list[str]:
    """Return model input columns."""

    excluded = {
        "timestamp",
        "city",
        "coverage_quality",
        "hour",
        "is_missing_hour",
        "location_id",
        "sensor_id",
        "target_24h",
        "target_48h",
        "target_72h",
    }

    features = [
        column
        for column in df.columns
        if column not in excluded
    ]

    if not features:
        raise ValueError(
            "No model features found."
        )

    return features


def prepare_latest_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Prepare latest row for prediction."""

    feature_columns = get_feature_columns(df)

    latest = df.iloc[[-1]].copy()

    X = latest[feature_columns].copy()

    X = X.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    for column in X.columns:
        X[column] = pd.to_numeric(
            X[column],
            errors="coerce",
        )

    # Fill missing values using historical medians.
    medians = (
        df[feature_columns]
        .apply(pd.to_numeric, errors="coerce")
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .median()
    )

    X = X.fillna(medians)

    return X


# ============================================================
# PREDICTION
# ============================================================

def predict_horizon(
    model,
    X: pd.DataFrame,
) -> float:
    """Generate PM2.5 prediction."""

    prediction = model.predict(X)

    value = float(
        np.asarray(prediction).reshape(-1)[0]
    )

    # PM2.5 cannot be negative.
    return max(0.0, value)


def main() -> None:

    logger.info("=" * 72)
    logger.info("PEARLS AQI PREDICTOR")
    logger.info("3-DAY AQI FORECAST")
    logger.info("=" * 72)

    df = load_data()

    X_latest = prepare_latest_features(df)

    latest_timestamp = df["hour"].iloc[-1]

    current_pm25 = None

    if "pm2_5" in df.columns:
        current_pm25 = float(
            df["pm2_5"].iloc[-1]
        )

    predictions = []

    for horizon in HORIZONS:

        model_path = (
            MODELS_DIR
            / f"best_model_{horizon}h.joblib"
        )

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}"
            )

        logger.info(
            "Loading model: %s",
            model_path.name,
        )

        model = joblib.load(
            model_path
        )

        predicted_pm25 = predict_horizon(
            model,
            X_latest,
        )

        predicted_aqi = calculate_aqi_from_pm25(
            predicted_pm25
        )

        category = get_aqi_category(
            predicted_aqi
        )

        health_message = get_health_message(
            predicted_aqi
        )

        forecast_time = (
            latest_timestamp
            + pd.Timedelta(
                hours=horizon
            )
        )

        result = {
            "forecast_horizon_hours": horizon,
            "forecast_timestamp": forecast_time.isoformat(),
            "predicted_pm2_5": round(
                predicted_pm25,
                2,
            ),
            "predicted_aqi": predicted_aqi,
            "aqi_category": category,
            "health_message": health_message,
            "model": "Ridge",
        }

        predictions.append(result)

        logger.info(
            "%sh | PM2.5=%.2f | AQI=%d | %s",
            horizon,
            predicted_pm25,
            predicted_aqi,
            category,
        )

    # ========================================================
    # OUTPUT
    # ========================================================

    output = {
        "project": "Pearls AQI Predictor",
        "city": "3 Cities",
        "generated_at": pd.Timestamp.now(
            tz="UTC"
        ).isoformat(),
        "latest_observation": latest_timestamp.isoformat(),
        "current_pm2_5": (
            round(current_pm25, 2)
            if current_pm25 is not None
            else None
        ),
        "forecast": predictions,
    }

    output_file = (
        OUTPUT_DIR
        / "aqi_forecast.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            output,
            file,
            indent=4,
        )

    forecast_csv = (
        OUTPUT_DIR
        / "aqi_forecast.csv"
    )

    pd.DataFrame(
        predictions
    ).to_csv(
        forecast_csv,
        index=False,
    )

    logger.info(
        "JSON forecast saved: %s",
        output_file,
    )

    logger.info(
        "CSV forecast saved: %s",
        forecast_csv,
    )

    logger.info("=" * 72)
    logger.info(
        "3-DAY AQI FORECAST COMPLETED SUCCESSFULLY."
    )
    logger.info("=" * 72)

    print("\n")
    print("=" * 72)
    print("       PEARLS AQI PREDICTOR — 3 DAY FORECAST")
    print("=" * 72)

    for item in predictions:

        print(
            f"\n{item['forecast_horizon_hours']} HOURS"
        )

        print(
            f"Time       : "
            f"{item['forecast_timestamp']}"
        )

        print(
            f"PM2.5      : "
            f"{item['predicted_pm2_5']} µg/m³"
        )

        print(
            f"AQI        : "
            f"{item['predicted_aqi']}"
        )

        print(
            f"Category   : "
            f"{item['aqi_category']}"
        )

        print(
            f"Health     : "
            f"{item['health_message']}"
        )

    print("\n")
    print("=" * 72)


if __name__ == "__main__":
    main()











