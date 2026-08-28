"""
Production-grade canonical dataset builder.

Pearls AQI Predictor
--------------------
Purpose:
    Build the canonical hourly Lahore air-quality dataset by combining
    validated OpenAQ observations with validated historical weather data.

Design principles:
    - Real observations only.
    - No synthetic observations.
    - No blind interpolation.
    - No future-data leakage.
    - Sensor-aware aggregation.
    - Explicit timestamp normalization.
    - Reproducible processing.
    - Machine-readable quality report.
    - Parquet as canonical storage format.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd


# ============================================================================
# PROJECT CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

VALIDATED_AIR_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "validated_openaq_lahore.parquet"
)

WEATHER_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "weather_lahore.csv"
)

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "data_quality"
)

FINAL_PARQUET = (
    PROCESSED_DIR
    / "canonical_lahore_hourly.parquet"
)

FINAL_CSV = (
    PROCESSED_DIR
    / "canonical_lahore_hourly.csv"
)

QUALITY_REPORT = (
    REPORT_DIR
    / "canonical_dataset_report.json"
)


# ============================================================================
# CONSTANTS
# ============================================================================

CITY = "Lahore"

REQUIRED_AIR_COLUMNS = {
    "timestamp",
    "sensor_id",
    "parameter",
    "value",
}

REQUIRED_WEATHER_COLUMNS = {
    "timestamp",
    "temperature",
    "humidity",
    "pressure",
    "clouds",
    "wind_speed",
    "wind_direction",
    "precipitation",
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
# CUSTOM EXCEPTION
# ============================================================================

class DatasetBuildError(RuntimeError):
    """Raised when canonical dataset construction fails."""


# ============================================================================
# LOAD VALIDATED AIR QUALITY
# ============================================================================

def load_air_quality() -> pd.DataFrame:
    """Load validated OpenAQ observations."""

    if not VALIDATED_AIR_FILE.exists():
        raise FileNotFoundError(
            f"Validated OpenAQ file not found: "
            f"{VALIDATED_AIR_FILE}"
        )

    logger.info(
        "Loading validated OpenAQ data."
    )

    dataframe = pd.read_parquet(
        VALIDATED_AIR_FILE
    )

    missing = (
        REQUIRED_AIR_COLUMNS
        - set(dataframe.columns)
    )

    if missing:
        raise DatasetBuildError(
            "OpenAQ dataset is missing columns: "
            + ", ".join(sorted(missing))
        )

    dataframe["timestamp"] = pd.to_datetime(
        dataframe["timestamp"],
        utc=True,
        errors="coerce",
    )

    dataframe["parameter"] = (
        dataframe["parameter"]
        .astype("string")
        .str.lower()
        .str.strip()
    )

    dataframe["value"] = pd.to_numeric(
        dataframe["value"],
        errors="coerce",
    )

    dataframe = dataframe.dropna(
        subset=[
            "timestamp",
            "sensor_id",
            "parameter",
            "value",
        ]
    )

    logger.info(
        "OpenAQ observations loaded: %d",
        len(dataframe),
    )

    logger.info(
        "OpenAQ sensors: %d",
        dataframe["sensor_id"].nunique(),
    )

    return dataframe


# ============================================================================
# LOAD WEATHER
# ============================================================================

def load_weather() -> pd.DataFrame:
    """Load historical Open-Meteo observations."""

    if not WEATHER_FILE.exists():
        raise FileNotFoundError(
            f"Weather file not found: {WEATHER_FILE}"
        )

    logger.info(
        "Loading historical weather data."
    )

    weather = pd.read_csv(
        WEATHER_FILE
    )

    missing = (
        REQUIRED_WEATHER_COLUMNS
        - set(weather.columns)
    )

    if missing:
        raise DatasetBuildError(
            "Weather dataset is missing columns: "
            + ", ".join(sorted(missing))
        )

    weather["timestamp"] = pd.to_datetime(
        weather["timestamp"],
        utc=True,
        errors="coerce",
    )

    numeric_columns = [
        "temperature",
        "humidity",
        "pressure",
        "clouds",
        "wind_speed",
        "wind_direction",
        "precipitation",
    ]

    for column in numeric_columns:
        weather[column] = pd.to_numeric(
            weather[column],
            errors="coerce",
        )

    weather = weather.dropna(
        subset=["timestamp"]
    )

    logger.info(
        "Weather observations loaded: %d",
        len(weather),
    )

    return weather


# ============================================================================
# AIR QUALITY AGGREGATION
# ============================================================================

def aggregate_air_quality(
    air_quality: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate multiple Lahore monitoring sensors to hourly city level.

    Important:
        The aggregation is performed separately by timestamp and
        pollutant. Sensor measurements are averaged only when real
        observations exist for that hour.

    Additional provenance metrics are retained so the model can
    distinguish highly observed hours from poorly observed hours.
    """

    logger.info(
        "Aggregating sensor observations to hourly city level."
    )

    pm25 = air_quality[
        air_quality["parameter"] == "pm25"
    ].copy()

    if pm25.empty:
        raise DatasetBuildError(
            "No PM2.5 observations are available."
        )

    aggregation = (
        pm25
        .groupby("timestamp", as_index=False)
        .agg(
            pm2_5=("value", "mean"),
            pm2_5_median=("value", "median"),
            pm2_5_std=("value", "std"),
            pm2_5_min=("value", "min"),
            pm2_5_max=("value", "max"),
            sensor_count=("sensor_id", "nunique"),
            observation_count=("value", "count"),
        )
    )

    aggregation["pm2_5_std"] = (
        aggregation["pm2_5_std"]
        .fillna(0.0)
    )

    aggregation["city"] = CITY

    aggregation.sort_values(
        "timestamp",
        inplace=True,
    )

    aggregation.reset_index(
        drop=True,
        inplace=True,
    )

    logger.info(
        "Canonical PM2.5 hourly observations: %d",
        len(aggregation),
    )

    logger.info(
        "Maximum sensors contributing to one hour: %d",
        aggregation["sensor_count"].max(),
    )

    return aggregation


# ============================================================================
# WEATHER DEDUPLICATION
# ============================================================================

def prepare_weather(
    weather: pd.DataFrame,
) -> pd.DataFrame:
    """
    Ensure one weather record exists per timestamp.

    Weather data should normally already be hourly. If duplicate
    timestamps occur, numerical values are averaged rather than
    arbitrarily selecting one record.
    """

    numeric_columns = [
        "temperature",
        "humidity",
        "pressure",
        "clouds",
        "wind_speed",
        "wind_direction",
        "precipitation",
    ]

    duplicate_count = int(
        weather.duplicated(
            subset=["timestamp"]
        ).sum()
    )

    if duplicate_count:
        logger.warning(
            "Weather duplicate timestamps detected: %d",
            duplicate_count,
        )

    weather = (
        weather
        .groupby("timestamp", as_index=False)[
            numeric_columns
        ]
        .mean()
    )

    return weather


# ============================================================================
# TEMPORAL MERGE
# ============================================================================

def merge_air_and_weather(
    air_quality: pd.DataFrame,
    weather: pd.DataFrame,
) -> pd.DataFrame:
    """
    Perform exact timestamp alignment between air quality and weather.

    No nearest-neighbour matching and no interpolation are used.
    """

    logger.info(
        "Aligning air quality and weather observations."
    )

    dataset = pd.merge(
        air_quality,
        weather,
        on="timestamp",
        how="inner",
        validate="one_to_one",
    )

    if dataset.empty:
        raise DatasetBuildError(
            "Air-quality/weather temporal intersection is empty."
        )

    dataset.sort_values(
        "timestamp",
        inplace=True,
    )

    dataset.reset_index(
        drop=True,
        inplace=True,
    )

    logger.info(
        "Merged observations: %d",
        len(dataset),
    )

    return dataset


# ============================================================================
# DATA QUALITY METRICS
# ============================================================================

def calculate_quality_metrics(
    dataset: pd.DataFrame,
) -> dict:
    """Generate comprehensive canonical dataset metrics."""

    if dataset.empty:
        raise DatasetBuildError(
            "Cannot calculate quality metrics for empty dataset."
        )

    timestamps = (
        dataset["timestamp"]
        .drop_duplicates()
        .sort_values()
    )

    expected_hours = pd.date_range(
        start=timestamps.min(),
        end=timestamps.max(),
        freq="h",
        tz="UTC",
    )

    missing_hours = expected_hours.difference(
        pd.DatetimeIndex(timestamps)
    )

    metrics = {
        "dataset": "Pearls AQI Predictor",
        "city": CITY,
        "source": [
            "OpenAQ",
            "Open-Meteo",
        ],
        "status": "validated",

        "rows": int(len(dataset)),
        "columns": int(len(dataset.columns)),

        "unique_timestamps": int(
            dataset["timestamp"].nunique()
        ),

        "unique_sensors_observed": int(
            dataset["sensor_count"].max()
        ),

        "start_timestamp": str(
            dataset["timestamp"].min()
        ),

        "end_timestamp": str(
            dataset["timestamp"].max()
        ),

        "expected_hourly_slots": int(
            len(expected_hours)
        ),

        "observed_hourly_slots": int(
            len(timestamps)
        ),

        "missing_hourly_slots": int(
            len(missing_hours)
        ),

        "temporal_coverage_ratio": float(
            len(timestamps)
            / len(expected_hours)
        ),

        "duplicate_rows": int(
            dataset.duplicated().sum()
        ),

        "missing_values": {
            column: int(
                dataset[column].isna().sum()
            )
            for column in dataset.columns
        },

        "pm25_statistics": {
            "mean": float(
                dataset["pm2_5"].mean()
            ),
            "median": float(
                dataset["pm2_5"].median()
            ),
            "minimum": float(
                dataset["pm2_5"].min()
            ),
            "maximum": float(
                dataset["pm2_5"].max()
            ),
        },

        "sensor_statistics": {
            "mean_sensors_per_hour": float(
                dataset["sensor_count"].mean()
            ),
            "median_sensors_per_hour": float(
                dataset["sensor_count"].median()
            ),
            "minimum_sensors_per_hour": int(
                dataset["sensor_count"].min()
            ),
            "maximum_sensors_per_hour": int(
                dataset["sensor_count"].max()
            ),
        },
    }

    return metrics


# ============================================================================
# FINAL INTEGRITY CHECKS
# ============================================================================

def validate_canonical_dataset(
    dataset: pd.DataFrame,
) -> None:
    """Perform final canonical dataset integrity checks."""

    required_columns = {
        "timestamp",
        "pm2_5",
        "sensor_count",
        "observation_count",
        "temperature",
        "humidity",
        "pressure",
        "clouds",
        "wind_speed",
        "wind_direction",
        "precipitation",
        "city",
    }

    missing = (
        required_columns
        - set(dataset.columns)
    )

    if missing:
        raise DatasetBuildError(
            "Canonical dataset missing columns: "
            + ", ".join(sorted(missing))
        )

    if dataset.empty:
        raise DatasetBuildError(
            "Canonical dataset is empty."
        )

    if dataset["timestamp"].duplicated().any():
        raise DatasetBuildError(
            "Canonical dataset contains duplicate timestamps."
        )

    if dataset["pm2_5"].isna().any():
        raise DatasetBuildError(
            "Canonical dataset contains missing PM2.5 values."
        )

    if (
        dataset["pm2_5"] < 0
    ).any():
        raise DatasetBuildError(
            "Negative PM2.5 values detected."
        )

    if (
        dataset["sensor_count"] < 1
    ).any():
        raise DatasetBuildError(
            "Invalid sensor counts detected."
        )

    if (
        dataset["observation_count"] < 1
    ).any():
        raise DatasetBuildError(
            "Invalid observation counts detected."
        )

    logger.info(
        "Canonical dataset integrity validation passed."
    )


# ============================================================================
# SAVE DATASET
# ============================================================================

def save_dataset(
    dataset: pd.DataFrame,
) -> None:
    """Persist canonical dataset in Parquet and CSV formats."""

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset.to_parquet(
        FINAL_PARQUET,
        index=False,
    )

    dataset.to_csv(
        FINAL_CSV,
        index=False,
    )

    logger.info(
        "Canonical Parquet saved: %s",
        FINAL_PARQUET,
    )

    logger.info(
        "Canonical CSV saved: %s",
        FINAL_CSV,
    )


# ============================================================================
# SAVE REPORT
# ============================================================================

def save_quality_report(
    metrics: dict,
) -> None:
    """Save machine-readable dataset quality report."""

    with QUALITY_REPORT.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=2,
            ensure_ascii=False,
        )

    logger.info(
        "Canonical quality report saved: %s",
        QUALITY_REPORT,
    )


# ============================================================================
# PIPELINE
# ============================================================================

def build_canonical_dataset() -> pd.DataFrame:
    """Build the final canonical hourly dataset."""

    logger.info("=" * 72)
    logger.info(
        "PEARLS AQI PREDICTOR"
    )
    logger.info(
        "CANONICAL DATASET BUILD"
    )
    logger.info("=" * 72)

    air_quality = load_air_quality()

    weather = load_weather()

    air_quality = aggregate_air_quality(
        air_quality
    )

    weather = prepare_weather(
        weather
    )

    dataset = merge_air_and_weather(
        air_quality,
        weather,
    )

    validate_canonical_dataset(
        dataset
    )

    metrics = calculate_quality_metrics(
        dataset
    )

    save_dataset(
        dataset
    )

    save_quality_report(
        metrics
    )

    logger.info("=" * 72)
    logger.info(
        "CANONICAL DATASET BUILD COMPLETED"
    )
    logger.info("=" * 72)

    logger.info(
        "Final shape: %s",
        dataset.shape,
    )

    logger.info(
        "Final columns: %s",
        dataset.columns.tolist(),
    )

    logger.info(
        "Final time range: %s → %s",
        dataset["timestamp"].min(),
        dataset["timestamp"].max(),
    )

    return dataset


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    build_canonical_dataset()