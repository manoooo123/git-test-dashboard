"""
Production-grade, time-aware feature engineering
for Pearls AQI Predictor.

IMPORTANT
---------
This module uses timestamp-based lags and future targets.

It does NOT assume that every DataFrame row represents
the next consecutive hour.

This is critical because real-world monitoring data
contains temporal gaps.

No synthetic observations are created.
No missing observations are interpolated.
No future information is used in input features.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "city_hourly_3cities.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "data_quality"
)

OUTPUT_PARQUET = (
    OUTPUT_DIR
    / "model_features.parquet"
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "model_features.csv"
)

REPORT_FILE = (
    REPORT_DIR
    / "feature_engineering_report.json"
)


# ============================================================================
# CONFIGURATION
# ============================================================================

TARGET_COLUMN = "pm2_5"

LAG_HOURS = (
    1,
    3,
    6,
    12,
    24,
    48,
    72,
)

ROLLING_WINDOWS_HOURS = (
    6,
    12,
    24,
    48,
    72,
)

FORECAST_HORIZONS = {
    "target_pm2_5_24h": 24,
    "target_pm2_5_48h": 48,
    "target_pm2_5_72h": 72,
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
# DATA LOADING
# ============================================================================

def load_canonical_dataset() -> pd.DataFrame:
    """
    Load the canonical hourly dataset.

    Returns
    -------
    pd.DataFrame
        Validated canonical observations.
    """

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Canonical dataset not found: {INPUT_FILE}"
        )

    logger.info(
        "Loading canonical dataset: %s",
        INPUT_FILE,
    )

    df = pd.read_csv(INPUT_FILE)

    if df.empty:
        raise ValueError(
            "Canonical dataset is empty."
        )

    required_columns = {
        "city",
        "timestamp",
        "pm2_5",
        "pm2_5_mean",
        "pm2_5_std",
        "pm2_5_min",
        "pm2_5_max",
        "sensor_count",
        "observation_count",
        "temperature",
        "humidity",
        "pressure",
        "clouds",
        "wind_speed",
        "wind_direction",
        "precipitation",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            "Canonical dataset is missing columns: "
            + ", ".join(sorted(missing))
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
        errors="coerce",
    )

    df = (
        df.dropna(
            subset=[
                "timestamp",
                "pm2_5",
            ]
        )
        .sort_values("timestamp")
        .drop_duplicates(
            subset=["city", "timestamp"]
        )
        .reset_index(drop=True)
    )

    logger.info(
        "Canonical rows: %d",
        len(df),
    )

    return df


# ============================================================================
# TIME FEATURES
# ============================================================================

def add_time_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Add calendar and cyclical time features."""

    result = df.copy()

    ts = result["timestamp"]

    result["hour"] = ts.dt.hour

    result["day_of_week"] = (
        ts.dt.dayofweek
    )

    result["day_of_year"] = (
        ts.dt.dayofyear
    )

    result["month"] = ts.dt.month

    result["quarter"] = ts.dt.quarter

    result["is_weekend"] = (
        ts.dt.dayofweek >= 5
    ).astype(int)

    result["hour_sin"] = np.sin(
        2 * np.pi * result["hour"] / 24
    )

    result["hour_cos"] = np.cos(
        2 * np.pi * result["hour"] / 24
    )

    result["day_of_week_sin"] = np.sin(
        2 * np.pi
        * result["day_of_week"]
        / 7
    )

    result["day_of_week_cos"] = np.cos(
        2 * np.pi
        * result["day_of_week"]
        / 7
    )

    result["month_sin"] = np.sin(
        2 * np.pi
        * result["month"]
        / 12
    )

    result["month_cos"] = np.cos(
        2 * np.pi
        * result["month"]
        / 12
    )

    return result


# ============================================================================
# EXACT TIME-BASED LAG HELPER
# ============================================================================

def add_exact_lag(
    df: pd.DataFrame,
    source_column: str,
    lag_hours: int,
    output_name: str,
) -> pd.DataFrame:
    """
    Add an exact timestamp-based lag using city-level aggregation.

    Multiple sensors can exist for the same city and timestamp.
    Therefore, the lag lookup uses the city-level median value.
    """

    result = df.copy()

    lookup = (
        df[
            [
                "city",
                "timestamp",
                source_column,
            ]
        ]
        .groupby(
            ["city", "timestamp"],
            as_index=False,
        )[source_column]
        .median()
    )

    lookup["timestamp"] = (
        lookup["timestamp"]
        + pd.Timedelta(hours=lag_hours)
    )

    lookup.rename(
        columns={
            source_column: output_name
        },
        inplace=True,
    )

    result = result.merge(
        lookup,
        on=["city", "timestamp"],
        how="left",
        validate="many_to_one",
    )

    return result


# ============================================================================
# PM2.5 LAGS
# ============================================================================

def add_pm25_lags(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Add exact timestamp-based historical PM2.5 lags."""

    result = df.copy()

    for lag in LAG_HOURS:

        result = add_exact_lag(
            result,
            source_column=TARGET_COLUMN,
            lag_hours=lag,
            output_name=(
                f"pm2_5_lag_{lag}h"
            ),
        )

    return result


# ============================================================================
# TIME-BASED ROLLING FEATURES
# ============================================================================

def add_rolling_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add city-level time-based rolling PM2.5 statistics.

    Rolling statistics are calculated separately for each city.
    No future observations are used.
    """

    result = df.copy()

    base = (
        df[
            [
                "city",
                "timestamp",
                TARGET_COLUMN,
            ]
        ]
        .groupby(
            ["city", "timestamp"],
            as_index=False,
        )[TARGET_COLUMN]
        .median()
        .sort_values(
            ["city", "timestamp"]
        )
    )

    grouped = base.groupby("city")[TARGET_COLUMN]

    for hours in ROLLING_WINDOWS_HOURS:
        result_name = f"pm2_5_rolling_mean_{hours}h"

        rolling_values = (
            grouped
            .rolling(
                window=hours,
                min_periods=1,
            )
            .mean()
            .reset_index(
                level=0,
                drop=True,
            )
        )

        rolling_df = base[
            [
                "city",
                "timestamp",
            ]
        ].copy()

        rolling_df[result_name] = (
            rolling_values.to_numpy()
        )

        result = result.merge(
            rolling_df,
            on=["city", "timestamp"],
            how="left",
            validate="many_to_one",
        )

    return result


# ============================================================================
# TREND FEATURES
# ============================================================================

def add_pm25_trends(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Create exact historical PM2.5 change features."""

    result = df.copy()

    lag_columns = {
        lag: f"pm2_5_lag_{lag}h"
        for lag in LAG_HOURS
    }

    result["pm2_5_change_1h"] = (
        result[TARGET_COLUMN]
        - result[lag_columns[1]]
    )

    result["pm2_5_change_3h"] = (
        result[TARGET_COLUMN]
        - result[lag_columns[3]]
    )

    result["pm2_5_change_6h"] = (
        result[TARGET_COLUMN]
        - result[lag_columns[6]]
    )

    result["pm2_5_change_24h"] = (
        result[TARGET_COLUMN]
        - result[lag_columns[24]]
    )

    result["pm2_5_pct_change_24h"] = (
        (
            result[TARGET_COLUMN]
            / result[lag_columns[24]]
        )
        - 1.0
    ).replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return result


# ============================================================================
# WEATHER FEATURES
# ============================================================================

def add_weather_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Create leakage-safe weather-derived features."""

    result = df.copy()

    wind_radians = np.deg2rad(
        result["wind_direction"]
    )

    result["wind_direction_sin"] = np.sin(
        wind_radians
    )

    result["wind_direction_cos"] = np.cos(
        wind_radians
    )

    result["temperature_humidity_interaction"] = (
        result["temperature"]
        * result["humidity"]
    )

    # Exact historical weather changes.
    weather_columns = {
        "temperature": "temperature",
        "humidity": "humidity",
        "pressure": "pressure",
        "wind_speed": "wind_speed",
    }

    for source_column, prefix in weather_columns.items():

        result = add_exact_lag(
            result,
            source_column=source_column,
            lag_hours=24,
            output_name=(
                f"{prefix}_lag_24h"
            ),
        )

        result[f"{prefix}_change_24h"] = (
            result[source_column]
            - result[f"{prefix}_lag_24h"]
        )

    return result


# ============================================================================
# MEASUREMENT QUALITY
# ============================================================================

def add_quality_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Create measurement-quality features for the hourly 3-city dataset."""

    result = df.copy()

    result["observation_available"] = (
        result["pm2_5"].notna().astype(int)
    )

    result["sensor_count"] = (
        pd.to_numeric(
            result["sensor_count"],
            errors="coerce",
        )
        .fillna(0)
    )

    result["observation_count"] = (
        pd.to_numeric(
            result["observation_count"],
            errors="coerce",
        )
        .fillna(0)
    )

    return result

# ============================================================================
# FUTURE TARGETS
# ============================================================================

def add_future_targets(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Create exact 24h, 48h and 72h future PM2.5 targets per city."""

    result = df.copy()

    for hours in (24, 48, 72):

        lookup = result[
            [
                "city",
                "timestamp",
                TARGET_COLUMN,
            ]
        ].copy()

        lookup["timestamp"] = (
            lookup["timestamp"]
            - pd.Timedelta(hours=hours)
        )

        lookup.rename(
            columns={
                TARGET_COLUMN: f"target_pm2_5_{hours}h"
            },
            inplace=True,
        )

        lookup = (
            lookup
            .drop_duplicates(
                subset=["city", "timestamp"],
                keep="last",
            )
        )

        result = result.merge(
            lookup,
            on=["city", "timestamp"],
            how="left",
            validate="many_to_one",
        )

    return result


# ============================================================================
# VALIDATION
# ============================================================================

def validate_feature_dataset(
    df: pd.DataFrame,
) -> None:
    """Validate feature/target integrity."""

    if df.empty:
        raise ValueError(
            "Feature dataset is empty."
        )

    if df.duplicated(
        subset=["city", "timestamp"]
    ).any():
        raise ValueError(
            "Duplicate city/timestamp combinations detected."
        )

    numeric_columns = df.select_dtypes(
        include=[np.number]
    ).columns

    if (
        df[numeric_columns]
        .isin([np.inf, -np.inf])
        .any()
        .any()
    ):
        raise ValueError(
            "Infinite numeric values detected."
        )

    # Targets must never be present in feature columns.
    target_columns = set(
        FORECAST_HORIZONS.keys()
    )

    feature_columns = set(
        df.columns
    )

    accidental_overlap = (
        target_columns
        & feature_columns
    )

    if accidental_overlap != target_columns:
        raise ValueError(
            "Forecast target columns missing."
        )

    logger.info(
        "Feature dataset validation passed."
    )


# ============================================================================
# REPORT
# ============================================================================

def build_report(
    df: pd.DataFrame,
) -> dict:
    """Build feature engineering audit report."""

    target_columns = list(
        FORECAST_HORIZONS.keys()
    )

    excluded = {
        "timestamp",
        "city",
        *target_columns,
    }

    feature_columns = [
        column
        for column in df.columns
        if column not in excluded
        and pd.api.types.is_numeric_dtype(
            df[column]
        )
    ]

    target_coverage = {
        column: float(
            df[column].notna().mean()
            * 100
        )
        for column in target_columns
    }

    return {
        "dataset": "Pearls AQI Predictor",
        "feature_engineering_version": "2.0",
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "target_columns": target_columns,
        "target_coverage_percent": (
            target_coverage
        ),
        "timestamp_start": str(
            df["timestamp"].min()
        ),
        "timestamp_end": str(
            df["timestamp"].max()
        ),
        "missing_values": {
            str(key): int(value)
            for key, value in (
                df.isna().sum().items()
            )
        },
    }


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    """Execute time-aware feature engineering."""

    logger.info("=" * 72)
    logger.info(
        "PEARLS AQI PREDICTOR"
    )
    logger.info(
        "TIME-AWARE FEATURE ENGINEERING"
    )
    logger.info("=" * 72)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_canonical_dataset()

    logger.info(
        "Creating temporal features."
    )
    df = add_time_features(df)

    logger.info(
        "Creating exact timestamp-based PM2.5 lags."
    )
    df = add_pm25_lags(df)

    logger.info(
        "Creating time-based rolling statistics."
    )
    df = add_rolling_features(df)

    logger.info(
        "Creating PM2.5 trend features."
    )
    df = add_pm25_trends(df)

    logger.info(
        "Creating weather features."
    )
    df = add_weather_features(df)

    logger.info(
        "Creating measurement quality features."
    )
    df = add_quality_features(df)

    logger.info(
        "Creating exact 24h/48h/72h future targets."
    )
    df = add_future_targets(df)

    validate_feature_dataset(
        df
    )

    report = build_report(
        df
    )

    df.to_parquet(
        OUTPUT_PARQUET,
        index=False,
    )

    df.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    REPORT_FILE.write_text(
        json.dumps(
            report,
            indent=4,
        ),
        encoding="utf-8",
    )

    logger.info(
        "Feature dataset saved: %s",
        OUTPUT_PARQUET,
    )

    logger.info(
        "Feature CSV saved: %s",
        OUTPUT_CSV,
    )

    logger.info(
        "Feature report saved: %s",
        REPORT_FILE,
    )

    logger.info(
        "Feature dataset shape: %s",
        df.shape,
    )

    logger.info(
        "Feature count: %d",
        report["feature_count"],
    )

    logger.info(
        "Target coverage: %s",
        report["target_coverage_percent"],
    )

    logger.info("=" * 72)
    logger.info(
        "TIME-AWARE FEATURE ENGINEERING COMPLETED."
    )
    logger.info("=" * 72)


if __name__ == "__main__":
    main()












