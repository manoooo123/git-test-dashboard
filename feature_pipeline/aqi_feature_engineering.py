"""
Pearls AQI Predictor
====================

Production-grade daily AQI feature engineering pipeline.

Purpose
-------
Build a leakage-safe daily forecasting dataset for:

    Day +1 AQI
    Day +2 AQI
    Day +3 AQI

Inputs
------
1. data/processed/daily_lahore_aqi.parquet
2. data/processed/canonical_lahore_hourly.parquet

Outputs
-------
1. data/processed/aqi_model_features.parquet
2. data/processed/aqi_model_features.csv
3. reports/data_quality/aqi_feature_engineering_report.json

Important
---------
- Uses real observations only.
- Does not fabricate missing days.
- Does not interpolate missing AQI.
- Does not use future information in predictor features.
- Future AQI is stored only in target columns.
- Uses exact calendar dates rather than DataFrame row positions.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DAILY_AQI_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "daily_lahore_aqi.parquet"
)

HOURLY_CANONICAL_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "canonical_lahore_hourly.parquet"
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
    / "aqi_model_features.parquet"
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "aqi_model_features.csv"
)

REPORT_FILE = (
    REPORT_DIR
    / "aqi_feature_engineering_report.json"
)


# ============================================================================
# CONFIGURATION
# ============================================================================

TARGET_COLUMNS = (
    "target_aqi_day_1",
    "target_aqi_day_2",
    "target_aqi_day_3",
)

AQI_LAG_DAYS = (
    1,
    2,
    3,
    7,
)

AQI_ROLLING_WINDOWS = (
    3,
    7,
)

PM25_ROLLING_WINDOWS = (
    3,
    7,
)


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
# LOAD DAILY AQI
# ============================================================================

def load_daily_aqi() -> pd.DataFrame:
    """Load calculated daily Lahore AQI observations."""

    if not DAILY_AQI_FILE.exists():
        raise FileNotFoundError(
            f"Daily AQI dataset not found: {DAILY_AQI_FILE}"
        )

    logger.info(
        "Loading daily AQI dataset: %s",
        DAILY_AQI_FILE,
    )

    df = pd.read_parquet(
        DAILY_AQI_FILE
    )

    if df.empty:
        raise ValueError(
            "Daily AQI dataset is empty."
        )

    required_columns = {
        "date",
        "city",
        "pm2_5_24h_mean",
        "aqi",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            "Daily AQI dataset missing columns: "
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

    df["pm2_5_24h_mean"] = pd.to_numeric(
        df["pm2_5_24h_mean"],
        errors="coerce",
    )

    df.dropna(
        subset=[
            "date",
            "aqi",
            "pm2_5_24h_mean",
        ],
        inplace=True,
    )

    df = (
        df
        .sort_values("date")
        .drop_duplicates("date")
        .reset_index(drop=True)
    )

    logger.info(
        "Daily AQI observations: %d",
        len(df),
    )

    logger.info(
        "AQI period: %s → %s",
        df["date"].min(),
        df["date"].max(),
    )

    return df


# ============================================================================
# LOAD HOURLY WEATHER/CANONICAL DATA
# ============================================================================

def load_canonical_data() -> pd.DataFrame:
    """Load canonical hourly Lahore observations."""

    if not HOURLY_CANONICAL_FILE.exists():
        raise FileNotFoundError(
            "Canonical hourly dataset not found: "
            f"{HOURLY_CANONICAL_FILE}"
        )

    logger.info(
        "Loading canonical hourly dataset."
    )

    df = pd.read_parquet(
        HOURLY_CANONICAL_FILE
    )

    if df.empty:
        raise ValueError(
            "Canonical hourly dataset is empty."
        )

    required_columns = {
        "timestamp",
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
            "Canonical hourly dataset missing columns: "
            + ", ".join(sorted(missing))
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
        errors="coerce",
    )

    df.dropna(
        subset=["timestamp"],
        inplace=True,
    )

    logger.info(
        "Hourly canonical observations: %d",
        len(df),
    )

    return df


# ============================================================================
# DAILY WEATHER
# ============================================================================

def build_daily_weather(
    hourly: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate real hourly weather observations into daily features.

    Lahore local calendar dates are used.
    """

    data = hourly.copy()

    local_timestamp = (
        data["timestamp"]
        .dt.tz_convert("Asia/Karachi")
    )

    data["date"] = (
        local_timestamp.dt.normalize()
        .dt.tz_localize(None)
    )

    # ------------------------------------------------------------------------
    # Wind direction is circular.
    # ------------------------------------------------------------------------

    wind_radians = np.deg2rad(
        data["wind_direction"]
    )

    data["wind_direction_sin"] = np.sin(
        wind_radians
    )

    data["wind_direction_cos"] = np.cos(
        wind_radians
    )

    # ------------------------------------------------------------------------
    # Daily weather aggregation
    # ------------------------------------------------------------------------

    daily_weather = (
        data
        .groupby("date", as_index=False)
        .agg(
            temperature_mean=(
                "temperature",
                "mean",
            ),
            temperature_min=(
                "temperature",
                "min",
            ),
            temperature_max=(
                "temperature",
                "max",
            ),
            humidity_mean=(
                "humidity",
                "mean",
            ),
            humidity_min=(
                "humidity",
                "min",
            ),
            humidity_max=(
                "humidity",
                "max",
            ),
            pressure_mean=(
                "pressure",
                "mean",
            ),
            clouds_mean=(
                "clouds",
                "mean",
            ),
            wind_speed_mean=(
                "wind_speed",
                "mean",
            ),
            wind_speed_max=(
                "wind_speed",
                "max",
            ),
            wind_direction_sin=(
                "wind_direction_sin",
                "mean",
            ),
            wind_direction_cos=(
                "wind_direction_cos",
                "mean",
            ),
            precipitation_total=(
                "precipitation",
                "sum",
            ),
            weather_observations=(
                "temperature",
                "count",
            ),
        )
    )

    logger.info(
        "Daily weather observations: %d",
        len(daily_weather),
    )

    return daily_weather


# ============================================================================
# CALENDAR FEATURES
# ============================================================================

def add_calendar_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Add leakage-safe calendar features."""

    result = df.copy()

    timestamp = result["date"]

    result["day_of_week"] = (
        timestamp.dt.dayofweek
    )

    result["day_of_year"] = (
        timestamp.dt.dayofyear
    )

    result["month"] = (
        timestamp.dt.month
    )

    result["quarter"] = (
        timestamp.dt.quarter
    )

    result["week_of_year"] = (
        timestamp.dt.isocalendar()
        .week
        .astype(int)
    )

    result["is_weekend"] = (
        timestamp.dt.dayofweek >= 5
    ).astype(int)

    # Cyclical encoding.

    result["day_of_year_sin"] = np.sin(
        2
        * np.pi
        * result["day_of_year"]
        / 365.25
    )

    result["day_of_year_cos"] = np.cos(
        2
        * np.pi
        * result["day_of_year"]
        / 365.25
    )

    result["month_sin"] = np.sin(
        2
        * np.pi
        * result["month"]
        / 12
    )

    result["month_cos"] = np.cos(
        2
        * np.pi
        * result["month"]
        / 12
    )

    return result


# ============================================================================
# EXACT DAILY LAGS
# ============================================================================

def add_exact_daily_lag(
    df: pd.DataFrame,
    source_column: str,
    lag_days: int,
    output_name: str,
) -> pd.DataFrame:
    """
    Add a date-based lag.

    Example:
        At 2025-10-10,
        lag_7d = value from 2025-10-03.

    Missing calendar dates remain NaN.
    """

    result = df.copy()

    lookup = (
        df[
            [
                "date",
                source_column,
            ]
        ]
        .copy()
    )

    lookup["date"] = (
        lookup["date"]
        + pd.Timedelta(
            days=lag_days
        )
    )

    lookup.rename(
        columns={
            source_column: output_name
        },
        inplace=True,
    )

    result = result.merge(
        lookup,
        on="date",
        how="left",
        validate="one_to_one",
    )

    return result


# ============================================================================
# AQI LAG FEATURES
# ============================================================================

def add_aqi_lag_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create historical AQI lag features.

    Only past dates are used.
    """

    result = df.copy()

    for lag_days in AQI_LAG_DAYS:

        result = add_exact_daily_lag(
            result,
            source_column="aqi",
            lag_days=lag_days,
            output_name=(
                f"aqi_lag_{lag_days}d"
            ),
        )

    return result


# ============================================================================
# PM2.5 LAG FEATURES
# ============================================================================

def add_pm25_lag_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Create historical daily PM2.5 lag features."""

    result = df.copy()

    for lag_days in AQI_LAG_DAYS:

        result = add_exact_daily_lag(
            result,
            source_column="pm2_5_24h_mean",
            lag_days=lag_days,
            output_name=(
                f"pm2_5_lag_{lag_days}d"
            ),
        )

    return result


# ============================================================================
# ROLLING AQI
# ============================================================================

def add_aqi_rolling_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate historical rolling AQI statistics.

    Shift by one day before rolling so current AQI
    cannot leak into its own historical features.
    """

    result = df.copy()

    historical_aqi = (
        result
        .set_index("date")["aqi"]
        .shift(1)
    )

    for window in AQI_ROLLING_WINDOWS:

        rolling = historical_aqi.rolling(
            window=window,
            min_periods=2,
        )

        result[
            f"aqi_rolling_mean_{window}d"
        ] = rolling.mean().values

        result[
            f"aqi_rolling_std_{window}d"
        ] = rolling.std().values

        result[
            f"aqi_rolling_min_{window}d"
        ] = rolling.min().values

        result[
            f"aqi_rolling_max_{window}d"
        ] = rolling.max().values

    return result


# ============================================================================
# ROLLING PM2.5
# ============================================================================

def add_pm25_rolling_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate historical PM2.5 rolling statistics.

    Current day's PM2.5 is excluded.
    """

    result = df.copy()

    historical_pm25 = (
        result
        .set_index("date")[
            "pm2_5_24h_mean"
        ]
        .shift(1)
    )

    for window in PM25_ROLLING_WINDOWS:

        rolling = historical_pm25.rolling(
            window=window,
            min_periods=2,
        )

        result[
            f"pm2_5_rolling_mean_{window}d"
        ] = rolling.mean().values

        result[
            f"pm2_5_rolling_std_{window}d"
        ] = rolling.std().values

        result[
            f"pm2_5_rolling_max_{window}d"
        ] = rolling.max().values

    return result


# ============================================================================
# AQI TREND FEATURES
# ============================================================================

def add_trend_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Create historical AQI trend features."""

    result = df.copy()

    result["aqi_change_1d"] = (
        result["aqi"]
        - result["aqi_lag_1d"]
    )

    result["aqi_change_3d"] = (
        result["aqi"]
        - result["aqi_lag_3d"]
    )

    result["aqi_change_7d"] = (
        result["aqi"]
        - result["aqi_lag_7d"]
    )

    result["pm2_5_change_1d"] = (
        result["pm2_5_24h_mean"]
        - result["pm2_5_lag_1d"]
    )

    result["pm2_5_change_3d"] = (
        result["pm2_5_24h_mean"]
        - result["pm2_5_lag_3d"]
    )

    result["pm2_5_change_7d"] = (
        result["pm2_5_24h_mean"]
        - result["pm2_5_lag_7d"]
    )

    result["aqi_pct_change_1d"] = (
        (
            result["aqi"]
            / result["aqi_lag_1d"]
        )
        - 1.0
    ).replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return result


# ============================================================================
# WEATHER TREND FEATURES
# ============================================================================

def add_weather_history(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add historical weather features using previous-day values only.
    """

    result = df.copy()

    weather_columns = [
        "temperature_mean",
        "humidity_mean",
        "pressure_mean",
        "wind_speed_mean",
        "precipitation_total",
    ]

    for column in weather_columns:

        result = add_exact_daily_lag(
            result,
            source_column=column,
            lag_days=1,
            output_name=(
                f"{column}_lag_1d"
            ),
        )

    result["temperature_change_1d"] = (
        result["temperature_mean"]
        - result["temperature_mean_lag_1d"]
    )

    result["humidity_change_1d"] = (
        result["humidity_mean"]
        - result["humidity_mean_lag_1d"]
    )

    result["pressure_change_1d"] = (
        result["pressure_mean"]
        - result["pressure_mean_lag_1d"]
    )

    result["wind_speed_change_1d"] = (
        result["wind_speed_mean"]
        - result["wind_speed_mean_lag_1d"]
    )

    return result


# ============================================================================
# FUTURE AQI TARGETS
# ============================================================================

def add_future_targets(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create exact calendar-day AQI targets.

    No row-shifting is used because real-world daily observations
    contain missing dates.
    """

    result = df.copy()

    source = (
        df[
            [
                "date",
                "aqi",
            ]
        ]
        .copy()
    )

    for horizon in (
        1,
        2,
        3,
    ):

        lookup = source.copy()

        lookup["date"] = (
            lookup["date"]
            - pd.Timedelta(
                days=horizon
            )
        )

        lookup.rename(
            columns={
                "aqi": (
                    f"target_aqi_day_{horizon}"
                )
            },
            inplace=True,
        )

        result = result.merge(
            lookup[
                [
                    "date",
                    f"target_aqi_day_{horizon}",
                ]
            ],
            on="date",
            how="left",
            validate="one_to_one",
        )

    return result


# ============================================================================
# FEATURE VALIDATION
# ============================================================================

def validate_features(
    df: pd.DataFrame,
) -> None:
    """Validate AQI forecasting feature dataset."""

    if df.empty:
        raise ValueError(
            "AQI feature dataset is empty."
        )

    if df["date"].duplicated().any():
        raise ValueError(
            "Duplicate dates detected."
        )

    if not df["date"].is_monotonic_increasing:
        raise ValueError(
            "Dates are not sorted."
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

    required_targets = set(
        TARGET_COLUMNS
    )

    missing_targets = (
        required_targets
        - set(df.columns)
    )

    if missing_targets:
        raise ValueError(
            "Missing AQI target columns: "
            + ", ".join(
                sorted(missing_targets)
            )
        )

    # Future target columns must never be considered input features.
    logger.info(
        "AQI feature validation passed."
    )


# ============================================================================
# REPORT
# ============================================================================

def build_report(
    df: pd.DataFrame,
) -> dict:
    """Build an auditable feature-engineering report."""

    excluded = {
        "date",
        "city",
        "aqi",
        "aqi_rounded",
        "aqi_category",
        *TARGET_COLUMNS,
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
        target: float(
            df[target]
            .notna()
            .mean()
            * 100
        )
        for target in TARGET_COLUMNS
    }

    return {
        "project": "Pearls AQI Predictor",
        "dataset": "Daily AQI forecasting",
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "target_columns": list(
            TARGET_COLUMNS
        ),
        "target_coverage_percent": (
            target_coverage
        ),
        "date_start": str(
            df["date"].min()
        ),
        "date_end": str(
            df["date"].max()
        ),
        "missing_values": {
            str(column): int(value)
            for column, value in (
                df.isna().sum().items()
            )
        },
    }


# ============================================================================
# SAVE OUTPUTS
# ============================================================================

def save_outputs(
    df: pd.DataFrame,
    report: dict,
) -> None:
    """Persist AQI forecasting features."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
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
        "AQI feature Parquet saved: %s",
        OUTPUT_PARQUET,
    )

    logger.info(
        "AQI feature CSV saved: %s",
        OUTPUT_CSV,
    )

    logger.info(
        "AQI feature report saved: %s",
        REPORT_FILE,
    )


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main() -> None:
    """Execute daily AQI feature engineering."""

    logger.info("=" * 72)
    logger.info(
        "PEARLS AQI PREDICTOR"
    )
    logger.info(
        "DAILY AQI FEATURE ENGINEERING"
    )
    logger.info("=" * 72)

    daily_aqi = load_daily_aqi()

    hourly = load_canonical_data()

    logger.info(
        "Building daily weather aggregates."
    )

    daily_weather = build_daily_weather(
        hourly
    )

    logger.info(
        "Merging daily AQI and daily weather."
    )

    df = daily_aqi.merge(
        daily_weather,
        on="date",
        how="left",
        validate="one_to_one",
    )

    if df.empty:
        raise ValueError(
            "AQI/weather daily merge produced no rows."
        )

    df = (
        df
        .sort_values("date")
        .reset_index(drop=True)
    )

    logger.info(
        "Merged daily dataset: %d rows",
        len(df),
    )

    logger.info(
        "Creating calendar features."
    )

    df = add_calendar_features(
        df
    )

    logger.info(
        "Creating historical AQI lags."
    )

    df = add_aqi_lag_features(
        df
    )

    logger.info(
        "Creating historical PM2.5 lags."
    )

    df = add_pm25_lag_features(
        df
    )

    logger.info(
        "Creating rolling AQI features."
    )

    df = add_aqi_rolling_features(
        df
    )

    logger.info(
        "Creating rolling PM2.5 features."
    )

    df = add_pm25_rolling_features(
        df
    )

    logger.info(
        "Creating AQI and PM2.5 trends."
    )

    df = add_trend_features(
        df
    )

    logger.info(
        "Creating weather history features."
    )

    df = add_weather_history(
        df
    )

    logger.info(
        "Creating Day +1 / +2 / +3 AQI targets."
    )

    df = add_future_targets(
        df
    )

    validate_features(
        df
    )

    report = build_report(
        df
    )

    save_outputs(
        df,
        report,
    )

    logger.info(
        "AQI feature dataset shape: %s",
        df.shape,
    )

    logger.info(
        "AQI feature count: %d",
        report["feature_count"],
    )

    logger.info(
        "Target coverage: %s",
        report[
            "target_coverage_percent"
        ],
    )

    logger.info("=" * 72)
    logger.info(
        "DAILY AQI FEATURE ENGINEERING COMPLETED."
    )
    logger.info("=" * 72)

    print(
        "\nAQI feature preview:"
    )

    print(
        df.head(10)
        .to_string(index=False)
    )


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()