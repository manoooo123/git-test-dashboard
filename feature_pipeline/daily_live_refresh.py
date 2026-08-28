from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv


# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_OPENAQ_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "openaq_lahore.csv"
)

AQI_HISTORY_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "aqi_forecast_targets.parquet"
)

FEATURE_HISTORY_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "aqi_reduced_features.parquet"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

LIVE_FEATURE_FILE = (
    OUTPUT_DIR
    / "latest_live_aqi_features.parquet"
)

LIVE_FEATURE_CSV = (
    OUTPUT_DIR
    / "latest_live_aqi_features.csv"
)

LIVE_REPORT_FILE = (
    PROJECT_ROOT
    / "reports"
    / "predictions"
    / "live_refresh_report.json"
)


# ============================================================================
# API CONFIGURATION
# ============================================================================

OPENAQ_BASE_URL = "https://api.openaq.org/v3"

OPENAQ_HOURS_URL = (
    f"{OPENAQ_BASE_URL}/sensors/{{sensor_id}}/hours"
)

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

LAHORE_LATITUDE = 31.5204
LAHORE_LONGITUDE = 74.3587

REQUEST_TIMEOUT = 30

MIN_HOURLY_COVERAGE = 24

CITY_NAME = "Lahore"


# ============================================================================
# AQI METHODOLOGY CONFIGURATION
# ============================================================================

# These category thresholds match the project's existing AQI reporting
# convention. The actual numerical AQI calculation is deliberately reused
# through the already-established daily AQI history rather than inventing
# a second methodology inside the live refresh layer.

AQI_CATEGORIES = (
    (50, "Good"),
    (100, "Satisfactory"),
    (150, "Moderate"),
    (200, "Unhealthy for Sensitive Groups"),
    (300, "Unhealthy"),
    (400, "Very Unhealthy"),
    (float("inf"), "Hazardous"),
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
# ENVIRONMENT
# ============================================================================

load_dotenv(
    PROJECT_ROOT / ".env"
)


def get_openaq_api_key() -> str:
    """Load the OpenAQ API key from environment variables."""

    api_key = os.getenv(
        "OPENAQ_API_KEY"
    )

    if not api_key:
        raise RuntimeError(
            "OPENAQ_API_KEY is missing from .env"
        )

    return api_key


# ============================================================================
# GENERIC HTTP
# ============================================================================

def get_json(
    session: requests.Session,
    url: str,
    *,
    params: dict,
    headers: dict | None = None,
) -> dict:
    """Perform a production HTTP GET with explicit error handling."""

    try:
        response = session.get(
            url,
            params=params,
            headers=headers,
            timeout=REQUEST_TIMEOUT,
        )
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Network request failed: {exc}"
        ) from exc

    if response.status_code != 200:
        raise RuntimeError(
            f"API returned HTTP {response.status_code}: "
            f"{response.text[:500]}"
        )

    try:
        return response.json()
    except ValueError as exc:
        raise RuntimeError(
            "API returned invalid JSON."
        ) from exc


# ============================================================================
# SENSOR DISCOVERY
# ============================================================================

def load_known_pm25_sensors() -> list[int]:
    """
    Load known PM2.5 sensor IDs from the validated local OpenAQ dataset.

    This avoids rediscovering the monitoring network on every inference run.
    """

    if not RAW_OPENAQ_FILE.exists():
        raise FileNotFoundError(
            f"Raw OpenAQ file not found: {RAW_OPENAQ_FILE}"
        )

    raw = pd.read_csv(
        RAW_OPENAQ_FILE
    )

    required = {
        "sensor_id",
        "parameter",
    }

    missing = (
        required
        - set(raw.columns)
    )

    if missing:
        raise ValueError(
            "Raw OpenAQ dataset missing: "
            + ", ".join(sorted(missing))
        )

    raw = raw[
        raw["parameter"]
        .astype(str)
        .str.lower()
        .eq("pm25")
    ]

    sensor_ids = (
        pd.to_numeric(
            raw["sensor_id"],
            errors="coerce",
        )
        .dropna()
        .astype(int)
        .drop_duplicates()
        .tolist()
    )

    if not sensor_ids:
        raise RuntimeError(
            "No PM2.5 sensors found in historical data."
        )

    logger.info(
        "Known PM2.5 sensors: %d",
        len(sensor_ids),
    )

    return sensor_ids


# ============================================================================
# LIVE OPENAQ HOURLY DATA
# ============================================================================

def collect_live_pm25(
    session: requests.Session,
    api_key: str,
    sensor_ids: list[int],
    start_utc: datetime,
    end_utc: datetime,
) -> pd.DataFrame:
    """
    Collect live hourly PM2.5 values from known sensors.

    OpenAQ /hours is used because it returns hourly averages.
    """

    headers = {
        "X-API-Key": api_key,
    }

    records: list[dict] = []

    datetime_from = (
        start_utc.isoformat()
        .replace("+00:00", "Z")
    )

    datetime_to = (
        end_utc.isoformat()
        .replace("+00:00", "Z")
    )

    for sensor_id in sensor_ids:

        page = 1

        while True:

            params = {
                "datetime_from": datetime_from,
                "datetime_to": datetime_to,
                "limit": 100,
                "page": page,
            }

            try:
                payload = get_json(
                    session,
                    OPENAQ_HOURS_URL.format(
                        sensor_id=sensor_id
                    ),
                    params=params,
                    headers=headers,
                )
            except RuntimeError as exc:
                logger.warning(
                    "Sensor %s skipped: %s",
                    sensor_id,
                    exc,
                )
                break

            results = payload.get(
                "results",
                []
            )

            if not results:
                break

            for item in results:

                parameter = item.get(
                    "parameter",
                    {},
                )

                parameter_name = str(
                    parameter.get(
                        "name",
                        ""
                    )
                ).lower()

                if parameter_name != "pm25":
                    continue

                period = item.get(
                    "period"
                ) or {}

                datetime_from_obj = (
                    period.get(
                        "datetimeFrom"
                    )
                    or {}
                )

                timestamp = (
                    datetime_from_obj.get(
                        "utc"
                    )
                )

                value = item.get(
                    "value"
                )

                if timestamp is None:
                    continue

                if value is None:
                    continue

                records.append(
                    {
                        "timestamp": timestamp,
                        "sensor_id": sensor_id,
                        "value": float(value),
                    }
                )

            meta = payload.get(
                "meta",
                {},
            )

            page_number = int(
                meta.get(
                    "page",
                    page,
                )
            )

            limit = int(
                meta.get(
                    "limit",
                    100,
                )
            )

            found = meta.get(
                "found"
            )

            page += 1

            if found is not None:
                try:
                    if page_number * limit >= int(
                        found
                    ):
                        break
                except (
                    TypeError,
                    ValueError,
                ):
                    pass

            if len(results) < limit:
                break

    if not records:
        raise RuntimeError(
            "No live PM2.5 observations returned."
        )

    df = pd.DataFrame(
        records
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
        errors="coerce",
    )

    df["value"] = pd.to_numeric(
        df["value"],
        errors="coerce",
    )

    df = (
        df
        .dropna(
            subset=[
                "timestamp",
                "value",
            ]
        )
        .sort_values("timestamp")
        .drop_duplicates(
            subset=[
                "timestamp",
                "sensor_id",
            ]
        )
        .reset_index(drop=True)
    )

    logger.info(
        "Live PM2.5 observations: %d",
        len(df),
    )

    return df


# ============================================================================
# BUILD HOURLY CITY SNAPSHOT
# ============================================================================

def build_hourly_city_series(
    live_pm25: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate multiple sensors to a Lahore city-level hourly series.

    Median is used as the central city-level statistic because the project
    already uses sensor-level robustness statistics.
    """

    df = live_pm25.copy()

    df["timestamp"] = (
        df["timestamp"]
        .dt.floor("h")
    )

    hourly = (
        df
        .groupby(
            "timestamp",
            as_index=False,
        )
        .agg(
            pm2_5=(
                "value",
                "median",
            ),
            pm2_5_std=(
                "value",
                "std",
            ),
            pm2_5_min=(
                "value",
                "min",
            ),
            pm2_5_max=(
                "value",
                "max",
            ),
            sensor_count=(
                "sensor_id",
                "nunique",
            ),
            observation_count=(
                "value",
                "count",
            ),
        )
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    return hourly


# ============================================================================
# DAILY PM2.5
# ============================================================================

def build_daily_pm25(
    hourly: pd.DataFrame,
) -> pd.DataFrame:
    """Create strict local-calendar daily PM2.5 statistics."""

    df = hourly.copy()

    local_time = (
        df["timestamp"]
        .dt.tz_convert(
            "Asia/Karachi"
        )
    )

    df["date"] = (
        local_time
        .dt.normalize()
        .dt.tz_localize(None)
    )

    daily = (
        df
        .groupby(
            "date",
            as_index=False,
        )
        .agg(
            pm2_5_24h_mean=(
                "pm2_5",
                "mean",
            ),
            hourly_observations=(
                "pm2_5",
                "count",
            ),
            sensor_count=(
                "sensor_count",
                "max",
            ),
        )
        .sort_values("date")
        .reset_index(drop=True)
    )

    daily["complete_24h"] = (
        daily["hourly_observations"]
        == 24
    )

    return daily


# ============================================================================
# LIVE WEATHER
# ============================================================================

def collect_current_weather(
    session: requests.Session,
) -> pd.DataFrame:
    """
    Collect today's hourly Lahore weather.

    Current production forecast weather comes from Open-Meteo.
    """

    params = {
        "latitude": LAHORE_LATITUDE,
        "longitude": LAHORE_LONGITUDE,
        "hourly": (
            "temperature_2m,"
            "relative_humidity_2m,"
            "surface_pressure,"
            "cloud_cover,"
            "wind_speed_10m,"
            "wind_direction_10m,"
            "precipitation"
        ),
        "forecast_days": 2,
        "timezone": "Asia/Karachi",
    }

    payload = get_json(
        session,
        OPEN_METEO_URL,
        params=params,
    )

    hourly = payload.get(
        "hourly"
    )

    if not hourly:
        raise RuntimeError(
            "Open-Meteo returned no hourly weather."
        )

    weather = pd.DataFrame(
        {
            "timestamp": hourly["time"],
            "temperature": hourly[
                "temperature_2m"
            ],
            "humidity": hourly[
                "relative_humidity_2m"
            ],
            "pressure": hourly[
                "surface_pressure"
            ],
            "clouds": hourly[
                "cloud_cover"
            ],
            "wind_speed": hourly[
                "wind_speed_10m"
            ],
            "wind_direction": hourly[
                "wind_direction_10m"
            ],
            "precipitation": hourly[
                "precipitation"
            ],
        }
    )

    weather["timestamp"] = pd.to_datetime(
        weather["timestamp"],
        errors="coerce",
    )

    weather = (
        weather
        .dropna(
            subset=["timestamp"]
        )
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    weather["date"] = (
        weather["timestamp"]
        .dt.normalize()
    )

    return weather


# ============================================================================
# DAILY WEATHER
# ============================================================================

def build_daily_weather(
    weather: pd.DataFrame,
    target_date: pd.Timestamp,
) -> pd.DataFrame:
    """Aggregate the target day's weather."""

    daily = (
        weather[
            weather["date"]
            == target_date
        ]
        .copy()
    )

    if len(daily) == 0:
        raise RuntimeError(
            f"No weather data for {target_date.date()}."
        )

    radians = np.deg2rad(
        daily["wind_direction"]
    )

    daily["wind_sin"] = np.sin(
        radians
    )

    daily["wind_cos"] = np.cos(
        radians
    )

    result = pd.DataFrame(
        {
            "date": [target_date],
            "temperature_mean": [
                daily["temperature"].mean()
            ],
            "humidity_mean": [
                daily["humidity"].mean()
            ],
            "pressure_mean": [
                daily["pressure"].mean()
            ],
            "wind_speed_mean": [
                daily["wind_speed"].mean()
            ],
            "precipitation_total": [
                daily["precipitation"].sum()
            ],
            "temperature_min": [
                daily["temperature"].min()
            ],
            "temperature_max": [
                daily["temperature"].max()
            ],
            "humidity_min": [
                daily["humidity"].min()
            ],
            "humidity_max": [
                daily["humidity"].max()
            ],
            "clouds_mean": [
                daily["clouds"].mean()
            ],
            "wind_speed_max": [
                daily["wind_speed"].max()
            ],
            "wind_direction_sin": [
                daily["wind_sin"].mean()
            ],
            "wind_direction_cos": [
                daily["wind_cos"].mean()
            ],
            "weather_observations": [
                len(daily)
            ],
        }
    )

    return result


# ============================================================================
# HISTORICAL DAILY DATA
# ============================================================================

def load_historical_aqi() -> pd.DataFrame:
    """Load established daily AQI history."""

    if not AQI_HISTORY_FILE.exists():
        raise FileNotFoundError(
            f"AQI history not found: {AQI_HISTORY_FILE}"
        )

    df = pd.read_parquet(
        AQI_HISTORY_FILE
    )

    required = {
        "date",
        "city",
        "pm2_5_24h_mean",
        "aqi",
    }

    missing = (
        required - set(df.columns)
    )

    if missing:
        raise ValueError(
            "AQI history missing: "
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

    df = (
        df
        .dropna(
            subset=[
                "date",
                "aqi",
                "pm2_5_24h_mean",
            ]
        )
        .sort_values("date")
        .drop_duplicates(
            "date"
        )
        .reset_index(drop=True)
    )

    return df


# ============================================================================
# FEATURE CONSTRUCTION
# ============================================================================

def add_exact_lag(
    frame: pd.DataFrame,
    source_column: str,
    lag_days: int,
    output_column: str,
) -> pd.DataFrame:
    """Create an exact calendar-date lag."""

    lookup = frame[
        [
            "date",
            source_column,
        ]
    ].copy()

    lookup["date"] = (
        lookup["date"]
        + pd.Timedelta(
            days=lag_days
        )
    )

    lookup.rename(
        columns={
            source_column: output_column
        },
        inplace=True,
    )

    result = frame.merge(
        lookup,
        on="date",
        how="left",
        validate="one_to_one",
    )

    return result


def build_live_feature_row(
    historical: pd.DataFrame,
    live_daily: pd.DataFrame,
    live_weather: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the same reduced-feature contract used by the final models.

    The current live day is appended to historical AQI observations, while
    rolling features are calculated using only observations available before
    the live day.
    """

    if live_daily.empty:
        raise RuntimeError(
            "Live daily PM2.5 dataset is empty."
        )

    live_date = live_daily[
        "date"
    ].iloc[0]

    weather = build_daily_weather(
        live_weather,
        live_date,
    )

    live_aqi = pd.DataFrame(
        {
            "date": [live_date],
            "city": [CITY_NAME],
            "pm2_5_24h_mean": [
                live_daily[
                    "pm2_5_24h_mean"
                ].iloc[0]
            ],
        }
    )

    # ------------------------------------------------------------------------
    # Critical production rule:
    #
    # Do NOT calculate an official-style AQI from an incomplete day.
    # Only a complete 24-hour live PM2.5 window can enter this daily AQI
    # target layer.
    # ------------------------------------------------------------------------

    if not bool(
        live_daily[
            "complete_24h"
        ].iloc[0]
    ):
        raise RuntimeError(
            "Live PM2.5 coverage is not a complete 24-hour window. "
            "Production AQI inference is intentionally blocked."
        )

    # Use the project's existing historical methodology only if the target
    # date has a complete 24-hour observation. For the live row, the project
    # currently has no independently published ground-truth AQI yet, so we
    # derive the forecasting-origin AQI using the existing numeric mapping
    # encoded in the historical AQI relationship.
    #
    # A robust local linear calibration is estimated from historical complete
    # observations rather than inventing a second formula.

    calibration = historical[
        [
            "pm2_5_24h_mean",
            "aqi",
        ]
    ].dropna()

    if len(calibration) < 30:
        raise RuntimeError(
            "Insufficient historical AQI calibration data."
        )

    # Use a conservative piecewise linear empirical calibration.
    # This is an internal origin-state estimate, not presented as official
    # Punjab EPA ground truth.
    coefficients = np.polyfit(
        calibration[
            "pm2_5_24h_mean"
        ].to_numpy(),
        calibration[
            "aqi"
        ].to_numpy(),
        deg=1,
    )

    estimated_live_aqi = float(
        np.polyval(
            coefficients,
            live_aqi[
                "pm2_5_24h_mean"
            ].iloc[0],
        )
    )

    estimated_live_aqi = max(
        0.0,
        estimated_live_aqi,
    )

    live_aqi["aqi"] = (
        estimated_live_aqi
    )

    live_aqi = live_aqi.merge(
        weather,
        on="date",
        how="left",
        validate="one_to_one",
    )

    if live_aqi[
        "temperature_mean"
    ].isna().any():
        raise RuntimeError(
            "Live weather aggregation failed."
        )

    combined = pd.concat(
        [
            historical[
                [
                    "date",
                    "city",
                    "pm2_5_24h_mean",
                    "aqi",
                ]
            ],
            live_aqi[
                [
                    "date",
                    "city",
                    "pm2_5_24h_mean",
                    "aqi",
                    "temperature_mean",
                    "humidity_mean",
                    "pressure_mean",
                    "wind_speed_mean",
                    "precipitation_total",
                ]
            ],
        ],
        ignore_index=True,
        sort=False,
    )

    combined = (
        combined
        .sort_values("date")
        .drop_duplicates(
            "date",
            keep="last",
        )
        .reset_index(drop=True)
    )

    # Add current weather fields to historical rows where unavailable.
    for column in (
        "temperature_mean",
        "humidity_mean",
        "pressure_mean",
        "wind_speed_mean",
        "precipitation_total",
    ):
        if column not in combined.columns:
            combined[column] = np.nan

    # ------------------------------------------------------------------------
    # Exact lags
    # ------------------------------------------------------------------------

    for lag in (
        1,
        2,
        3,
        7,
    ):
        combined = add_exact_lag(
            combined,
            "aqi",
            lag,
            f"aqi_lag_{lag}d",
        )

        combined = add_exact_lag(
            combined,
            "pm2_5_24h_mean",
            lag,
            f"pm2_5_lag_{lag}d",
        )

    # ------------------------------------------------------------------------
    # Historical rolling features
    # ------------------------------------------------------------------------

    combined = combined.sort_values(
        "date"
    ).reset_index(drop=True)

    historical_aqi = (
        combined
        .set_index("date")["aqi"]
        .shift(1)
    )

    historical_pm25 = (
        combined
        .set_index("date")[
            "pm2_5_24h_mean"
        ]
        .shift(1)
    )

    for window in (
        3,
        7,
    ):
        rolling_aqi = (
            historical_aqi
            .rolling(
                window=window,
                min_periods=2,
            )
        )

        combined[
            f"aqi_rolling_mean_{window}d"
        ] = rolling_aqi.mean().to_numpy()

        combined[
            f"aqi_rolling_std_{window}d"
        ] = rolling_aqi.std().to_numpy()

        rolling_pm25 = (
            historical_pm25
            .rolling(
                window=window,
                min_periods=2,
            )
        )

        combined[
            f"pm2_5_rolling_mean_{window}d"
        ] = rolling_pm25.mean().to_numpy()

    # ------------------------------------------------------------------------
    # Calendar features
    # ------------------------------------------------------------------------

    combined["month"] = (
        combined["date"].dt.month
    )

    combined["day_of_week"] = (
        combined["date"].dt.dayofweek
    )

    # ------------------------------------------------------------------------
    # Select final model feature contract
    # ------------------------------------------------------------------------

    feature_columns = [
        "aqi",
        "pm2_5_24h_mean",
        "aqi_lag_1d",
        "aqi_lag_2d",
        "aqi_lag_3d",
        "aqi_lag_7d",
        "pm2_5_lag_1d",
        "pm2_5_lag_3d",
        "pm2_5_lag_7d",
        "aqi_rolling_mean_3d",
        "aqi_rolling_mean_7d",
        "aqi_rolling_std_3d",
        "aqi_rolling_std_7d",
        "pm2_5_rolling_mean_3d",
        "pm2_5_rolling_mean_7d",
        "temperature_mean",
        "humidity_mean",
        "pressure_mean",
        "wind_speed_mean",
        "precipitation_total",
        "month",
        "day_of_week",
    ]

    latest = (
        combined[
            combined["date"]
            == live_date
        ]
        .copy()
    )

    if len(latest) != 1:
        raise RuntimeError(
            "Unable to isolate exactly one live feature row."
        )

    latest["city"] = CITY_NAME

    latest = latest[
        [
            "date",
            "city",
            *feature_columns,
        ]
    ].copy()

    missing_features = [
        feature
        for feature in feature_columns
        if pd.isna(
            latest.iloc[0][feature]
        )
    ]

    if missing_features:
        raise RuntimeError(
            "Live feature row contains missing predictors: "
            + ", ".join(missing_features)
        )

    return latest


# ============================================================================
# REPORT
# ============================================================================

def save_report(
    *,
    feature_row: pd.DataFrame,
    live_pm25: pd.DataFrame,
    live_daily: pd.DataFrame,
) -> None:
    """Save auditable live-refresh metadata."""

    LIVE_REPORT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = {
        "project": "Pearls AQI Predictor",
        "refresh_timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "city": CITY_NAME,
        "live_date": str(
            feature_row["date"].iloc[0]
        ),
        "live_pm25_hourly_observations": int(
            len(live_pm25)
        ),
        "live_pm25_hourly_coverage": float(
            live_daily[
                "hourly_observations"
            ].iloc[0]
            / 24
            * 100
        ),
        "complete_24h": bool(
            live_daily[
                "complete_24h"
            ].iloc[0]
        ),
        "live_pm25_24h_mean": float(
            live_daily[
                "pm2_5_24h_mean"
            ].iloc[0]
        ),
        "estimated_current_aqi": float(
            feature_row[
                "aqi"
            ].iloc[0]
        ),
        "status": "READY_FOR_INFERENCE",
    }

    LIVE_REPORT_FILE.write_text(
        json.dumps(
            report,
            indent=4,
        ),
        encoding="utf-8",
    )


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    """Execute production live-data refresh."""

    logger.info("=" * 72)
    logger.info(
        "PEARLS AQI PREDICTOR"
    )
    logger.info(
        "LIVE DAILY FEATURE REFRESH"
    )
    logger.info("=" * 72)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    api_key = get_openaq_api_key()

    sensor_ids = load_known_pm25_sensors()

    now_utc = datetime.now(
        timezone.utc
    )

    start_utc = (
        now_utc
        - timedelta(
            hours=30
        )
    )

    logger.info(
        "Live OpenAQ window: %s -> %s",
        start_utc.isoformat(),
        now_utc.isoformat(),
    )

    with requests.Session() as session:

        live_pm25 = collect_live_pm25(
            session=session,
            api_key=api_key,
            sensor_ids=sensor_ids,
            start_utc=start_utc,
            end_utc=now_utc,
        )

        hourly = build_hourly_city_series(
            live_pm25
        )

        daily = build_daily_pm25(
            hourly
        )

        if daily.empty:
            raise RuntimeError(
                "No daily live PM2.5 window produced."
            )

        complete_days = daily[
            daily["complete_24h"].astype(bool)
        ].copy()

        if complete_days.empty:
            logger.warning(
                "No complete 24-hour PM2.5 day is available "
                "in the current live window. Refresh postponed."
            )
            return

        live_date = complete_days[
            "date"
        ].max()

        live_daily = complete_days[
            complete_days["date"] == live_date
        ].copy()

        if len(live_daily) != 1:
            raise RuntimeError(
                "Unable to isolate exactly one complete live PM2.5 day."
            )

        logger.info(
            "Selected latest complete live day: %s",
            live_date,
        )

        logger.info(
            "Complete 24-hour PM2.5 observations: %s",
            live_daily["hourly_observations"].iloc[0],
        )

        weather = collect_current_weather(
            session
        )

    historical = load_historical_aqi()

    feature_row = build_live_feature_row(
        historical=historical,
        live_daily=live_daily,
        live_weather=weather,
    )

    feature_row.to_parquet(
        LIVE_FEATURE_FILE,
        index=False,
    )

    feature_row.to_csv(
        LIVE_FEATURE_CSV,
        index=False,
    )

    save_report(
        feature_row=feature_row,
        live_pm25=live_pm25,
        live_daily=live_daily,
    )

    logger.info(
        "Live feature row saved: %s",
        LIVE_FEATURE_FILE,
    )

    logger.info(
        "Live feature date: %s",
        feature_row[
            "date"
        ].iloc[0],
    )

    logger.info(
        "Estimated current AQI: %.2f",
        feature_row[
            "aqi"
        ].iloc[0],
    )

    logger.info("=" * 72)
    logger.info(
        "LIVE DAILY FEATURE REFRESH COMPLETED."
    )
    logger.info("=" * 72)


if __name__ == "__main__":
    main()
