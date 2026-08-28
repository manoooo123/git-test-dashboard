from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv
from requests import Response
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# ============================================================================
# PROJECT CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ENV_FILE = PROJECT_ROOT / ".env"

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

AIR_QUALITY_FILE = RAW_DATA_DIR / "openaq_lahore.csv"

WEATHER_FILE = RAW_DATA_DIR / "weather_lahore.csv"

FINAL_DATASET_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "dataset_lahore.csv"
)


# ============================================================================
# ENVIRONMENT
# ============================================================================

load_dotenv(ENV_FILE)

OPENAQ_API_KEY = os.getenv("OPENAQ_API_KEY")

OPENAQ_URL = "https://api.openaq.org/v3"

OPEN_METEO_URL = (
    "https://archive-api.open-meteo.com/v1/archive"
)


# ============================================================================
# CONFIGURATION
# ============================================================================

REQUEST_TIMEOUT = 60

MAX_RETRIES = 4

BACKOFF_FACTOR = 1.5

OPENAQ_PAGE_SIZE = 1000

OPENAQ_LOCATION_RADIUS_METERS = 25_000

API_REQUEST_DELAY_SECONDS = 0.20


# ============================================================================
# LOCATION
# ============================================================================

@dataclass(frozen=True)
class Location:
    """Target geographical location."""

    city: str
    latitude: float
    longitude: float


LAHORE = Location(
    city="Lahore",
    latitude=31.5204,
    longitude=74.3587,
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
# EXCEPTIONS
# ============================================================================

class DataIngestionError(RuntimeError):
    """Base ingestion error."""


class AuthenticationError(DataIngestionError):
    """Authentication failure."""


class APIRequestError(DataIngestionError):
    """API request failure."""


class DataValidationError(DataIngestionError):
    """Data validation failure."""


# ============================================================================
# CONFIGURATION VALIDATION
# ============================================================================

def validate_configuration() -> None:
    """Validate required environment variables."""

    if not OPENAQ_API_KEY:
        raise AuthenticationError(
            "OPENAQ_API_KEY is missing from .env"
        )

    if not OPENAQ_API_KEY.strip():
        raise AuthenticationError(
            "OPENAQ_API_KEY is empty."
        )

    logger.info(
        "Runtime configuration validated."
    )


# ============================================================================
# HTTP SESSION
# ============================================================================

def create_session() -> requests.Session:
    """Create reusable HTTP session."""

    session = requests.Session()

    retry_policy = Retry(
        total=MAX_RETRIES,
        connect=MAX_RETRIES,
        read=MAX_RETRIES,
        status=MAX_RETRIES,
        backoff_factor=BACKOFF_FACTOR,
        status_forcelist=(
            429,
            500,
            502,
            503,
            504,
        ),
        allowed_methods=frozenset({"GET"}),
        raise_on_status=False,
    )

    adapter = HTTPAdapter(
        max_retries=retry_policy,
        pool_connections=10,
        pool_maxsize=10,
    )

    session.mount(
        "https://",
        adapter,
    )

    session.mount(
        "http://",
        adapter,
    )

    session.headers.update(
        {
            "Accept": "application/json",
            "User-Agent": (
                "Pearls-AQI-Predictor/1.0 "
                "(research-project)"
            ),
            "X-API-Key": OPENAQ_API_KEY,
        }
    )

    return session


# ============================================================================
# API REQUEST
# ============================================================================

def get_json(
    session: requests.Session,
    url: str,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute GET request and return JSON."""

    try:

        response: Response = session.get(
            url,
            params=params,
            timeout=REQUEST_TIMEOUT,
        )

    except requests.RequestException as exc:

        raise APIRequestError(
            f"Network request failed: {exc}"
        ) from exc

    if response.status_code in (401, 403):

        raise AuthenticationError(
            "OpenAQ authentication failed. "
            "Check OPENAQ_API_KEY."
        )

    if response.status_code == 422:

        raise APIRequestError(
            "OpenAQ returned HTTP 422. "
            "Check query parameters."
        )

    if response.status_code >= 400:

        raise APIRequestError(
            f"API returned HTTP "
            f"{response.status_code}."
        )

    try:

        payload = response.json()

    except ValueError as exc:

        raise APIRequestError(
            "API returned invalid JSON."
        ) from exc

    if not isinstance(payload, dict):

        raise APIRequestError(
            "Unexpected API response."
        )

    return payload


# ============================================================================
# LOCATION DISCOVERY
# ============================================================================

def find_lahore_locations(
    session: requests.Session,
) -> pd.DataFrame:
    """Find real OpenAQ monitoring locations."""

    logger.info(
        "Discovering OpenAQ monitoring locations."
    )

    url = f"{OPENAQ_URL}/locations"

    params = {
        "coordinates": (
            f"{LAHORE.latitude},"
            f"{LAHORE.longitude}"
        ),
        "radius": OPENAQ_LOCATION_RADIUS_METERS,
        "limit": 100,
    }

    payload = get_json(
        session,
        url,
        params,
    )

    results = payload.get(
        "results",
        [],
    )

    if not results:

        raise DataIngestionError(
            "No OpenAQ locations found."
        )

    records = []

    for location in results:

        coordinates = location.get(
            "coordinates",
            {},
        )

        records.append(
            {
                "location_id": location.get("id"),
                "location_name": location.get("name"),
                "latitude": coordinates.get("latitude"),
                "longitude": coordinates.get("longitude"),
                "country": location.get("country"),
                "provider": location.get("provider"),
            }
        )

    locations = pd.DataFrame(records)

    locations.dropna(
        subset=["location_id"],
        inplace=True,
    )

    locations.drop_duplicates(
        subset=["location_id"],
        inplace=True,
    )

    locations.reset_index(
        drop=True,
        inplace=True,
    )

    logger.info(
        "Discovered %d OpenAQ locations.",
        len(locations),
    )

    return locations


# ============================================================================
# SENSOR DISCOVERY
# ============================================================================

def get_location_sensors(
    session: requests.Session,
    location_id: int,
) -> list[dict[str, Any]]:
    """Get sensors for location."""

    url = (
        f"{OPENAQ_URL}/locations/"
        f"{location_id}/sensors"
    )

    payload = get_json(
        session,
        url,
    )

    sensors = payload.get(
        "results",
        [],
    )

    if not isinstance(
        sensors,
        list,
    ):

        raise APIRequestError(
            f"Invalid sensor response "
            f"for location {location_id}."
        )

    return sensors


# ============================================================================
# PARAMETER NORMALIZATION
# ============================================================================

def normalize_parameter(
    parameter: str,
) -> str | None:
    """Normalize pollutant parameter."""

    normalized = (
        str(parameter)
        .strip()
        .lower()
        .replace(".", "")
        .replace("_", "")
    )

    if normalized in {
        "pm25",
        "particulatematter25",
    }:

        return "pm25"

    if normalized in {
        "pm10",
        "particulatematter10",
    }:

        return "pm10"

    return None


# ============================================================================
# SENSOR DATA
# ============================================================================

def collect_sensor_hours(
    session: requests.Session,
    sensor_id: int,
    parameter: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Collect paginated hourly sensor observations."""

    url = (
        f"{OPENAQ_URL}/sensors/"
        f"{sensor_id}/hours"
    )

    records = []

    page = 1

    while True:

        params = {
            "datetime_from": (
                f"{start_date}T00:00:00Z"
            ),
            "datetime_to": (
                f"{end_date}T23:59:59Z"
            ),
            "limit": OPENAQ_PAGE_SIZE,
            "page": page,
        }

        payload = get_json(
            session,
            url,
            params,
        )

        results = payload.get(
            "results",
            [],
        )

        if not results:
            break

        for item in results:

            period = item.get(
                "period",
                {},
            )

            timestamp = (
                period
                .get("datetimeFrom", {})
                .get("utc")
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
                    "parameter": parameter,
                    "value": value,
                }
            )

        logger.info(
            "Sensor=%s | Parameter=%s | "
            "Page=%s | Records=%s",
            sensor_id,
            parameter,
            page,
            len(results),
        )

        if len(results) < OPENAQ_PAGE_SIZE:
            break

        page += 1

        time.sleep(
            API_REQUEST_DELAY_SECONDS
        )

    if not records:

        return pd.DataFrame()

    data = pd.DataFrame(records)

    data["timestamp"] = pd.to_datetime(
        data["timestamp"],
        utc=True,
        errors="coerce",
    )

    data["value"] = pd.to_numeric(
        data["value"],
        errors="coerce",
    )

    data.dropna(
        subset=[
            "timestamp",
            "value",
        ],
        inplace=True,
    )

    data.drop_duplicates(
        subset=[
            "timestamp",
            "sensor_id",
            "parameter",
        ],
        inplace=True,
    )

    return data.reset_index(
        drop=True
    )


# ============================================================================
# LOCATION MEASUREMENTS
# ============================================================================

def collect_location_measurements(
    session: requests.Session,
    location_id: int,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Collect PM2.5 and PM10 for a location."""

    sensors = get_location_sensors(
        session,
        location_id,
    )

    if not sensors:

        logger.warning(
            "Location %s has no sensors.",
            location_id,
        )

        return pd.DataFrame()

    frames = []

    for sensor in sensors:

        sensor_id = sensor.get("id")

        parameter_info = sensor.get(
            "parameter",
            {},
        )

        parameter_name = parameter_info.get(
            "name"
        )

        if sensor_id is None:
            continue

        if parameter_name is None:
            continue

        parameter = normalize_parameter(
            parameter_name
        )

        if parameter is None:
            continue

        logger.info(
            "Collecting %s | location=%s | sensor=%s",
            parameter,
            location_id,
            sensor_id,
        )

        try:

            data = collect_sensor_hours(
                session=session,
                sensor_id=int(sensor_id),
                parameter=parameter,
                start_date=start_date,
                end_date=end_date,
            )

        except DataIngestionError as exc:

            logger.warning(
                "Sensor %s failed: %s",
                sensor_id,
                exc,
            )

            continue

        if not data.empty:

            # VERY IMPORTANT:
            # Preserve OpenAQ location identity.
            data["location_id"] = int(
                location_id
            )

            frames.append(data)

    if not frames:

        return pd.DataFrame()

    return pd.concat(
        frames,
        ignore_index=True,
    )


# ============================================================================
# AIR QUALITY NORMALIZATION
# ============================================================================

def normalize_air_quality(
    measurements: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize OpenAQ data while preserving location_id."""

    if measurements.empty:

        raise DataValidationError(
            "Air-quality measurements are empty."
        )

    data = measurements.copy()

    required_columns = {
        "timestamp",
        "sensor_id",
        "location_id",
        "parameter",
        "value",
    }

    missing = (
        required_columns
        - set(data.columns)
    )

    if missing:

        raise DataValidationError(
            "Missing OpenAQ columns: "
            f"{sorted(missing)}"
        )

    data["timestamp"] = pd.to_datetime(
        data["timestamp"],
        utc=True,
        errors="coerce",
    )

    data["sensor_id"] = pd.to_numeric(
        data["sensor_id"],
        errors="coerce",
    )

    data["location_id"] = pd.to_numeric(
        data["location_id"],
        errors="coerce",
    )

    data["value"] = pd.to_numeric(
        data["value"],
        errors="coerce",
    )

    data["parameter"] = (
        data["parameter"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    data.dropna(
        subset=[
            "timestamp",
            "sensor_id",
            "location_id",
            "parameter",
            "value",
        ],
        inplace=True,
    )

    wide = (
        data
        .pivot_table(
            index=[
                "location_id",
                "sensor_id",
                "timestamp",
            ],
            columns="parameter",
            values="value",
            aggfunc="mean",
        )
        .reset_index()
    )

    wide.columns.name = None

    wide.rename(
        columns={
            "pm25": "pm2_5",
            "pm10": "pm10",
        },
        inplace=True,
    )

    if "pm2_5" not in wide.columns:

        raise DataValidationError(
            "PM2.5 was not found."
        )

    logger.info(
        "Normalized OpenAQ observations: %d",
        len(wide),
    )

    logger.info(
        "Unique monitoring sensors: %d",
        wide["sensor_id"].nunique(),
    )

    logger.info(
        "Unique monitoring locations: %d",
        wide["location_id"].nunique(),
    )

    return (
        wide
        .sort_values(
            ["timestamp", "location_id", "sensor_id"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


# ============================================================================
# WEATHER
# ============================================================================

def collect_weather(
    session: requests.Session,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Collect historical Open-Meteo weather."""

    from calendar import monthrange
    from datetime import date

    logger.info(
        "Collecting historical meteorological data."
    )

    start = date.fromisoformat(start_date)

    end = date.fromisoformat(end_date)

    frames = []

    current_year = start.year

    current_month = start.month

    while (
        current_year < end.year
        or (
            current_year == end.year
            and current_month <= end.month
        )
    ):

        month_start = date(
            current_year,
            current_month,
            1,
        )

        last_day = monthrange(
            current_year,
            current_month,
        )[1]

        month_end = date(
            current_year,
            current_month,
            last_day,
        )

        chunk_start = max(
            month_start,
            start,
        )

        chunk_end = min(
            month_end,
            end,
        )

        logger.info(
            "Weather chunk: %s → %s",
            chunk_start,
            chunk_end,
        )

        params = {
            "latitude": LAHORE.latitude,
            "longitude": LAHORE.longitude,
            "start_date": chunk_start.isoformat(),
            "end_date": chunk_end.isoformat(),
            "hourly": (
                "temperature_2m,"
                "relative_humidity_2m,"
                "surface_pressure,"
                "cloud_cover,"
                "wind_speed_10m,"
                "wind_direction_10m,"
                "precipitation"
            ),
            "timezone": "Asia/Karachi",
        }

        response = session.get(
            OPEN_METEO_URL,
            params=params,
            timeout=(15, 120),
        )

        if response.status_code >= 400:

            raise APIRequestError(
                "Open-Meteo returned HTTP "
                f"{response.status_code}."
            )

        payload = response.json()

        hourly = payload.get(
            "hourly"
        )

        if not isinstance(
            hourly,
            dict,
        ):

            raise DataValidationError(
                "Invalid Open-Meteo response."
            )

        chunk = pd.DataFrame(hourly)

        if chunk.empty:

            raise DataValidationError(
                f"No weather data for "
                f"{chunk_start} → {chunk_end}."
            )

        chunk.rename(
            columns={
                "time": "timestamp",
                "temperature_2m": "temperature",
                "relative_humidity_2m": "humidity",
                "surface_pressure": "pressure",
                "cloud_cover": "clouds",
                "wind_speed_10m": "wind_speed",
                "wind_direction_10m": "wind_direction",
            },
            inplace=True,
        )

        chunk["timestamp"] = pd.to_datetime(
            chunk["timestamp"],
            errors="coerce",
        )

        chunk.dropna(
            subset=["timestamp"],
            inplace=True,
        )

        chunk["timestamp"] = (
            chunk["timestamp"]
            .dt.tz_localize(
                "Asia/Karachi"
            )
            .dt.tz_convert("UTC")
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

            if column in chunk.columns:

                chunk[column] = pd.to_numeric(
                    chunk[column],
                    errors="coerce",
                )

        chunk.drop_duplicates(
            subset=["timestamp"],
            inplace=True,
        )

        frames.append(chunk)

        logger.info(
            "Weather chunk collected successfully: "
            "%d observations.",
            len(chunk),
        )

        if current_month == 12:

            current_year += 1
            current_month = 1

        else:

            current_month += 1

    weather = pd.concat(
        frames,
        ignore_index=True,
    )

    weather.sort_values(
        "timestamp",
        inplace=True,
    )

    weather.drop_duplicates(
        subset=["timestamp"],
        inplace=True,
    )

    weather.reset_index(
        drop=True,
        inplace=True,
    )

    logger.info(
        "Total weather observations: %d",
        len(weather),
    )

    logger.info(
        "Weather coverage: %s → %s",
        weather["timestamp"].min(),
        weather["timestamp"].max(),
    )

    return weather


# ============================================================================
# DATASET MERGE
# ============================================================================

def build_dataset(
    air_quality: pd.DataFrame,
    weather: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge air quality and weather.

    IMPORTANT:
    merge_asof requires the merge key itself to be globally sorted.
    """

    if air_quality.empty:

        raise DataValidationError(
            "Air-quality dataset is empty."
        )

    if weather.empty:

        raise DataValidationError(
            "Weather dataset is empty."
        )

    air = air_quality.copy()

    met = weather.copy()

    # ------------------------------------------------------------------
    # Timestamp normalization
    # ------------------------------------------------------------------

    air["timestamp"] = pd.to_datetime(
        air["timestamp"],
        utc=True,
        errors="coerce",
    )

    met["timestamp"] = pd.to_datetime(
        met["timestamp"],
        utc=True,
        errors="coerce",
    )

    air.dropna(
        subset=["timestamp"],
        inplace=True,
    )

    met.dropna(
        subset=["timestamp"],
        inplace=True,
    )

    # ------------------------------------------------------------------
    # Location identity
    # ------------------------------------------------------------------

    if "location_id" not in air.columns:

        raise DataValidationError(
            "location_id is missing from "
            "air-quality data."
        )

    # ------------------------------------------------------------------
    # PM2.5
    # ------------------------------------------------------------------

    if "pm2_5" not in air.columns:

        raise DataValidationError(
            "PM2.5 column is missing."
        )

    # ------------------------------------------------------------------
    # Remove duplicate observations
    # ------------------------------------------------------------------

    air.drop_duplicates(
        subset=[
            "location_id",
            "sensor_id",
            "timestamp",
        ],
        keep="last",
        inplace=True,
    )

    # ------------------------------------------------------------------
    # CRITICAL merge_asof FIX
    #
    # The timestamp must be globally sorted.
    # DO NOT sort location_id first.
    # ------------------------------------------------------------------

    air.sort_values(
        by=[
            "timestamp",
            "location_id",
            "sensor_id",
        ],
        kind="mergesort",
        inplace=True,
    )

    met.sort_values(
        by=["timestamp"],
        kind="mergesort",
        inplace=True,
    )

    air.reset_index(
        drop=True,
        inplace=True,
    )

    met.reset_index(
        drop=True,
        inplace=True,
    )

    # ------------------------------------------------------------------
    # Verify sorting
    # ------------------------------------------------------------------

    if not air["timestamp"].is_monotonic_increasing:

        raise DataValidationError(
            "Air-quality timestamps are still "
            "not sorted."
        )

    if not met["timestamp"].is_monotonic_increasing:

        raise DataValidationError(
            "Weather timestamps are not sorted."
        )

    logger.info(
        "Air-quality timestamps verified sorted."
    )

    logger.info(
        "Weather timestamps verified sorted."
    )

    # ------------------------------------------------------------------
    # MERGE
    # ------------------------------------------------------------------

    dataset = pd.merge_asof(
        air,
        met,
        on="timestamp",
        direction="nearest",
        tolerance=pd.Timedelta("1 hour"),
    )

    # ------------------------------------------------------------------
    # Remove rows without PM2.5
    # ------------------------------------------------------------------

    dataset.dropna(
        subset=["pm2_5"],
        inplace=True,
    )

    if dataset.empty:

        raise DataValidationError(
            "No valid observations after merge."
        )

    # ------------------------------------------------------------------
    # City
    # ------------------------------------------------------------------

    dataset["city"] = LAHORE.city

    # ------------------------------------------------------------------
    # Final ordering
    # ------------------------------------------------------------------

    dataset.sort_values(
        by=[
            "location_id",
            "sensor_id",
            "timestamp",
        ],
        kind="mergesort",
        inplace=True,
    )

    dataset.reset_index(
        drop=True,
        inplace=True,
    )

    logger.info(
        "Merged dataset rows: %d",
        len(dataset),
    )

    logger.info(
        "Monitoring locations preserved: %d",
        dataset["location_id"].nunique(),
    )

    logger.info(
        "Monitoring sensors preserved: %d",
        dataset["sensor_id"].nunique(),
    )

    return dataset


# ============================================================================
# VALIDATION
# ============================================================================

def validate_dataset(
    dataset: pd.DataFrame,
) -> None:
    """Validate final dataset."""

    if dataset.empty:

        raise DataValidationError(
            "Final dataset is empty."
        )

    required_columns = {
        "timestamp",
        "city",
        "location_id",
        "sensor_id",
        "pm2_5",
    }

    missing = (
        required_columns
        - set(dataset.columns)
    )

    if missing:

        raise DataValidationError(
            "Missing final columns: "
            f"{sorted(missing)}"
        )

    if dataset["timestamp"].isna().any():

        raise DataValidationError(
            "Null timestamps remain."
        )

    if dataset["pm2_5"].isna().all():

        raise DataValidationError(
            "All PM2.5 values are missing."
        )

    logger.info(
        "Final dataset validation passed."
    )


# ============================================================================
# QUALITY REPORT
# ============================================================================

def generate_quality_report(
    dataset: pd.DataFrame,
) -> None:
    """Print final data quality report."""

    logger.info("=" * 72)

    logger.info(
        "FINAL DATA QUALITY REPORT"
    )

    logger.info("=" * 72)

    logger.info(
        "Rows: %d",
        len(dataset),
    )

    logger.info(
        "Columns: %d",
        len(dataset.columns),
    )

    logger.info(
        "Locations: %d",
        dataset["location_id"].nunique(),
    )

    logger.info(
        "Sensors: %d",
        dataset["sensor_id"].nunique(),
    )

    logger.info(
        "Start: %s",
        dataset["timestamp"].min(),
    )

    logger.info(
        "End: %s",
        dataset["timestamp"].max(),
    )

    logger.info(
        "Missing values:\n%s",
        dataset.isna()
        .sum()
        .to_string(),
    )

    if "temperature" in dataset.columns:

        logger.info(
            "Weather matched: %.2f%%",
            dataset["temperature"]
            .notna()
            .mean()
            * 100,
        )

    logger.info("=" * 72)


# ============================================================================
# SAVE
# ============================================================================

def save_csv(
    dataframe: pd.DataFrame,
    destination: Path,
) -> None:
    """Atomically save dataframe."""

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_file = destination.with_suffix(
        ".tmp.csv"
    )

    dataframe.to_csv(
        temporary_file,
        index=False,
    )

    temporary_file.replace(
        destination
    )

    logger.info(
        "Saved: %s",
        destination,
    )


# ============================================================================
# LOAD EXISTING DATA
# ============================================================================

def load_existing_raw_data() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Load previously collected raw CSV files.

    This prevents unnecessary API re-downloads.
    """

    logger.info(
        "Checking for existing raw data..."
    )

    if not AIR_QUALITY_FILE.exists():

        raise FileNotFoundError(
            "Existing OpenAQ CSV not found."
        )

    if not WEATHER_FILE.exists():

        raise FileNotFoundError(
            "Existing weather CSV not found."
        )

    air = pd.read_csv(
        AIR_QUALITY_FILE
    )

    weather = pd.read_csv(
        WEATHER_FILE
    )

    logger.info(
        "Existing OpenAQ CSV loaded: %d rows.",
        len(air),
    )

    logger.info(
        "Existing weather CSV loaded: %d rows.",
        len(weather),
    )

    return air, weather


# ============================================================================
# COMPLETE INGESTION
# ============================================================================

def run_ingestion(
    start_date: str,
    end_date: str,
    reuse_existing: bool = True,
) -> pd.DataFrame:
    """Run complete real-data ingestion pipeline."""

    validate_configuration()

    RAW_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------------
    # REUSE EXISTING DATA
    # ------------------------------------------------------------------

    if (
        reuse_existing
        and AIR_QUALITY_FILE.exists()
        and WEATHER_FILE.exists()
    ):

        logger.info(
            "Existing raw datasets found."
        )

        try:

            raw_air, weather = (
                load_existing_raw_data()
            )

            # Existing raw OpenAQ data already contains
            # location_id because the corrected collector
            # writes it.

            air_quality = normalize_air_quality(
                raw_air
            )

            dataset = build_dataset(
                air_quality=air_quality,
                weather=weather,
            )

            validate_dataset(
                dataset
            )

            generate_quality_report(
                dataset
            )

            save_csv(
                dataset,
                FINAL_DATASET_FILE,
            )

            logger.info(
                "PIPELINE COMPLETED USING "
                "EXISTING RAW DATA."
            )

            return dataset

        except (
            DataValidationError,
            KeyError,
            ValueError,
        ) as exc:

            logger.warning(
                "Existing raw data cannot be reused: %s",
                exc,
            )

            logger.info(
                "Starting fresh API collection."
            )

    # ------------------------------------------------------------------
    # API COLLECTION
    # ------------------------------------------------------------------

    session = create_session()

    logger.info(
        "Starting REAL data ingestion."
    )

    logger.info(
        "Historical period: %s → %s",
        start_date,
        end_date,
    )

    # ------------------------------------------------------------------
    # LOCATIONS
    # ------------------------------------------------------------------

    locations = find_lahore_locations(
        session
    )

    logger.info(
        "Locations available: %d",
        len(locations),
    )

    # ------------------------------------------------------------------
    # AIR QUALITY
    # ------------------------------------------------------------------

    measurement_frames = []

    for location_id in locations[
        "location_id"
    ].dropna():

        try:

            location_data = (
                collect_location_measurements(
                    session=session,
                    location_id=int(location_id),
                    start_date=start_date,
                    end_date=end_date,
                )
            )

            if not location_data.empty:

                measurement_frames.append(
                    location_data
                )

        except DataIngestionError as exc:

            logger.warning(
                "Location %s skipped: %s",
                location_id,
                exc,
            )

    if not measurement_frames:

        raise DataIngestionError(
            "No real OpenAQ measurements collected."
        )

    raw_air_quality = pd.concat(
        measurement_frames,
        ignore_index=True,
    )

    raw_air_quality.drop_duplicates(
        inplace=True
    )

    save_csv(
        raw_air_quality,
        AIR_QUALITY_FILE,
    )

    # ------------------------------------------------------------------
    # NORMALIZE
    # ------------------------------------------------------------------

    air_quality = normalize_air_quality(
        raw_air_quality
    )

    # ------------------------------------------------------------------
    # WEATHER
    # ------------------------------------------------------------------

    weather = collect_weather(
        session=session,
        start_date=start_date,
        end_date=end_date,
    )

    save_csv(
        weather,
        WEATHER_FILE,
    )

    # ------------------------------------------------------------------
    # MERGE
    # ------------------------------------------------------------------

    dataset = build_dataset(
        air_quality=air_quality,
        weather=weather,
    )

    # ------------------------------------------------------------------
    # VALIDATION
    # ------------------------------------------------------------------

    validate_dataset(
        dataset
    )

    generate_quality_report(
        dataset
    )

    save_csv(
        dataset,
        FINAL_DATASET_FILE,
    )

    logger.info(
        "REAL DATA INGESTION COMPLETED."
    )

    return dataset


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    """Application entry point."""

    dataset = run_ingestion(
        start_date="2023-01-01",
        end_date="2025-12-31",
        reuse_existing=True,
    )

    print("\n" + "=" * 72)

    print(
        "PEARLS AQI PREDICTOR "
        "DATA INGESTION SUCCESS"
    )

    print("=" * 72)

    print(
        "\nDataset shape:"
    )

    print(
        dataset.shape
    )

    print(
        "\nDataset columns:"
    )

    print(
        dataset.columns.tolist()
    )

    print(
        "\nFirst 10 rows:"
    )

    print(
        dataset.head(10)
        .to_string(index=False)
    )

    print(
        "\nFinal dataset saved at:"
    )

    print(
        FINAL_DATASET_FILE
    )


if __name__ == "__main__":
    main()