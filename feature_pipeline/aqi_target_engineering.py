"""
Pearls AQI Predictor
====================

Production-grade AQI target engineering.

Purpose
-------
Convert real hourly Lahore PM2.5 observations into a daily AQI target
using a Punjab-oriented PM2.5 AQI breakpoint framework.

Source basis
------------
Punjab EPA Lahore AQI reports state that AQI is calculated using PM2.5
data from the previous 24 hours.

Punjab's published smog-control policy defines PM2.5 concentration bands
against AQI bands using the Punjab Environmental Quality Standard (PEQS)
value of 35 ug/m3.

Important
---------
- No synthetic observations are generated.
- No missing hourly observations are fabricated.
- A daily AQI target is created only when all 24 hourly PM2.5 observations
  required for that calendar day are available.
- The methodology is explicitly versioned for reproducibility.
- This is a Punjab-aligned calculated AQI target, not an official
  EPA Punjab AQI record.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
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

DAILY_AQI_FILE = (
    OUTPUT_DIR
    / "daily_lahore_aqi.parquet"
)

DAILY_AQI_CSV = (
    OUTPUT_DIR
    / "daily_lahore_aqi.csv"
)

FORECAST_TARGET_FILE = (
    OUTPUT_DIR
    / "aqi_forecast_targets.parquet"
)

FORECAST_TARGET_CSV = (
    OUTPUT_DIR
    / "aqi_forecast_targets.csv"
)

REPORT_FILE = (
    REPORT_DIR
    / "aqi_target_report.json"
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
# METHODOLOGY CONFIGURATION
# ============================================================================

@dataclass(frozen=True)
class AQIMethodology:
    """
    Versioned Punjab-aligned PM2.5 AQI methodology.

    Concentration units:
        micrograms per cubic metre (ug/m3)

    The breakpoint bands correspond to the Punjab PM2.5 framework:
        0-35       -> AQI 0-100
        36-70      -> AQI 101-200
        71-105     -> AQI 200-300
        106-140    -> AQI 301-400
        141-300    -> AQI 401-500
        >300       -> AQI >500
    """

    version: str = "Punjab_PM25_PEQ_35_v1"

    peqs_pm25: float = 35.0

    # PM2.5 concentration breakpoints.
    concentration_breakpoints: tuple[float, ...] = (
        0.0,
        35.0,
        70.0,
        105.0,
        140.0,
        300.0,
    )

    # Corresponding AQI breakpoints.
    aqi_breakpoints: tuple[float, ...] = (
        0.0,
        100.0,
        200.0,
        300.0,
        400.0,
        500.0,
    )


METHODOLOGY = AQIMethodology()


# ============================================================================
# CATEGORY DEFINITIONS
# ============================================================================

def classify_aqi(
    aqi: float,
) -> str:
    """Map calculated AQI to the Punjab reporting category."""

    if aqi <= 50:
        return "Good"

    if aqi <= 100:
        return "Satisfactory"

    if aqi <= 150:
        return "Moderate"

    if aqi <= 200:
        return "Unhealthy for Sensitive Groups"

    if aqi <= 300:
        return "Unhealthy"

    if aqi <= 400:
        return "Very Unhealthy"

    if aqi <= 500:
        return "Hazardous"

    return "Severe"


def classify_aqi_color(
    aqi: float,
) -> str:
    """Return dashboard-friendly AQI severity class."""

    if aqi <= 100:
        return "Green"

    if aqi <= 200:
        return "Yellow"

    if aqi <= 300:
        return "Orange"

    if aqi <= 500:
        return "Red"

    return "Maroon"


# ============================================================================
# LOAD CANONICAL DATA
# ============================================================================

def load_canonical_data() -> pd.DataFrame:
    """
    Load real canonical Lahore hourly data.
    """

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Canonical dataset not found: {INPUT_FILE}"
        )

    logger.info(
        "Loading canonical hourly dataset: %s",
        INPUT_FILE,
    )

    df = pd.read_parquet(
        INPUT_FILE
    )

    if df.empty:
        raise ValueError(
            "Canonical dataset is empty."
        )

    required_columns = {
        "timestamp",
        "pm2_5",
        "city",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            "Canonical dataset missing columns: "
            + ", ".join(sorted(missing))
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
        errors="coerce",
    )

    df["pm2_5"] = pd.to_numeric(
        df["pm2_5"],
        errors="coerce",
    )

    df.dropna(
        subset=[
            "timestamp",
            "pm2_5",
        ],
        inplace=True,
    )

    if (df["pm2_5"] < 0).any():
        raise ValueError(
            "Negative PM2.5 observations detected."
        )

    df = (
        df
        .sort_values("timestamp")
        .drop_duplicates("timestamp")
        .reset_index(drop=True)
    )

    logger.info(
        "Canonical observations loaded: %d",
        len(df),
    )

    logger.info(
        "Period: %s → %s",
        df["timestamp"].min(),
        df["timestamp"].max(),
    )

    return df


# ============================================================================
# DAILY PM2.5
# ============================================================================

def create_daily_pm25(
    hourly: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create daily PM2.5 observations.

    A daily observation is accepted only when all 24 UTC hourly records
    for the calendar day exist.

    No interpolation or imputation is performed.
    """

    data = hourly.copy()

    # Lahore is UTC+05:00. Convert UTC timestamps to local Lahore time
    # before determining calendar-day boundaries.
    local_timestamp = (
        data["timestamp"]
        .dt.tz_convert("Asia/Karachi")
    )

    data["local_date"] = (
        local_timestamp
        .dt.date
    )

    data["local_hour"] = (
        local_timestamp
        .dt.hour
    )

    daily = (
        data
        .groupby(
            "local_date",
            as_index=False,
        )
        .agg(
            pm2_5_24h_mean=("pm2_5", "mean"),
            hourly_observations=("pm2_5", "count"),
        )
    )

    # Strict 24-hour completeness.
    daily["complete_24h"] = (
        daily["hourly_observations"] == 24
    )

    incomplete_count = int(
        (~daily["complete_24h"]).sum()
    )

    logger.info(
        "Calendar days observed: %d",
        len(daily),
    )

    logger.info(
        "Incomplete 24-hour days: %d",
        incomplete_count,
    )

    daily = daily[
        daily["complete_24h"]
    ].copy()

    if daily.empty:
        raise ValueError(
            "No complete 24-hour PM2.5 periods are available."
        )

    daily["date"] = pd.to_datetime(
        daily["local_date"]
    )

    daily.drop(
        columns=["local_date"],
        inplace=True,
    )

    daily.sort_values(
        "date",
        inplace=True,
    )

    daily.reset_index(
        drop=True,
        inplace=True,
    )

    logger.info(
        "Complete daily PM2.5 observations: %d",
        len(daily),
    )

    return daily


# ============================================================================
# AQI CALCULATION
# ============================================================================

def calculate_pm25_aqi(
    concentration: float,
) -> float:
    """
    Calculate AQI using linear interpolation between Punjab PM2.5
    concentration/AQI breakpoints.

    Formula:

        AQI =
            AQI_low
            + (
                ((C - C_low) / (C_high - C_low))
                * (AQI_high - AQI_low)
              )

    For concentrations above 300 ug/m3, the final Punjab AQI band
    continues beyond 500. The calculated value is therefore allowed
    to exceed 500 rather than artificially clipping the concentration.
    """

    if not np.isfinite(
        concentration
    ):
        return np.nan

    if concentration < 0:
        raise ValueError(
            "PM2.5 concentration cannot be negative."
        )

    concentrations = np.array(
        METHODOLOGY.concentration_breakpoints,
        dtype=float,
    )

    aqis = np.array(
        METHODOLOGY.aqi_breakpoints,
        dtype=float,
    )

    # ------------------------------------------------------------------------
    # Standard breakpoint interpolation
    # ------------------------------------------------------------------------

    if concentration <= concentrations[-1]:

        index = np.searchsorted(
            concentrations,
            concentration,
            side="right",
        ) - 1

        index = max(
            0,
            min(
                index,
                len(concentrations) - 2,
            ),
        )

        c_low = concentrations[index]
        c_high = concentrations[index + 1]

        i_low = aqis[index]
        i_high = aqis[index + 1]

        if c_high == c_low:
            return float(i_low)

        aqi = (
            i_low
            + (
                (
                    concentration
                    - c_low
                )
                / (
                    c_high
                    - c_low
                )
            )
            * (
                i_high
                - i_low
            )
        )

        return float(aqi)

    # ------------------------------------------------------------------------
    # Above 300 ug/m3
    # ------------------------------------------------------------------------
    #
    # The Punjab framework identifies concentrations above 300 ug/m3
    # as the >500 AQI category. We extrapolate from the final defined
    # concentration/AQI segment instead of clipping all severe pollution
    # to exactly 500.
    # ------------------------------------------------------------------------

    c_low = concentrations[-1]
    c_high = (
        c_low
        + (
            concentrations[-1]
            - concentrations[-2]
        )
    )

    i_low = aqis[-1]

    # Continue the final concentration-to-AQI slope.
    final_slope = (
        aqis[-1]
        - aqis[-2]
    ) / (
        concentrations[-1]
        - concentrations[-2]
    )

    aqi = (
        i_low
        + (
            concentration
            - c_low
        )
        * final_slope
    )

    return float(aqi)


# ============================================================================
# DAILY AQI DATASET
# ============================================================================

def build_daily_aqi(
    daily_pm25: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert complete daily PM2.5 observations into calculated AQI targets.
    """

    daily = daily_pm25.copy()

    daily["aqi"] = (
        daily["pm2_5_24h_mean"]
        .apply(
            calculate_pm25_aqi
        )
    )

    daily["aqi_rounded"] = (
        daily["aqi"]
        .round()
        .astype("Int64")
    )

    daily["aqi_category"] = (
        daily["aqi"]
        .apply(
            classify_aqi
        )
    )

    daily["aqi_color"] = (
        daily["aqi"]
        .apply(
            classify_aqi_color
        )
    )

    daily["methodology"] = (
        METHODOLOGY.version
    )

    daily["source"] = (
        "OpenAQ PM2.5 + Punjab-aligned AQI calculation"
    )

    daily["city"] = "Lahore"

    daily = daily[
        [
            "date",
            "city",
            "pm2_5_24h_mean",
            "hourly_observations",
            "complete_24h",
            "aqi",
            "aqi_rounded",
            "aqi_category",
            "aqi_color",
            "methodology",
            "source",
        ]
    ]

    return daily


# ============================================================================
# FUTURE AQI TARGETS
# ============================================================================

def create_forecast_targets(
    daily_aqi: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create next-day, next-2-day and next-3-day AQI targets.

    Targets are created only when the corresponding future day exists
    as a valid calculated AQI observation.

    No future values are inserted into predictor features here.
    """

    base = daily_aqi.copy()

    target_source = (
        daily_aqi[
            [
                "date",
                "aqi",
            ]
        ]
        .copy()
    )

    target_source.rename(
        columns={
            "date": "target_date",
            "aqi": "target_aqi",
        },
        inplace=True,
    )

    result = base[
        [
            "date",
            "city",
            "pm2_5_24h_mean",
            "aqi",
            "aqi_rounded",
            "aqi_category",
        ]
    ].copy()

    for horizon, target_name in (
        [
            (1, "target_aqi_day_1"),
            (2, "target_aqi_day_2"),
            (3, "target_aqi_day_3"),
        ]
    ):

        lookup = target_source.copy()

        lookup["date"] = (
            lookup["target_date"]
            - pd.Timedelta(
                days=horizon
            )
        )

        lookup.rename(
            columns={
                "target_aqi": target_name,
            },
            inplace=True,
        )

        lookup = lookup[
            [
                "date",
                target_name,
            ]
        ]

        result = result.merge(
            lookup,
            on="date",
            how="left",
            validate="one_to_one",
        )

    return result


# ============================================================================
# VALIDATION
# ============================================================================

def validate_daily_aqi(
    daily: pd.DataFrame,
) -> None:
    """Validate calculated daily AQI data."""

    if daily.empty:
        raise ValueError(
            "Daily AQI dataset is empty."
        )

    if daily["date"].duplicated().any():
        raise ValueError(
            "Duplicate daily dates detected."
        )

    if daily["aqi"].isna().any():
        raise ValueError(
            "Missing AQI values detected."
        )

    if (
        daily["aqi"] < 0
    ).any():
        raise ValueError(
            "Negative AQI values detected."
        )

    if (
        daily["hourly_observations"] != 24
    ).any():
        raise ValueError(
            "Incomplete days remain in the AQI dataset."
        )

    logger.info(
        "Daily AQI validation passed."
    )


def validate_forecast_targets(
    targets: pd.DataFrame,
) -> None:
    """Validate future AQI target columns."""

    required = {
        "target_aqi_day_1",
        "target_aqi_day_2",
        "target_aqi_day_3",
    }

    missing = (
        required
        - set(targets.columns)
    )

    if missing:
        raise ValueError(
            "Missing forecast targets: "
            + ", ".join(sorted(missing))
        )

    logger.info(
        "Forecast target validation completed."
    )


# ============================================================================
# REPORT
# ============================================================================

def build_report(
    daily: pd.DataFrame,
    targets: pd.DataFrame,
) -> dict:
    """Create a machine-readable AQI methodology report."""

    return {
        "project": "Pearls AQI Predictor",

        "status": "calculated_target",

        "warning": (
            "AQI values in this dataset are calculated targets "
            "using a Punjab-aligned PM2.5 methodology. "
            "They are not official EPA Punjab AQI observations."
        ),

        "source_data": {
            "air_quality_source": "OpenAQ",
            "primary_pollutant": "PM2.5",
            "weather_source": "Open-Meteo",
        },

        "methodology": {
            "version": METHODOLOGY.version,
            "pm25_peqs_ug_m3": METHODOLOGY.peqs_pm25,
            "daily_basis": (
                "complete previous/local calendar day "
                "with 24 hourly observations"
            ),
            "concentration_breakpoints": (
                list(
                    METHODOLOGY.concentration_breakpoints
                )
            ),
            "aqi_breakpoints": (
                list(
                    METHODOLOGY.aqi_breakpoints
                )
            ),
            "interpolation": "linear",
        },

        "daily_observations": int(
            len(daily)
        ),

        "date_start": str(
            daily["date"].min()
        ),

        "date_end": str(
            daily["date"].max()
        ),

        "aqi_statistics": {
            "mean": float(
                daily["aqi"].mean()
            ),
            "median": float(
                daily["aqi"].median()
            ),
            "minimum": float(
                daily["aqi"].min()
            ),
            "maximum": float(
                daily["aqi"].max()
            ),
        },

        "forecast_target_coverage": {
            column: float(
                targets[column]
                .notna()
                .mean()
                * 100
            )
            for column in [
                "target_aqi_day_1",
                "target_aqi_day_2",
                "target_aqi_day_3",
            ]
        },
    }


# ============================================================================
# SAVE
# ============================================================================

def save_outputs(
    daily: pd.DataFrame,
    targets: pd.DataFrame,
    report: dict,
) -> None:
    """Save AQI datasets and methodology report."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    daily.to_parquet(
        DAILY_AQI_FILE,
        index=False,
    )

    daily.to_csv(
        DAILY_AQI_CSV,
        index=False,
    )

    targets.to_parquet(
        FORECAST_TARGET_FILE,
        index=False,
    )

    targets.to_csv(
        FORECAST_TARGET_CSV,
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
        "Daily AQI dataset saved: %s",
        DAILY_AQI_FILE,
    )

    logger.info(
        "Daily AQI CSV saved: %s",
        DAILY_AQI_CSV,
    )

    logger.info(
        "Forecast target dataset saved: %s",
        FORECAST_TARGET_FILE,
    )

    logger.info(
        "Forecast target CSV saved: %s",
        FORECAST_TARGET_CSV,
    )

    logger.info(
        "AQI methodology report saved: %s",
        REPORT_FILE,
    )


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main() -> None:
    """Execute AQI target engineering."""

    logger.info("=" * 72)
    logger.info(
        "PEARLS AQI PREDICTOR"
    )
    logger.info(
        "AQI TARGET ENGINEERING"
    )
    logger.info("=" * 72)

    hourly = load_canonical_data()

    logger.info(
        "Creating complete daily PM2.5 observations."
    )

    daily_pm25 = create_daily_pm25(
        hourly
    )

    logger.info(
        "Calculating Punjab-aligned PM2.5 AQI."
    )

    daily_aqi = build_daily_aqi(
        daily_pm25
    )

    validate_daily_aqi(
        daily_aqi
    )

    logger.info(
        "Creating Day-1 / Day-2 / Day-3 AQI targets."
    )

    forecast_targets = create_forecast_targets(
        daily_aqi
    )

    validate_forecast_targets(
        forecast_targets
    )

    report = build_report(
        daily=daily_aqi,
        targets=forecast_targets,
    )

    save_outputs(
        daily=daily_aqi,
        targets=forecast_targets,
        report=report,
    )

    logger.info(
        "Daily AQI rows: %d",
        len(daily_aqi),
    )

    logger.info(
        "Forecast target rows: %d",
        len(forecast_targets),
    )

    logger.info(
        "Day-1 target coverage: %.2f%%",
        report[
            "forecast_target_coverage"
        ][
            "target_aqi_day_1"
        ],
    )

    logger.info(
        "Day-2 target coverage: %.2f%%",
        report[
            "forecast_target_coverage"
        ][
            "target_aqi_day_2"
        ],
    )

    logger.info(
        "Day-3 target coverage: %.2f%%",
        report[
            "forecast_target_coverage"
        ][
            "target_aqi_day_3"
        ],
    )

    logger.info("=" * 72)
    logger.info(
        "AQI TARGET ENGINEERING COMPLETED."
    )
    logger.info("=" * 72)

    print(
        "\nDaily AQI preview:"
    )

    print(
        daily_aqi.head(10)
        .to_string(index=False)
    )

    print(
        "\nForecast target preview:"
    )

    print(
        forecast_targets.head(10)
        .to_string(index=False)
    )


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()