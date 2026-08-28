"""
Pearls AQI Predictor
====================

Reduced, production-oriented AQI feature dataset.

Purpose
-------
Reduce the daily AQI feature space to a compact set of
historically available predictors.

No future target columns are used as features.
No synthetic observations are created.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd


# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "aqi_model_features.parquet"
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

OUTPUT_FILE = (
    OUTPUT_DIR
    / "aqi_reduced_features.parquet"
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "aqi_reduced_features.csv"
)

REPORT_FILE = (
    REPORT_DIR
    / "aqi_reduced_feature_report.json"
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
# FEATURE POLICY
# ============================================================================

SELECTED_FEATURES = [
    # Current known state
    "aqi",
    "pm2_5_24h_mean",

    # Historical AQI
    "aqi_lag_1d",
    "aqi_lag_2d",
    "aqi_lag_3d",
    "aqi_lag_7d",

    # Historical PM2.5
    "pm2_5_lag_1d",
    "pm2_5_lag_3d",
    "pm2_5_lag_7d",

    # Historical AQI behaviour
    "aqi_rolling_mean_3d",
    "aqi_rolling_mean_7d",
    "aqi_rolling_std_3d",
    "aqi_rolling_std_7d",

    # Historical PM2.5 behaviour
    "pm2_5_rolling_mean_3d",
    "pm2_5_rolling_mean_7d",

    # Weather available at forecast origin
    "temperature_mean",
    "humidity_mean",
    "pressure_mean",
    "wind_speed_mean",
    "precipitation_total",

    # Calendar
    "month",
    "day_of_week",
]


TARGETS = [
    "target_aqi_day_1",
    "target_aqi_day_2",
    "target_aqi_day_3",
]


# ============================================================================
# LOAD
# ============================================================================

def load_dataset() -> pd.DataFrame:
    """Load the previously engineered daily AQI dataset."""

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input dataset not found: {INPUT_FILE}"
        )

    df = pd.read_parquet(
        INPUT_FILE
    )

    if df.empty:
        raise ValueError(
            "Input AQI feature dataset is empty."
        )

    logger.info(
        "Input shape: %s",
        df.shape,
    )

    return df


# ============================================================================
# VALIDATION
# ============================================================================

def validate_columns(
    df: pd.DataFrame,
) -> None:
    """Validate required columns."""

    required = {
        "date",
        "city",
        *SELECTED_FEATURES,
        *TARGETS,
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(
                sorted(missing)
            )
        )


# ============================================================================
# BUILD REDUCED DATASET
# ============================================================================

def build_reduced_dataset(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Build the compact forecasting dataset."""

    columns = [
        "date",
        "city",
        *SELECTED_FEATURES,
        *TARGETS,
    ]

    reduced = (
        df[columns]
        .copy()
        .sort_values("date")
        .drop_duplicates("date")
        .reset_index(drop=True)
    )

    return reduced


# ============================================================================
# FUTURE LEAKAGE CHECK
# ============================================================================

def leakage_check(
    df: pd.DataFrame,
) -> None:
    """
    Ensure target columns do not appear among predictors.
    """

    overlap = (
        set(SELECTED_FEATURES)
        & set(TARGETS)
    )

    if overlap:
        raise ValueError(
            "Target leakage detected: "
            + ", ".join(
                sorted(overlap)
            )
        )

    logger.info(
        "Target leakage check passed."
    )


# ============================================================================
# REPORT
# ============================================================================

def build_report(
    df: pd.DataFrame,
) -> dict:
    """Build machine-readable feature-selection report."""

    return {
        "project": "Pearls AQI Predictor",
        "input_file": str(INPUT_FILE),
        "output_file": str(OUTPUT_FILE),
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "predictor_count": len(
            SELECTED_FEATURES
        ),
        "predictors": SELECTED_FEATURES,
        "targets": TARGETS,
        "date_start": str(
            df["date"].min()
        ),
        "date_end": str(
            df["date"].max()
        ),
        "missing_values": {
            column: int(
                df[column].isna().sum()
            )
            for column in df.columns
        },
    }


# ============================================================================
# SAVE
# ============================================================================

def save_outputs(
    df: pd.DataFrame,
    report: dict,
) -> None:
    """Save reduced AQI feature dataset."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_parquet(
        OUTPUT_FILE,
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
        "Reduced feature dataset saved: %s",
        OUTPUT_FILE,
    )

    logger.info(
        "Reduced feature CSV saved: %s",
        OUTPUT_CSV,
    )

    logger.info(
        "Feature report saved: %s",
        REPORT_FILE,
    )


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    """Execute AQI feature reduction."""

    logger.info("=" * 72)
    logger.info(
        "PEARLS AQI PREDICTOR"
    )
    logger.info(
        "REDUCED AQI FEATURE ENGINEERING"
    )
    logger.info("=" * 72)

    df = load_dataset()

    validate_columns(
        df
    )

    leakage_check(
        df
    )

    reduced = build_reduced_dataset(
        df
    )

    report = build_report(
        reduced
    )

    save_outputs(
        reduced,
        report,
    )

    logger.info(
        "Reduced dataset shape: %s",
        reduced.shape,
    )

    logger.info(
        "Predictor count: %d",
        len(SELECTED_FEATURES),
    )

    logger.info("=" * 72)
    logger.info(
        "REDUCED AQI FEATURE DATASET READY."
    )
    logger.info("=" * 72)


if __name__ == "__main__":
    main()