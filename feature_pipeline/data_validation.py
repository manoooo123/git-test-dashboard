"""
Production-grade data validation pipeline
for Pearls AQI Predictor.

Purpose:
- Validate real OpenAQ observations.
- Preserve real measurements.
- Detect schema/type problems.
- Detect duplicates.
- Analyse sensor coverage.
- Analyse temporal coverage.
- Detect invalid pollutant values.
- Generate an auditable quality report.
- Save validated data as Parquet.

No synthetic observations are created.
No missing observations are artificially filled.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports" / "data_quality"

INPUT_FILE = RAW_DATA_DIR / "openaq_lahore.csv"

OUTPUT_FILE = (
    PROCESSED_DATA_DIR / "validated_openaq_lahore.parquet"
)

REPORT_FILE = (
    REPORTS_DIR / "openaq_quality_report.json"
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
# CONFIGURATION
# ============================================================================

@dataclass(frozen=True)
class ValidationConfig:
    """Validation rules for OpenAQ observations."""

    required_columns: tuple[str, ...] = (
        "timestamp",
        "sensor_id",
        "parameter",
        "value",
    )

    supported_parameters: tuple[str, ...] = (
        "pm25",
        "pm10",
    )

    minimum_value: float = 0.0
    maximum_value: float = 1000.0


CONFIG = ValidationConfig()


# ============================================================================
# CUSTOM EXCEPTION
# ============================================================================

class DataValidationError(RuntimeError):
    """Raised when critical dataset validation fails."""


# ============================================================================
# LOAD DATA
# ============================================================================

def load_raw_data(path: Path) -> pd.DataFrame:
    """Load the raw OpenAQ CSV dataset."""

    if not path.exists():
        raise FileNotFoundError(
            f"Raw OpenAQ dataset not found: {path}"
        )

    logger.info(
        "Loading raw dataset: %s",
        path,
    )

    dataframe = pd.read_csv(path)

    if dataframe.empty:
        raise DataValidationError(
            "Raw OpenAQ dataset is empty."
        )

    logger.info(
        "Raw dataset loaded: rows=%d | columns=%d",
        len(dataframe),
        len(dataframe.columns),
    )

    return dataframe


# ============================================================================
# SCHEMA VALIDATION
# ============================================================================

def validate_schema(dataframe: pd.DataFrame) -> None:
    """Validate required dataset columns."""

    missing_columns = [
        column
        for column in CONFIG.required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise DataValidationError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    logger.info("Schema validation passed.")


# ============================================================================
# TYPE NORMALIZATION
# ============================================================================

def normalize_types(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize timestamp, sensor, parameter and measurement types."""

    dataframe = dataframe.copy()

    dataframe["timestamp"] = pd.to_datetime(
        dataframe["timestamp"],
        utc=True,
        errors="coerce",
    )

    dataframe["sensor_id"] = pd.to_numeric(
        dataframe["sensor_id"],
        errors="coerce",
    )

    dataframe["parameter"] = (
        dataframe["parameter"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    dataframe["value"] = pd.to_numeric(
        dataframe["value"],
        errors="coerce",
    )

    logger.info("Data types normalized.")

    return dataframe


# ============================================================================
# STRUCTURAL VALIDATION
# ============================================================================

def remove_invalid_rows(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    """
    Remove rows that cannot represent a valid observation.

    Only structurally invalid records are removed.
    """

    dataframe = dataframe.copy()

    before = len(dataframe)

    dataframe.dropna(
        subset=[
            "timestamp",
            "sensor_id",
            "parameter",
            "value",
        ],
        inplace=True,
    )

    removed = before - len(dataframe)

    logger.info(
        "Structurally invalid rows removed: %d",
        removed,
    )

    return dataframe, removed


# ============================================================================
# PARAMETER ANALYSIS
# ============================================================================

def analyse_parameters(
    dataframe: pd.DataFrame,
) -> dict[str, int]:
    """Analyse pollutant parameters present in the dataset."""

    counts = (
        dataframe["parameter"]
        .value_counts(dropna=False)
        .to_dict()
    )

    logger.info(
        "Parameters detected: %s",
        counts,
    )

    unsupported = sorted(
        set(counts)
        - set(CONFIG.supported_parameters)
    )

    if unsupported:
        logger.warning(
            "Unsupported parameters detected: %s",
            unsupported,
        )

    return {
        str(parameter): int(count)
        for parameter, count in counts.items()
    }


# ============================================================================
# RANGE VALIDATION
# ============================================================================

def validate_measurement_ranges(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    """
    Detect obviously invalid pollutant measurements.

    Values outside the configured physical range are removed.
    No replacement or synthetic value is generated.
    """

    dataframe = dataframe.copy()

    invalid_mask = (
        (dataframe["value"] < CONFIG.minimum_value)
        | (
            dataframe["value"]
            > CONFIG.maximum_value
        )
    )

    invalid_count = int(
        invalid_mask.sum()
    )

    if invalid_count:
        logger.warning(
            "Invalid measurement values detected: %d",
            invalid_count,
        )

    dataframe = dataframe.loc[
        ~invalid_mask
    ].copy()

    return dataframe, invalid_count


# ============================================================================
# DUPLICATE ANALYSIS
# ============================================================================

def analyse_duplicates(
    dataframe: pd.DataFrame,
) -> dict[str, int]:
    """Analyse duplicate records."""

    exact_duplicates = int(
        dataframe.duplicated().sum()
    )

    observation_duplicates = int(
        dataframe.duplicated(
            subset=[
                "timestamp",
                "sensor_id",
                "parameter",
            ]
        ).sum()
    )

    logger.info(
        "Exact duplicate rows: %d",
        exact_duplicates,
    )

    logger.info(
        "Sensor-time-parameter duplicates: %d",
        observation_duplicates,
    )

    return {
        "exact_duplicate_rows": exact_duplicates,
        "sensor_time_parameter_duplicates": (
            observation_duplicates
        ),
    }


# ============================================================================
# SENSOR COVERAGE
# ============================================================================

def build_sensor_coverage(
    dataframe: pd.DataFrame,
) -> list[dict]:
    """Generate sensor-level coverage statistics."""

    coverage = (
        dataframe
        .groupby(
            ["sensor_id", "parameter"],
            dropna=False,
        )
        .agg(
            observations=("value", "size"),
            start=("timestamp", "min"),
            end=("timestamp", "max"),
            mean_value=("value", "mean"),
            minimum_value=("value", "min"),
            maximum_value=("value", "max"),
        )
        .reset_index()
    )

    coverage["duration_hours"] = (
        (
            coverage["end"]
            - coverage["start"]
        )
        / pd.Timedelta(hours=1)
    )

    coverage["expected_hourly_observations"] = (
        coverage["duration_hours"] + 1
    )

    coverage["coverage_ratio"] = (
        coverage["observations"]
        / coverage[
            "expected_hourly_observations"
        ]
    ).clip(upper=1.0)

    coverage.sort_values(
        "observations",
        ascending=False,
        inplace=True,
    )

    logger.info(
        "Unique sensors: %d",
        dataframe["sensor_id"].nunique(),
    )

    return coverage.to_dict(
        orient="records"
    )


# ============================================================================
# TEMPORAL COVERAGE
# ============================================================================

def build_temporal_coverage(
    dataframe: pd.DataFrame,
) -> dict:
    """
    Analyse unique timestamps.

    Missing timestamps are reported only.
    They are NOT filled.
    """

    timestamps = (
        dataframe["timestamp"]
        .drop_duplicates()
        .sort_values()
    )

    if timestamps.empty:
        return {
            "unique_hours": 0,
            "expected_hours": 0,
            "missing_hours": 0,
        }

    expected = pd.date_range(
        start=timestamps.min(),
        end=timestamps.max(),
        freq="h",
        tz="UTC",
    )

    actual = pd.DatetimeIndex(
        timestamps
    )

    missing = expected.difference(
        actual
    )

    report = {
        "unique_hours": int(len(actual)),
        "expected_hours": int(len(expected)),
        "missing_hours": int(len(missing)),
        "start": str(actual.min()),
        "end": str(actual.max()),
    }

    logger.info(
        "Unique timestamps: %d",
        report["unique_hours"],
    )

    logger.info(
        "Expected hourly slots: %d",
        report["expected_hours"],
    )

    logger.info(
        "Missing hourly slots: %d",
        report["missing_hours"],
    )

    return report


# ============================================================================
# FINAL VALIDATION
# ============================================================================

def validate_final_dataset(
    dataframe: pd.DataFrame,
) -> None:
    """Perform final integrity checks."""

    if dataframe.empty:
        raise DataValidationError(
            "Validated dataset is empty."
        )

    if dataframe["timestamp"].isna().any():
        raise DataValidationError(
            "Invalid timestamps remain."
        )

    if dataframe["sensor_id"].isna().any():
        raise DataValidationError(
            "Missing sensor IDs remain."
        )

    if dataframe["value"].isna().any():
        raise DataValidationError(
            "Missing measurement values remain."
        )

    logger.info(
        "Final dataset validation passed."
    )


# ============================================================================
# QUALITY REPORT
# ============================================================================

def create_quality_report(
    dataframe: pd.DataFrame,
    removed_structural: int,
    removed_range: int,
) -> dict:
    """Build machine-readable data-quality report."""

    duplicates = analyse_duplicates(
        dataframe
    )

    parameters = analyse_parameters(
        dataframe
    )

    sensor_coverage = build_sensor_coverage(
        dataframe
    )

    temporal_coverage = build_temporal_coverage(
        dataframe
    )

    report = {
        "project": "Pearls AQI Predictor",
        "dataset": "OpenAQ Lahore",
        "source": "OpenAQ",
        "validation_status": "passed",

        "rows": int(len(dataframe)),
        "columns": list(
            dataframe.columns
        ),

        "unique_sensors": int(
            dataframe["sensor_id"].nunique()
        ),

        "parameters": parameters,

        "removed_structural_rows": int(
            removed_structural
        ),

        "removed_invalid_measurements": int(
            removed_range
        ),

        "duplicates": duplicates,

        "temporal_coverage": (
            temporal_coverage
        ),

        "value_statistics": {
            "mean": float(
                dataframe["value"].mean()
            ),
            "median": float(
                dataframe["value"].median()
            ),
            "minimum": float(
                dataframe["value"].min()
            ),
            "maximum": float(
                dataframe["value"].max()
            ),
        },

        "sensor_coverage": sensor_coverage,
    }

    return report


# ============================================================================
# SAVE
# ============================================================================

def save_outputs(
    dataframe: pd.DataFrame,
    report: dict,
) -> None:
    """Save validated dataset and quality report."""

    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    with REPORT_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=2,
            default=str,
        )

    logger.info(
        "Validated dataset saved: %s",
        OUTPUT_FILE,
    )

    logger.info(
        "Quality report saved: %s",
        REPORT_FILE,
    )


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_validation() -> pd.DataFrame:
    """Execute the complete OpenAQ validation pipeline."""

    logger.info("=" * 72)
    logger.info(
        "PEARLS AQI PREDICTOR"
    )
    logger.info(
        "PRODUCTION DATA VALIDATION"
    )
    logger.info("=" * 72)

    dataframe = load_raw_data(
        INPUT_FILE
    )

    validate_schema(
        dataframe
    )

    dataframe = normalize_types(
        dataframe
    )

    dataframe, removed_structural = (
        remove_invalid_rows(
            dataframe
        )
    )

    dataframe, removed_range = (
        validate_measurement_ranges(
            dataframe
        )
    )

    dataframe.sort_values(
        [
            "timestamp",
            "sensor_id",
        ],
        inplace=True,
    )

    dataframe.reset_index(
        drop=True,
        inplace=True,
    )

    validate_final_dataset(
        dataframe
    )

    report = create_quality_report(
        dataframe=dataframe,
        removed_structural=(
            removed_structural
        ),
        removed_range=(
            removed_range
        ),
    )

    save_outputs(
        dataframe=dataframe,
        report=report,
    )

    logger.info("=" * 72)
    logger.info(
        "OPENAQ DATA VALIDATION COMPLETED"
    )
    logger.info("=" * 72)

    return dataframe


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    run_validation()