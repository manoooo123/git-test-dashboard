from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_features.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "reports" / "eda"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


def load_data() -> pd.DataFrame:
    """Load the production feature dataset."""
    logger.info("Loading dataset: %s", INPUT_FILE)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
        utc=True,
    )

    df = df.sort_values("timestamp").reset_index(drop=True)

    logger.info("Dataset shape: %s", df.shape)

    return df


def generate_summary(df: pd.DataFrame) -> None:
    """Generate statistical and data-quality summary."""

    summary = {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "start_timestamp": str(df["timestamp"].min()),
        "end_timestamp": str(df["timestamp"].max()),
        "missing_values": {
            str(k): int(v)
            for k, v in df.isna().sum().items()
            if v > 0
        },
        "numeric_statistics": {},
    }

    numeric_df = df.select_dtypes(include="number")

    for column in numeric_df.columns:
        summary["numeric_statistics"][column] = {
            "mean": float(numeric_df[column].mean()),
            "std": float(numeric_df[column].std()),
            "min": float(numeric_df[column].min()),
            "max": float(numeric_df[column].max()),
        }

    output_file = OUTPUT_DIR / "eda_summary.json"

    with output_file.open("w", encoding="utf-8") as file:
        json.dump(
            summary,
            file,
            indent=4,
            default=str,
        )

    logger.info("EDA summary saved: %s", output_file)


def plot_pm25_trend(df: pd.DataFrame) -> None:
    """Plot PM2.5 time-series trend."""

    if "pm2_5" not in df.columns:
        logger.warning("pm2_5 column not found.")
        return

    plt.figure(figsize=(14, 6))

    plt.plot(
        df["timestamp"],
        df["pm2_5"],
        linewidth=0.8,
    )

    plt.title("PM2.5 Time-Series Trend")
    plt.xlabel("Time")
    plt.ylabel("PM2.5 (µg/m³)")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    output_file = OUTPUT_DIR / "pm25_trend.png"

    plt.savefig(
        output_file,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    logger.info("Saved PM2.5 trend: %s", output_file)


def plot_daily_pm25(df: pd.DataFrame) -> None:
    """Plot daily average PM2.5."""

    if "pm2_5" not in df.columns:
        return

    temp = df.copy()

    temp["date"] = temp["timestamp"].dt.date

    daily = (
        temp.groupby("date")["pm2_5"]
        .mean()
        .reset_index()
    )

    plt.figure(figsize=(14, 6))

    plt.plot(
        daily["date"],
        daily["pm2_5"],
        linewidth=1,
    )

    plt.title("Daily Average PM2.5")
    plt.xlabel("Date")
    plt.ylabel("Average PM2.5 (µg/m³)")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    output_file = OUTPUT_DIR / "daily_pm25.png"

    plt.savefig(
        output_file,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    logger.info("Saved daily PM2.5: %s", output_file)


def plot_hourly_pattern(df: pd.DataFrame) -> None:
    """Analyze hourly PM2.5 pattern."""

    if "pm2_5" not in df.columns:
        return

    temp = df.copy()

    temp["hour"] = temp["timestamp"].dt.hour

    hourly = (
        temp.groupby("hour")["pm2_5"]
        .mean()
        .reset_index()
    )

    plt.figure(figsize=(12, 6))

    plt.plot(
        hourly["hour"],
        hourly["pm2_5"],
        marker="o",
    )

    plt.title("Average PM2.5 by Hour")
    plt.xlabel("Hour of Day")
    plt.ylabel("Average PM2.5 (µg/m³)")
    plt.xticks(range(24))
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    output_file = OUTPUT_DIR / "hourly_pm25_pattern.png"

    plt.savefig(
        output_file,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    logger.info("Saved hourly pattern: %s", output_file)


def plot_correlation(df: pd.DataFrame) -> None:
    """Create numeric correlation matrix."""

    numeric_df = df.select_dtypes(include="number")

    correlation = numeric_df.corr()

    plt.figure(figsize=(14, 11))

    plt.imshow(
        correlation,
        aspect="auto",
    )

    plt.colorbar()

    plt.xticks(
        range(len(correlation.columns)),
        correlation.columns,
        rotation=90,
        fontsize=7,
    )

    plt.yticks(
        range(len(correlation.columns)),
        correlation.columns,
        fontsize=7,
    )

    plt.title("Feature Correlation Matrix")

    plt.tight_layout()

    output_file = OUTPUT_DIR / "correlation_matrix.png"

    plt.savefig(
        output_file,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    logger.info("Saved correlation matrix: %s", output_file)


def main() -> None:
    """Run complete exploratory data analysis."""

    logger.info("=" * 72)
    logger.info("PEARLS AQI PREDICTOR")
    logger.info("EXPLORATORY DATA ANALYSIS")
    logger.info("=" * 72)

    df = load_data()

    generate_summary(df)

    plot_pm25_trend(df)

    plot_daily_pm25(df)

    plot_hourly_pattern(df)

    plot_correlation(df)

    logger.info("=" * 72)
    logger.info("EDA COMPLETED SUCCESSFULLY")
    logger.info("Reports saved to: %s", OUTPUT_DIR)
    logger.info("=" * 72)


if __name__ == "__main__":
    main()