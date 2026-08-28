"""
Production baseline evaluation for Pearls AQI Predictor.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_features.csv"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "model_evaluation"
)

REPORT_FILE = (
    REPORT_DIR
    / "baseline_evaluation_report.json"
)


HORIZONS = {
    24: {
        "target": "target_pm2_5_24h",
        "lag": "pm2_5_lag_24h",
    },
    48: {
        "target": "target_pm2_5_48h",
        "lag": "pm2_5_lag_48h",
    },
    72: {
        "target": "target_pm2_5_72h",
        "lag": "pm2_5_lag_72h",
    },
}


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


def calculate_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> dict[str, float]:
    """Calculate regression metrics."""

    return {
        "mae": float(
            mean_absolute_error(
                y_true,
                y_pred,
            )
        ),
        "rmse": float(
            np.sqrt(
                mean_squared_error(
                    y_true,
                    y_pred,
                )
            )
        ),
        "r2": float(
            r2_score(
                y_true,
                y_pred,
            )
        ),
    }


def main() -> None:
    """Run persistence baseline evaluation."""

    logger.info("=" * 72)
    logger.info("PEARLS AQI PREDICTOR")
    logger.info("BASELINE MODEL EVALUATION")
    logger.info("=" * 72)

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Feature dataset not found: {DATA_FILE}"
        )

    df = pd.read_csv(DATA_FILE)

    logger.info(
        "Dataset shape: %s",
        df.shape,
    )

    results: dict[str, dict] = {}

    for horizon, config in HORIZONS.items():

        target_column = config["target"]
        lag_column = config["lag"]

        subset = (
            df[
                [
                    target_column,
                    lag_column,
                ]
            ]
            .dropna()
            .copy()
        )

        if subset.empty:
            raise ValueError(
                f"No valid observations for {horizon}h."
            )

        y_true = subset[target_column]
        y_pred = subset[lag_column]

        metrics = calculate_metrics(
            y_true,
            y_pred,
        )

        results[str(horizon)] = {
            "horizon_hours": horizon,
            "baseline": "persistence_lag",
            "samples": int(len(subset)),
            "target_column": target_column,
            "prediction_column": lag_column,
            "metrics": metrics,
        }

        logger.info(
            "%dh Persistence | MAE=%.4f | RMSE=%.4f | R2=%.4f",
            horizon,
            metrics["mae"],
            metrics["rmse"],
            metrics["r2"],
        )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_FILE.write_text(
        json.dumps(
            {
                "dataset": "Pearls AQI Predictor",
                "baseline_type": "persistence_lag",
                "results": results,
            },
            indent=4,
        ),
        encoding="utf-8",
    )

    logger.info(
        "Baseline report saved: %s",
        REPORT_FILE,
    )

    logger.info("=" * 72)
    logger.info(
        "BASELINE EVALUATION COMPLETED SUCCESSFULLY."
    )
    logger.info("=" * 72)


if __name__ == "__main__":
    main()