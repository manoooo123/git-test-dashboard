import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EVALUATION_FILE = (
    PROJECT_ROOT
    / "reports"
    / "model_evaluation"
    / "production_evaluation.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "model_comparison"
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 72)
    logger.info("PEARLS AQI PREDICTOR")
    logger.info("MODEL COMPARISON ANALYSIS")
    logger.info("=" * 72)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger.info(
        "Loading evaluation file: %s",
        EVALUATION_FILE,
    )

    if not EVALUATION_FILE.exists():
        raise FileNotFoundError(
            f"Evaluation file not found: {EVALUATION_FILE}"
        )

    df = pd.read_csv(EVALUATION_FILE)

    logger.info(
        "Evaluation rows: %d",
        len(df),
    )

    logger.info(
        "Columns: %s",
        list(df.columns),
    )

    df.columns = [
        str(column).strip().lower()
        for column in df.columns
    ]

    required_columns = {
        "horizon_hours",
        "model",
        "mae",
        "rmse",
        "r2",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(missing))
        )

    df["horizon_hours"] = pd.to_numeric(
        df["horizon_hours"],
        errors="coerce",
    )

    df["horizon"] = df["horizon_hours"].map(
        {
            24: "24h",
            48: "48h",
            72: "72h",
        }
    )

    df["model"] = (
        df["model"]
        .astype(str)
        .str.strip()
    )

    for column in ["mae", "rmse", "r2"]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.dropna(
        subset=[
            "horizon",
            "model",
            "mae",
            "rmse",
            "r2",
        ]
    )

    cleaned_file = (
        OUTPUT_DIR
        / "model_comparison_results.csv"
    )

    df.to_csv(
        cleaned_file,
        index=False,
    )

    logger.info(
        "Comparison CSV saved: %s",
        cleaned_file,
    )

    horizon_order = [
        "24h",
        "48h",
        "72h",
    ]

    best_models = []

    for horizon in horizon_order:

        horizon_df = df[
            df["horizon"] == horizon
        ]

        if horizon_df.empty:
            continue

        best_row = horizon_df.loc[
            horizon_df["r2"].idxmax()
        ]

        best_models.append(
            {
                "horizon": horizon,
                "best_model": str(
                    best_row["model"]
                ),
                "mae": float(
                    best_row["mae"]
                ),
                "rmse": float(
                    best_row["rmse"]
                ),
                "r2": float(
                    best_row["r2"]
                ),
            }
        )

    best_models_file = (
        OUTPUT_DIR
        / "best_models_summary.json"
    )

    with best_models_file.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            best_models,
            file,
            indent=4,
        )

    logger.info(
        "Best-model summary saved: %s",
        best_models_file,
    )

    logger.info("=" * 72)
    logger.info("BEST MODEL SUMMARY")
    logger.info("=" * 72)

    for result in best_models:

        logger.info(
            "%s | %s | MAE=%.4f | RMSE=%.4f | R2=%.4f",
            result["horizon"],
            result["best_model"],
            result["mae"],
            result["rmse"],
            result["r2"],
        )

    # R2 plot
    r2_pivot = df.pivot(
        index="horizon",
        columns="model",
        values="r2",
    ).reindex(horizon_order)

    ax = r2_pivot.plot(
        kind="bar",
        figsize=(10, 6),
    )

    ax.set_title(
        "AQI Forecasting Model Comparison - R2"
    )

    ax.set_xlabel(
        "Forecast Horizon"
    )

    ax.set_ylabel(
        "R2 Score"
    )

    ax.set_ylim(
        0,
        1,
    )

    ax.grid(
        axis="y",
        alpha=0.3,
    )

    plt.tight_layout()

    r2_plot = (
        OUTPUT_DIR
        / "r2_model_comparison.png"
    )

    plt.savefig(
        r2_plot,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    logger.info(
        "R2 comparison plot saved: %s",
        r2_plot,
    )

    # MAE plot
    mae_pivot = df.pivot(
        index="horizon",
        columns="model",
        values="mae",
    ).reindex(horizon_order)

    ax = mae_pivot.plot(
        kind="bar",
        figsize=(10, 6),
    )

    ax.set_title(
        "AQI Forecasting Model Comparison - MAE"
    )

    ax.set_xlabel(
        "Forecast Horizon"
    )

    ax.set_ylabel(
        "Mean Absolute Error"
    )

    ax.grid(
        axis="y",
        alpha=0.3,
    )

    plt.tight_layout()

    mae_plot = (
        OUTPUT_DIR
        / "mae_model_comparison.png"
    )

    plt.savefig(
        mae_plot,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    logger.info(
        "MAE comparison plot saved: %s",
        mae_plot,
    )

    # RMSE plot
    rmse_pivot = df.pivot(
        index="horizon",
        columns="model",
        values="rmse",
    ).reindex(horizon_order)

    ax = rmse_pivot.plot(
        kind="bar",
        figsize=(10, 6),
    )

    ax.set_title(
        "AQI Forecasting Model Comparison - RMSE"
    )

    ax.set_xlabel(
        "Forecast Horizon"
    )

    ax.set_ylabel(
        "Root Mean Squared Error"
    )

    ax.grid(
        axis="y",
        alpha=0.3,
    )

    plt.tight_layout()

    rmse_plot = (
        OUTPUT_DIR
        / "rmse_model_comparison.png"
    )

    plt.savefig(
        rmse_plot,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    logger.info(
        "RMSE comparison plot saved: %s",
        rmse_plot,
    )

    logger.info("=" * 72)
    logger.info(
        "MODEL COMPARISON COMPLETED SUCCESSFULLY."
    )
    logger.info(
        "Reports saved to: %s",
        OUTPUT_DIR,
    )
    logger.info("=" * 72)


if __name__ == "__main__":
    main()