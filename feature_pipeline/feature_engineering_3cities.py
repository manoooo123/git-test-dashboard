from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "hourly_3cities.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "model_features_3cities.csv"


def build_city_features(city_df: pd.DataFrame) -> pd.DataFrame:
    city_df = city_df.copy()
    city_df = city_df.sort_values("hour").set_index("hour")

    full_range = pd.date_range(
        start=city_df.index.min(),
        end=city_df.index.max(),
        freq="1h",
        tz=city_df.index.tz,
    )

    city_df = city_df.reindex(full_range)
    city_df.index.name = "hour"

    city_name = city_df["city"].dropna().iloc[0]
    city_df["city"] = city_name

    city_df["is_missing_hour"] = city_df["pm25_mean"].isna()

    # ---------------------------------------------------------
    # GAP-AWARE LAG FEATURES
    # A lag is valid only when the required previous hour exists.
    # ---------------------------------------------------------
    pm25 = city_df["pm25_mean"]

    for lag in [1, 3, 6, 12, 24]:
        col = f"pm25_lag_{lag}h"
        lag_values = pm25.shift(lag)

        previous_missing = (
            city_df["is_missing_hour"]
            .shift(lag, fill_value=True)
            .astype(bool)
        )

        city_df[col] = lag_values.mask(previous_missing)

    # ---------------------------------------------------------
    # TIME FEATURES
    # ---------------------------------------------------------
    city_df["hour_of_day"] = city_df.index.hour
    city_df["day_of_week"] = city_df.index.dayofweek
    city_df["month"] = city_df.index.month
    city_df["day_of_year"] = city_df.index.dayofyear
    city_df["is_weekend"] = (
        city_df["day_of_week"] >= 5
    ).astype(int)

    city_df["hour_sin"] = np.sin(
        2 * np.pi * city_df["hour_of_day"] / 24
    )
    city_df["hour_cos"] = np.cos(
        2 * np.pi * city_df["hour_of_day"] / 24
    )

    city_df["month_sin"] = np.sin(
        2 * np.pi * city_df["month"] / 12
    )
    city_df["month_cos"] = np.cos(
        2 * np.pi * city_df["month"] / 12
    )

    # ---------------------------------------------------------
    # ROLLING FEATURES
    # Only previous observations are used.
    # ---------------------------------------------------------
    shifted = pm25.shift(1)

    city_df["pm25_rolling_mean_3h"] = shifted.rolling(
        window=3,
        min_periods=3,
    ).mean()

    city_df["pm25_rolling_mean_6h"] = shifted.rolling(
        window=6,
        min_periods=6,
    ).mean()

    city_df["pm25_rolling_mean_24h"] = shifted.rolling(
        window=24,
        min_periods=24,
    ).mean()

    city_df["pm25_rolling_std_24h"] = shifted.rolling(
        window=24,
        min_periods=24,
    ).std()

    # If any missing hour exists inside the rolling window,
    # invalidate that rolling feature.
    missing_numeric = city_df["is_missing_hour"].astype(int)

    for window, col in [
        (3, "pm25_rolling_mean_3h"),
        (6, "pm25_rolling_mean_6h"),
        (24, "pm25_rolling_mean_24h"),
        (24, "pm25_rolling_std_24h"),
    ]:
        bad_window = (
            missing_numeric.shift(1, fill_value=0)
            .rolling(window=window, min_periods=window)
            .sum()
            > 0
        )
        city_df.loc[bad_window, col] = np.nan

    # ---------------------------------------------------------
    # PM2.5 CHANGE FEATURES
    # ---------------------------------------------------------
    city_df["pm25_change_1h"] = (
        city_df["pm25_mean"] - city_df["pm25_lag_1h"]
    )

    city_df["pm25_change_24h"] = (
        city_df["pm25_mean"] - city_df["pm25_lag_24h"]
    )

    # ---------------------------------------------------------
    # FUTURE TARGETS
    # ---------------------------------------------------------
    for horizon in [24, 48, 72]:
        target_col = f"target_{horizon}h"
        target = pm25.shift(-horizon)

        future_missing = (
            city_df["is_missing_hour"]
            .shift(-horizon, fill_value=True)
            .astype(bool)
        )

        city_df[target_col] = target.mask(future_missing)

    # ---------------------------------------------------------
    # RESTORE ORIGINAL TIMESTAMP COLUMN
    # ---------------------------------------------------------
    city_df = city_df.reset_index()

    return city_df


def main() -> None:
    print("=" * 70)
    print("PEARLS AQI | GAP-AWARE FEATURE ENGINEERING")
    print("=" * 70)

    df = pd.read_csv(
        INPUT_FILE,
        parse_dates=["hour"],
    )

    df = df.sort_values(
        ["city", "hour"]
    ).reset_index(drop=True)

    print("INPUT SHAPE:", df.shape)

    results = []

    for city in sorted(df["city"].dropna().unique()):
        city_df = df[df["city"] == city].copy()

        print()
        print(f"PROCESSING: {city}")
        print("Original rows:", len(city_df))
        print(
            "Original time:",
            city_df["hour"].min(),
            "->",
            city_df["hour"].max(),
        )

        result = build_city_features(city_df)

        print("Regularized rows:", len(result))
        print(
            "Missing hours:",
            int(result["is_missing_hour"].sum()),
        )

        results.append(result)

    final_df = pd.concat(
        results,
        ignore_index=True,
    )

    final_df = final_df.sort_values(
        ["city", "hour"]
    ).reset_index(drop=True)

    final_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print("=" * 70)
    print("TARGET AVAILABILITY")
    print("=" * 70)

    print(
        final_df[
            ["target_24h", "target_48h", "target_72h"]
        ]
        .notna()
        .sum()
        .to_string()
    )

    print()
    print("TARGET AVAILABILITY BY CITY:")

    print(
        final_df.groupby("city")[
            ["target_24h", "target_48h", "target_72h"]
        ]
        .count()
        .to_string()
    )

    print()
    print("=" * 70)
    print("FEATURE ENGINEERING COMPLETE")
    print("=" * 70)
    print("OUTPUT SHAPE:", final_df.shape)
    print("SAVED:", OUTPUT_FILE)
    print("=" * 70)


if __name__ == "__main__":
    main()
