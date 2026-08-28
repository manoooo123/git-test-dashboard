from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "dataset_3cities.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "clean_dataset_3cities.csv"

REQUIRED_COLUMNS = [
    "location_id",
    "sensor_id",
    "timestamp",
    "pm2_5",
    "temperature",
    "humidity",
    "pressure",
    "clouds",
    "wind_speed",
    "wind_direction",
    "precipitation",
    "city",
]

EXPECTED_CITIES = {"Lahore", "Islamabad", "Faisalabad"}


def main() -> None:
    print("=" * 70)
    print("PEARLS AQI | DATA VALIDATION")
    print("=" * 70)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    print(f"Input file : {INPUT_FILE}")
    print(f"Shape      : {df.shape}")

    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
        utc=True,
    )

    print("\nCITY DISTRIBUTION")
    print(df["city"].value_counts())

    actual_cities = set(df["city"].dropna().unique())
    print(f"\nExpected cities: {EXPECTED_CITIES}")
    print(f"Actual cities  : {actual_cities}")

    print("\nDUPLICATES")
    print(f"Duplicate rows: {df.duplicated().sum()}")

    print("\nMISSING VALUES")
    print(df[REQUIRED_COLUMNS].isna().sum())

    print("\nTIMESTAMP")
    print(f"Invalid timestamps: {df['timestamp'].isna().sum()}")
    print(f"Start: {df['timestamp'].min()}")
    print(f"End  : {df['timestamp'].max()}")

    print("\nPM2.5")
    print(f"Missing: {df['pm2_5'].isna().sum()}")
    print(f"Min   : {df['pm2_5'].min()}")
    print(f"Max   : {df['pm2_5'].max()}")
    print(f"Mean  : {df['pm2_5'].mean():.2f}")

    print("\nSENSORS")
    print(f"Unique sensors   : {df['sensor_id'].nunique()}")
    print(f"Unique locations : {df['location_id'].nunique()}")

    print("\nWEATHER MATCH")
    weather_columns = [
        "temperature",
        "humidity",
        "pressure",
        "clouds",
        "wind_speed",
        "wind_direction",
        "precipitation",
    ]

    weather_missing = df[weather_columns].isna().any(axis=1).sum()
    weather_match = (1 - weather_missing / len(df)) * 100

    print(f"Rows with weather data: {len(df) - weather_missing}")
    print(f"Weather coverage      : {weather_match:.2f}%")

    print("\nVALIDATION")
    if actual_cities != EXPECTED_CITIES:
        raise ValueError("City validation failed.")

    if df["timestamp"].isna().any():
        raise ValueError("Invalid timestamps found.")

    if df["pm2_5"].isna().any():
        raise ValueError("Missing PM2.5 values found.")

    if (df["pm2_5"] < 0).any():
        raise ValueError("Negative PM2.5 values found.")

    print("PASS: Dataset structure and core AQI data are valid.")

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(f"\nValidated dataset saved:")
    print(OUTPUT_FILE)

    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
