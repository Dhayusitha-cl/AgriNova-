from pathlib import Path

import pandas as pd

from src.imd_data import (
    open_rainfall_dataset,
    extract_daily_rainfall,
    summarize_rainfall,
)


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

LATITUDE = 20.39
LONGITUDE = 78.13

START_YEAR = 2019
END_YEAR = 2024


def classify_rainfall(rainfall_mm):
    """Convert rainfall amount into the project's weather states.

    States:
    - dry: 0 mm
    - drizzle: greater than 0 and less than 10 mm
    - rain: 10 mm or more
    """

    if pd.isna(rainfall_mm):
        raise ValueError("Rainfall value cannot be missing.")

    if rainfall_mm < 0:
        raise ValueError("Rainfall cannot be negative.")

    if rainfall_mm == 0:
        return "dry"
    elif rainfall_mm < 10:
        return "drizzle"
    else:
        return "rain"


def process_year(year):
    """Extract and preprocess one IMD yearly rainfall dataset."""

    input_file = RAW_DIR / f"RF25_ind{year}_rfp25.nc"
    output_file = PROCESSED_DIR / f"rainfall_yavatmal_{year}.csv"

    if not input_file.exists():
        raise FileNotFoundError(f"Missing input file: {input_file}")

    print(f"\nProcessing {year}...")
    print(f"Input: {input_file}")

    dataset = open_rainfall_dataset(input_file)

    try:
        extracted = extract_daily_rainfall(
            dataset,
            latitude=LATITUDE,
            longitude=LONGITUDE,
        )

        summary = summarize_rainfall(extracted)

    finally:
        dataset.close()

    rainfall = extracted["rainfall_mm"]

    if len(rainfall) != summary["days"]:
        raise ValueError(f"Unexpected rainfall length for {year}.")

    df = pd.DataFrame(
        {
            "date": pd.to_datetime(extracted["time"]),
            "rainfall_mm": rainfall,
            "latitude": extracted["selected_latitude"],
            "longitude": extracted["selected_longitude"],
        }
    )

    # Ensure chronological ordering.
    df = df.sort_values("date").reset_index(drop=True)

    # Validate rainfall values.
    if df["rainfall_mm"].isna().any():
        raise ValueError(f"Missing rainfall found in {year}.")

    if (df["rainfall_mm"] < 0).any():
        raise ValueError(f"Negative rainfall found in {year}.")

    # Convert rainfall into the weather states used by
    # the Markov Chain model.
    df["rainfall_state"] = df["rainfall_mm"].apply(classify_rainfall)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)

    print(
        f"Selected grid: "
        f"{extracted['selected_latitude']}, "
        f"{extracted['selected_longitude']}"
    )

    print(f"Days: {len(df)}")
    print(f"Mean rainfall: {summary['mean_mm']:.3f} mm")
    print(f"Zero-rain days: {summary['zero_rain_days']}")
    print(f"Output: {output_file}")

    print("Rainfall states:")
    print(df["rainfall_state"].value_counts().to_dict())

    return df


def main():
    """Process all requested IMD years."""

    print("=" * 60)
    print("AgriNova — IMD Multi-Year Rainfall Preprocessing")
    print("=" * 60)

    for year in range(START_YEAR, END_YEAR + 1):
        process_year(year)

    print("\n" + "=" * 60)
    print("All years processed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()