from pathlib import Path

import pandas as pd

from src.imd_data import (
    open_rainfall_dataset,
    extract_daily_rainfall,
    summarize_rainfall,
)
from src.location import GeographicLocation
from src.rainfall_states import classify_rainfall

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

DEFAULT_LOCATION_ID = "yavatmal"
DEFAULT_LATITUDE = 20.39
DEFAULT_LONGITUDE = 78.13
DEFAULT_START_YEAR = 2019
DEFAULT_END_YEAR = 2024


def process_year(
    year,
    *,
    latitude,
    longitude,
    location_id,
    raw_dir=RAW_DIR,
    processed_dir=PROCESSED_DIR,
):
    """Extract and preprocess one IMD yearly rainfall dataset."""

    input_file = raw_dir / f"RF25_ind{year}_rfp25.nc"
    output_file = (
        processed_dir
        / f"rainfall_{location_id}_{year}.csv"
    )

    if not input_file.exists():
        raise FileNotFoundError(f"Missing input file: {input_file}")

    print(f"\nProcessing {year}...")
    print(f"Input: {input_file}")

    dataset = open_rainfall_dataset(input_file)

    try:
        location = GeographicLocation(
            latitude=latitude,
            longitude=longitude,
        )

        extracted = extract_daily_rainfall(
            dataset,
            location,
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

    # Convert rainfall into the canonical weather states
    # used by the Markov Chain model.
    df["rainfall_state"] = df["rainfall_mm"].apply(
        classify_rainfall
    )

    processed_dir.mkdir(parents=True, exist_ok=True)
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


def main(
    *,
    location_id=DEFAULT_LOCATION_ID,
    latitude=DEFAULT_LATITUDE,
    longitude=DEFAULT_LONGITUDE,
    start_year=DEFAULT_START_YEAR,
    end_year=DEFAULT_END_YEAR,
    raw_dir=RAW_DIR,
    processed_dir=PROCESSED_DIR,
):
    """Process all requested IMD years for a location."""

    if not isinstance(location_id, str) or not location_id.strip():
        raise ValueError("location_id must not be empty.")

    if start_year > end_year:
        raise ValueError("start_year must not be greater than end_year.")

    print("=" * 60)
    print("AgriNova — IMD Multi-Year Rainfall Preprocessing")
    print("=" * 60)
    print(f"Location: {location_id}")
    print(f"Latitude: {latitude}")
    print(f"Longitude: {longitude}")
    print(f"Years: {start_year}-{end_year}")

    for year in range(start_year, end_year + 1):
        process_year(
            year,
            latitude=latitude,
            longitude=longitude,
            location_id=location_id,
            raw_dir=raw_dir,
            processed_dir=processed_dir,
        )

    print("\n" + "=" * 60)
    print("All years processed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()