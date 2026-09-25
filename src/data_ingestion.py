"""
data_ingestion.py
------------------
Handles raw data loading and cleaning for the Energy BI project.

Data sources
------------
1. "Sites Energy Consumption Part_*.csv" — half-hourly household-level
   (cell_id) energy consumption (kWh), tagged by site_id and region.
2. "Demand.xlsx" — half-hourly national/system demand status
   (Low / Normal / High).

Both sources are messy in realistic ways: inconsistent DateTime encoding,
whitespace-polluted column names, duplicate rows, sensor-glitch outliers,
and inconsistently-cased category labels. This module normalizes all of
that into two clean, analysis-ready DataFrames.
"""

import glob
import os
import re

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
DATE_TIME_PATTERN = re.compile(
    r"^(?P<time>\d{2}:\d{2}:\d{2})\s+(?P<yy>\d{2}),(?P<dd>\d{2}),(?P<mm>\d{2})$"
)

# Half-hourly household consumption above this level (kWh) is not
# physically plausible for a single domestic meter and is treated as a
# sensor/logging glitch rather than a genuine reading.
MAX_PLAUSIBLE_KWH = 5.0

# Canonical demand-status labels the source data *should* contain.
VALID_DEMAND_LABELS = {"Low", "Normal", "High"}


# ----------------------------------------------------------------------
# Consumption data (CSV parts)
# ----------------------------------------------------------------------
def _parse_source_datetime(raw: str):
    """
    Convert the source's non-standard timestamp format into a proper
    pandas Timestamp.

    Source format: "HH:MM:SS YY,DD,MM"  e.g. "18:30:00 13,05,06"
    -> 2013-06-05 18:30:00

    Returns pd.NaT for anything that does not match the expected pattern,
    so malformed rows can be identified and dropped rather than silently
    mis-parsed.
    """
    match = DATE_TIME_PATTERN.match(str(raw).strip())
    if not match:
        return pd.NaT
    parts = match.groupdict()
    year = 2000 + int(parts["yy"])
    try:
        return pd.Timestamp(
            year=year, month=int(parts["mm"]), day=int(parts["dd"])
        ) + pd.to_timedelta(parts["time"])
    except ValueError:
        # e.g. an impossible day/month combination
        return pd.NaT


def load_consumption_data(data_dir: str) -> pd.DataFrame:
    """Load and concatenate every 'Sites Energy Consumption Part_*.csv' file."""
    file_paths = sorted(glob.glob(os.path.join(data_dir, "Sites Energy Consumption Part_*.csv")))
    if not file_paths:
        raise FileNotFoundError(
            f"No 'Sites Energy Consumption Part_*.csv' files found in '{data_dir}'."
        )

    frames = [pd.read_csv(path) for path in file_paths]
    df = pd.concat(frames, ignore_index=True)

    # Normalize column names: strip stray whitespace baked into headers
    # (e.g. "KWH/hh (per half hour) " with a trailing space).
    df.columns = [col.strip() for col in df.columns]
    df = df.rename(columns={
        "KWH/hh (per half hour)": "kwh",
        "cell_id": "cell_id",
        "DateTime": "raw_datetime",
        "site_id": "site_id",
        "region": "region",
    })
    return df


def clean_consumption_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw consumption DataFrame:
      - parse the custom timestamp encoding
      - coerce readings to numeric
      - drop exact duplicate rows
      - drop unparseable timestamps / missing readings
      - cap sensor-glitch outliers at a physically plausible ceiling
      - tidy dtypes for memory efficiency
    """
    df = df.copy()

    # Timestamp parsing
    df["timestamp"] = df["raw_datetime"].apply(_parse_source_datetime)

    # Numeric coercion (any stray non-numeric token becomes NaN)
    df["kwh"] = pd.to_numeric(df["kwh"], errors="coerce")

    before = len(df)
    df = df.drop_duplicates()
    duplicates_removed = before - len(df)

    before = len(df)
    df = df.dropna(subset=["timestamp", "kwh"])
    missing_removed = before - len(df)

    # Negative readings are physically invalid for consumption meters.
    negative_removed = int((df["kwh"] < 0).sum())
    df = df[df["kwh"] >= 0]

    # Cap (winsorize) implausible spikes rather than dropping them outright,
    # so the timeline stays continuous for time-series analysis.
    outliers_capped = int((df["kwh"] > MAX_PLAUSIBLE_KWH).sum())
    df["kwh"] = df["kwh"].clip(upper=MAX_PLAUSIBLE_KWH)

    # Memory-efficient categorical dtypes
    for col in ["cell_id", "site_id", "region"]:
        df[col] = df[col].astype("category")

    df = df.drop(columns=["raw_datetime"]).sort_values("timestamp").reset_index(drop=True)

    print(
        f"[Consumption cleaning] duplicates removed: {duplicates_removed:,} | "
        f"unparseable/missing rows removed: {missing_removed:,} | "
        f"negative readings removed: {negative_removed:,} | "
        f"outliers capped at {MAX_PLAUSIBLE_KWH} kWh: {outliers_capped:,}"
    )
    return df[["timestamp", "cell_id", "site_id", "region", "kwh"]]


# ----------------------------------------------------------------------
# Demand status data (Excel)
# ----------------------------------------------------------------------
def load_demand_data(file_path: str) -> pd.DataFrame:
    """Load the raw system-demand status file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Demand file not found: '{file_path}'.")
    return pd.read_excel(file_path)


def clean_demand_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize the 'Demand' status column, which contains inconsistent
    casing and typos (e.g. 'high', 'Normall'), and standardize column names.
    """
    df = df.copy()
    df = df.rename(columns={"DemandDateTime": "timestamp", "Demand": "demand_status"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # Normalize casing, then correct known typos to the nearest valid label.
    cleaned = df["demand_status"].astype(str).str.strip().str.capitalize()
    typo_map = {"Normall": "Normal"}
    cleaned = cleaned.replace(typo_map)

    unexpected = sorted(set(cleaned.unique()) - VALID_DEMAND_LABELS)
    if unexpected:
        print(f"[Demand cleaning] Dropping rows with unrecognized labels: {unexpected}")
    df["demand_status"] = cleaned
    df = df[df["demand_status"].isin(VALID_DEMAND_LABELS)]

    df = df.dropna(subset=["timestamp"]).drop_duplicates(subset=["timestamp"])
    df["demand_status"] = pd.Categorical(
        df["demand_status"], categories=["Low", "Normal", "High"], ordered=True
    )
    return df.sort_values("timestamp").reset_index(drop=True)


# ----------------------------------------------------------------------
# Merge
# ----------------------------------------------------------------------
def merge_datasets(consumption_df: pd.DataFrame, demand_df: pd.DataFrame) -> pd.DataFrame:
    """Left-join household consumption readings onto the system demand status
    for the same half-hour timestamp."""
    merged = consumption_df.merge(demand_df, on="timestamp", how="left")
    merged["demand_status"] = merged["demand_status"].astype(object).fillna("Unknown")
    return merged


def run_ingestion(data_dir: str) -> pd.DataFrame:
    """End-to-end ingestion + cleaning + merge pipeline. Returns the final
    analysis-ready DataFrame."""
    raw_consumption = load_consumption_data(data_dir)
    consumption = clean_consumption_data(raw_consumption)

    raw_demand = load_demand_data(os.path.join(data_dir, "Demand.xlsx"))
    demand = clean_demand_data(raw_demand)

    merged = merge_datasets(consumption, demand)
    print(f"[Ingestion] Final analysis-ready dataset: {len(merged):,} rows, "
          f"{merged['cell_id'].nunique()} meters, "
          f"{merged['timestamp'].min().date()} to {merged['timestamp'].max().date()}")
    return merged
