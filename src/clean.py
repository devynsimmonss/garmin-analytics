"""
clean.py — Clean and normalize raw Garmin data into analysis-ready DataFrames.
"""

from pathlib import Path

import json
import numpy as np
import pandas as pd


# ── Unit conversion helpers ────────────────────────────────────────────────────

def _ms_to_datetime(ms_series: pd.Series, utc: bool = True) -> pd.Series:
    """Convert millisecond epoch timestamps to timezone-aware datetimes."""
    return pd.to_datetime(ms_series, unit="ms", utc=utc)


def _ms_to_seconds(ms_series: pd.Series) -> pd.Series:
    return ms_series / 1000


def _cm_to_miles(cm_series: pd.Series) -> pd.Series:
    # JSON summarizedActivities stores distance in centimeters
    return cm_series / 160_934.0


def _mps_to_pace(mps_series: pd.Series) -> pd.Series:
    """Convert m/s to min/mile pace. Returns NaN for zero/null speed."""
    with np.errstate(divide="ignore", invalid="ignore"):
        pace = np.where(
            mps_series > 0,
            (1 / (mps_series * 0.000621371)) / 60,
            np.nan,
        )
    return pd.Series(pace, index=mps_series.index)


# ── Columns to drop from summarizedActivities ─────────────────────────────────

# Internal IDs, flags, and sport-specific fields not useful for running analysis
_DROP_COLS = [
    "uuidMsb", "uuidLsb", "userProfileId", "eventTypeId", "rule",
    "timeZoneId", "beginTimestamp",
    # Running-specific form metrics replaced by cleaner derived columns
    "avgRunCadence", "avgFractionalCadence", "maxFractionalCadence",
    "maxDoubleCadence", "maxRunCadence",
    # Power — mostly null for GPS running (no power meter/stryd)
    "avgPower", "maxPower", "normPower",
    "powerTimeInZone_0", "powerTimeInZone_1", "powerTimeInZone_2",
    "powerTimeInZone_3", "powerTimeInZone_4", "powerTimeInZone_5",
    "isRunPowerWindDataEnabled", "runPowerWindDataEnabled",
    # Dive / strength — irrelevant for running analysis
    "summarizedDiveInfo", "decoDive",
    "summarizedExerciseSets", "activeSets", "totalSets", "totalReps",
    # Nested split objects — complex; parse separately if needed
    "splitSummaries", "splits",
    # Boolean flags with low analytical value
    "parent", "atpActivity", "purposeful", "autoCalcCalories",
    "elevationCorrected", "favorite", "pr",
    # Bounding-box coords (start/end already kept)
    "maxLatitude", "maxLongitude", "minLatitude", "minLongitude",
    # Rarely populated
    "workoutId", "avgVerticalSpeed", "waterEstimated",
    # HR zone 0 (below Zone 1) and zone 6 (above max) — edge noise
    "hrTimeInZone_0", "hrTimeInZone_6",
    # Manufacturer constant
    "manufacturer",
]


def clean_activities(json_path: str | Path) -> pd.DataFrame:
    """Load and clean the Garmin summarizedActivities JSON export.

    Performs the following transformations:
    - Flattens the nested export structure into a flat DataFrame
    - Converts millisecond epoch timestamps to UTC datetimes
    - Converts duration from ms → seconds; adds duration_min
    - Converts distance from meters → miles
    - Converts avgSpeed / maxSpeed from m/s → min/mile pace
    - Renames avgDoubleCadence → cadence_spm (steps per minute, both feet)
    - Drops internal, flag, and sport-specific columns irrelevant to running

    Args:
        json_path: Path to *_summarizedActivities.json.

    Returns:
        Cleaned DataFrame with one row per activity.
    """
    json_path = Path(json_path)
    with open(json_path) as f:
        raw = json.load(f)

    activities = raw[0]["summarizedActivitiesExport"]
    df = pd.DataFrame(activities)

    # ── Timestamps ────────────────────────────────────────────────────────────
    df["datetime_utc"] = _ms_to_datetime(df["startTimeGmt"], utc=True)
    # Local time stored as UTC-offset ms; convert then strip tz for display
    df["datetime_local"] = _ms_to_datetime(df["startTimeLocal"], utc=True).dt.tz_localize(None)
    df = df.drop(columns=["startTimeGmt", "startTimeLocal"])

    # ── Duration ──────────────────────────────────────────────────────────────
    df["duration_s"] = _ms_to_seconds(df["duration"])
    df["duration_min"] = df["duration_s"] / 60
    df["moving_duration_s"] = _ms_to_seconds(df["movingDuration"])
    df["elapsed_duration_s"] = _ms_to_seconds(df["elapsedDuration"])
    df = df.drop(columns=["duration", "movingDuration", "elapsedDuration"])

    # ── Distance ──────────────────────────────────────────────────────────────
    # Garmin JSON exports distance in centimeters (confirmed empirically via
    # cadence × stride-length cross-check against stored avgSpeed)
    df["distance_miles"] = _cm_to_miles(df["distance"])
    df = df.drop(columns=["distance"])

    # ── Speed → Pace ──────────────────────────────────────────────────────────
    # avgSpeed / maxSpeed are stored in cm/ms, which is numerically 10× m/s
    # (1 cm/ms = 10 m/s). Multiply by 10 before passing to pace formula.
    df["avg_pace_min_per_mile"] = _mps_to_pace(df["avgSpeed"] * 10)
    df["max_pace_min_per_mile"] = _mps_to_pace(df["maxSpeed"] * 10)
    df = df.drop(columns=["avgSpeed", "maxSpeed"])

    # ── Cadence ───────────────────────────────────────────────────────────────
    # avgDoubleCadence = total steps per minute (both feet); the true running cadence
    df = df.rename(columns={"avgDoubleCadence": "cadence_spm"})

    # ── Stride / form metrics ─────────────────────────────────────────────────
    # avgStrideLength is stored in centimeters (step length, single foot contact)
    if "avgStrideLength" in df.columns:
        df["avgStrideLength"] = df["avgStrideLength"] / 100  # → meters

    # ── Elevation ─────────────────────────────────────────────────────────────
    # elevationGain / elevationLoss appear to be in centimeters based on the
    # same export format; divide by 100 for meters.
    for col in ("elevationGain", "elevationLoss", "minElevation", "maxElevation"):
        if col in df.columns:
            df[col] = df[col] / 100

    # ── Drop irrelevant columns ───────────────────────────────────────────────
    cols_to_drop = [c for c in _DROP_COLS if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    # ── Sort chronologically ──────────────────────────────────────────────────
    df = df.sort_values("datetime_utc").reset_index(drop=True)

    return df


def filter_runs(df: pd.DataFrame) -> pd.DataFrame:
    """Return only outdoor and treadmill running activities.

    Args:
        df: DataFrame from clean_activities().

    Returns:
        Filtered DataFrame, index reset.
    """
    run_types = {"running", "treadmill_running"}
    return df[df["activityType"].isin(run_types)].reset_index(drop=True)


def clean_fit_records(df: pd.DataFrame) -> pd.DataFrame:
    """Clean a per-second FIT record DataFrame from parse.load_fit_file().

    - Drops rows missing both GPS position and heart rate (pure metadata rows)
    - Sorts by timestamp
    - Flags paused segments where speed_ms == 0

    Args:
        df: DataFrame from parse.load_fit_file().

    Returns:
        Cleaned DataFrame, index reset.
    """
    if df.empty:
        return df

    df = df.copy()

    # Drop rows that have no position AND no heart rate — not useful records
    has_position = df.get("position_lat", pd.Series(dtype=float)).notna()
    has_hr = df.get("heart_rate", pd.Series(dtype=float)).notna()
    df = df[has_position | has_hr].copy()

    # Sort by timestamp
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp").reset_index(drop=True)

    # Flag paused segments (device still recording, athlete stopped)
    if "speed_ms" in df.columns:
        df["is_paused"] = df["speed_ms"].fillna(0) == 0

    return df.reset_index(drop=True)
