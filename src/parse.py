"""
parse.py — Load and extract records from Garmin FIT files into a pandas DataFrame.
"""

from pathlib import Path

import fitparse
import pandas as pd


def load_fit_file(fit_path: str | Path) -> pd.DataFrame:
    """Parse a Garmin FIT file and return a DataFrame of 'record' messages.

    'record' messages contain per-second GPS/sensor data: position, speed,
    heart rate, cadence, altitude, etc.

    Args:
        fit_path: Path to the .fit file.

    Returns:
        DataFrame with one row per record message. Coordinate columns
        (position_lat, position_long) are converted from semicircles to
        decimal degrees. Speed is provided in both m/s (speed_ms) and
        min/mile pace (pace_min_per_mile).
    """
    fit_path = Path(fit_path)
    if not fit_path.exists():
        raise FileNotFoundError(f"FIT file not found: {fit_path}")

    fitfile = fitparse.FitFile(str(fit_path))

    rows = []
    for record in fitfile.get_messages("record"):
        row = {field.name: field.value for field in record}
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Convert semicircles → decimal degrees
    # Garmin stores GPS as signed 32-bit integers (semicircles)
    semicircle_to_deg = 180 / 2**31
    for col in ("position_lat", "position_long"):
        if col in df.columns:
            df[col] = df[col] * semicircle_to_deg

    # Derive pace columns from speed (m/s).
    # Newer FIT files use enhanced_speed; older ones use speed.
    speed_col = "enhanced_speed" if "enhanced_speed" in df.columns else "speed" if "speed" in df.columns else None
    if speed_col:
        df = df.rename(columns={speed_col: "speed_ms"})
        # Avoid division by zero for stopped/stationary records
        df["pace_min_per_mile"] = df["speed_ms"].apply(
            lambda s: (1 / (s * 0.000621371)) / 60 if s and s > 0 else None
        )

    # Parse timestamp to a proper datetime if present
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    return df
