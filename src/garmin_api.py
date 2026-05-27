"""
garmin_api.py — Garmin Connect API integration via the unofficial garminconnect library.

Usage:
    from src.garmin_api import authenticate, fetch_recent_activities
    client = authenticate()
    df = fetch_recent_activities(client, days=60)

Credentials are loaded from a .env file (GARMIN_EMAIL, GARMIN_PASSWORD).
Never hardcode credentials or commit them to version control.
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv


# ── Authentication ─────────────────────────────────────────────────────────────

def authenticate(
    email: str | None = None,
    password: str | None = None,
    env_path: str | Path | None = None,
) -> "garth.Client":
    """Authenticate with Garmin Connect and return an active client.

    Credentials are resolved in this order:
    1. Explicit email/password arguments
    2. GARMIN_EMAIL / GARMIN_PASSWORD environment variables
    3. .env file in the project root (or env_path if provided)

    Garmin Connect uses OAuth2 + MFA challenges internally; garminconnect
    handles the session management automatically and caches tokens in
    ~/.garth/ so repeated calls won't re-prompt for credentials.

    Args:
        email: Garmin account email. Overrides .env if provided.
        password: Garmin account password. Overrides .env if provided.
        env_path: Path to .env file. Defaults to project root .env.

    Returns:
        Authenticated Garmin client object.
    """
    from garminconnect import Garmin

    if env_path is None:
        env_path = Path(__file__).resolve().parents[1] / ".env"
    load_dotenv(dotenv_path=env_path)

    email = email or os.getenv("GARMIN_EMAIL")
    password = password or os.getenv("GARMIN_PASSWORD")

    if not email or not password:
        raise ValueError(
            "Garmin credentials not found. Set GARMIN_EMAIL and GARMIN_PASSWORD "
            "in your .env file or pass them explicitly."
        )

    def _mfa_prompt():
        return input("  Enter Garmin MFA/2FA code: ").strip()

    client = Garmin(email=email, password=password, prompt_mfa=_mfa_prompt)
    client.login()
    return client


# ── Activity fetching ──────────────────────────────────────────────────────────

def fetch_recent_activities(
    client,
    days: int = 60,
    activity_type: str | None = None,
) -> pd.DataFrame:
    """Fetch recent activities from Garmin Connect and normalize to project schema.

    Pulls up to `days` days of history. The returned DataFrame uses the same
    column names as clean_activities() so it can be passed directly into
    build_features().

    Args:
        client: Authenticated Garmin client from authenticate().
        days: How many days back to fetch (default 60).
        activity_type: Filter to this activity type (e.g. "running"). None = all.

    Returns:
        DataFrame normalized to the internal project schema.
    """
    end_date = datetime.now(tz=timezone.utc)
    start_date = end_date - timedelta(days=days)

    raw_activities = client.get_activities_by_date(
        startdate=start_date.strftime("%Y-%m-%d"),
        enddate=end_date.strftime("%Y-%m-%d"),
        activitytype=activity_type or "",
    )

    if not raw_activities:
        return pd.DataFrame()

    rows = []
    for act in raw_activities:
        row = _normalize_activity(act)
        if row is not None:
            rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.sort_values("datetime_local").reset_index(drop=True)
    return df


def _normalize_activity(act: dict) -> dict | None:
    """Map a single Garmin Connect API activity dict to the project column schema.

    The Connect API returns slightly different field names than the JSON export.
    This function bridges the two so fetch_recent_activities() output is
    drop-in compatible with clean_activities() output.
    """
    try:
        # ── Timestamps ────────────────────────────────────────────────────────
        # API returns ISO strings: "2025-03-01T10:30:00.0" (local) and "...Z" (UTC)
        local_str = act.get("startTimeLocal") or act.get("startTimeGmt", "")
        utc_str = act.get("startTimeGmt", "")

        datetime_local = pd.to_datetime(local_str)
        datetime_utc = pd.to_datetime(utc_str, utc=True)

        # ── Activity type ─────────────────────────────────────────────────────
        act_type_dict = act.get("activityType", {})
        if isinstance(act_type_dict, dict):
            activity_type = act_type_dict.get("typeKey", "").lower()
        else:
            activity_type = str(act_type_dict).lower()

        # ── Duration ──────────────────────────────────────────────────────────
        # Connect API duration field is in seconds (not milliseconds)
        duration_s = float(act.get("duration", 0) or 0)
        duration_min = duration_s / 60
        moving_duration_s = float(act.get("movingDuration", duration_s) or duration_s)
        elapsed_duration_s = float(act.get("elapsedDuration", duration_s) or duration_s)

        # ── Distance ──────────────────────────────────────────────────────────
        # Connect API distance is in meters
        distance_m = float(act.get("distance", 0) or 0)
        distance_miles = distance_m / 1609.34

        # ── Pace ──────────────────────────────────────────────────────────────
        # API avgSpeed in m/s
        avg_speed_mps = float(act.get("averageSpeed", 0) or 0)
        max_speed_mps = float(act.get("maxSpeed", 0) or 0)

        avg_pace = (1 / (avg_speed_mps * 0.000621371)) / 60 if avg_speed_mps > 0 else np.nan
        max_pace = (1 / (max_speed_mps * 0.000621371)) / 60 if max_speed_mps > 0 else np.nan

        # ── HR ────────────────────────────────────────────────────────────────
        avg_hr = act.get("averageHR") or act.get("avgHr")
        max_hr_val = act.get("maxHR") or act.get("maxHr")

        # ── Cadence ───────────────────────────────────────────────────────────
        # Connect API avgRunningCadenceInStepsPerMinute = total SPM (both feet)
        cadence_spm = (
            act.get("avgRunningCadenceInStepsPerMinute")
            or act.get("averageRunningCadenceInStepsPerMinute")
        )

        # ── Elevation ─────────────────────────────────────────────────────────
        # Connect API elevation fields are in meters
        elevation_gain = act.get("elevationGain")
        elevation_loss = act.get("elevationLoss")
        min_elevation = act.get("minElevation")
        max_elevation = act.get("maxElevation")

        # ── VO2Max ────────────────────────────────────────────────────────────
        vo2max = act.get("vO2MaxValue") or act.get("vo2MaxPreciseValue")

        # ── HR zone time ──────────────────────────────────────────────────────
        # Connect API returns hrTimeInZone as list or dict; normalize to zone1–5 ms
        hr_zones_raw = act.get("hrTimeInZone", [])
        zone_ms = _parse_hr_zone_times(hr_zones_raw)

        # ── Stride length ─────────────────────────────────────────────────────
        avg_stride = act.get("avgStrideLength")  # already in meters from API

        row = {
            "activityId": act.get("activityId"),
            "activityName": act.get("activityName", ""),
            "activityType": activity_type,
            "datetime_local": datetime_local,
            "datetime_utc": datetime_utc,
            "duration_s": duration_s,
            "duration_min": duration_min,
            "moving_duration_s": moving_duration_s,
            "elapsed_duration_s": elapsed_duration_s,
            "distance_miles": distance_miles,
            "avg_pace_min_per_mile": avg_pace,
            "max_pace_min_per_mile": max_pace,
            "avgHr": avg_hr,
            "maxHr": max_hr_val,
            "cadence_spm": cadence_spm,
            "avgStrideLength": avg_stride,
            "elevationGain": elevation_gain,
            "elevationLoss": elevation_loss,
            "minElevation": min_elevation,
            "maxElevation": max_elevation,
            "vO2MaxValue": vo2max,
            "calories": act.get("calories"),
            "steps": act.get("steps"),
            "startLatitude": act.get("startLatitude"),
            "startLongitude": act.get("startLongitude"),
            **zone_ms,
        }
        return row

    except Exception as e:
        print(f"  Warning: could not normalize activity {act.get('activityId')}: {e}")
        return None


def _parse_hr_zone_times(hr_zones_raw) -> dict:
    """Extract hrTimeInZone_1–5 in milliseconds from the API's zone time field.

    The Connect API may return zone times as a list of dicts or already as
    a flat dict. Converts all values to milliseconds to match the JSON export
    schema expected by add_hr_zone_pcts().
    """
    result = {f"hrTimeInZone_{i}": np.nan for i in range(1, 6)}
    if not hr_zones_raw:
        return result

    if isinstance(hr_zones_raw, list):
        for item in hr_zones_raw:
            if isinstance(item, dict):
                zone_num = item.get("zoneNumber") or item.get("zone")
                seconds = item.get("secsInZone") or item.get("seconds") or 0
                if zone_num and 1 <= int(zone_num) <= 5:
                    result[f"hrTimeInZone_{int(zone_num)}"] = float(seconds) * 1000
    elif isinstance(hr_zones_raw, dict):
        for k, v in hr_zones_raw.items():
            for i in range(1, 6):
                if str(i) in str(k):
                    result[f"hrTimeInZone_{i}"] = float(v or 0) * 1000

    return result


# ── Incremental update ─────────────────────────────────────────────────────────

def update_from_api(
    existing_df: pd.DataFrame,
    client,
    lookback_days: int = 14,
) -> pd.DataFrame:
    """Fetch new activities and append to an existing DataFrame.

    Only activities newer than the latest in existing_df (minus 1 day buffer)
    are fetched. Deduplicates on activityId.

    Args:
        existing_df: Existing normalized DataFrame (may be empty).
        client: Authenticated Garmin client.
        lookback_days: Days to look back when existing_df is empty.

    Returns:
        Combined DataFrame, sorted by datetime_local, duplicates removed.
    """
    if not existing_df.empty and "datetime_local" in existing_df.columns:
        latest = pd.to_datetime(existing_df["datetime_local"]).max()
        days_back = max(2, (datetime.now() - latest.replace(tzinfo=None)).days + 1)
    else:
        days_back = lookback_days

    new_df = fetch_recent_activities(client, days=days_back)

    if new_df.empty:
        return existing_df

    combined = pd.concat([existing_df, new_df], ignore_index=True)

    if "activityId" in combined.columns:
        combined = combined.drop_duplicates(subset="activityId", keep="last")

    return combined.sort_values("datetime_local").reset_index(drop=True)
