"""
features.py — Feature engineering for running analytics.

Functions are composable: each takes a DataFrame and returns a copy with new
columns added. Call build_features() to apply the full pipeline at once.
"""

from pathlib import Path

import json
import numpy as np
import pandas as pd


# ── HR zone config ─────────────────────────────────────────────────────────────

def load_hr_zones(zones_path: str | Path) -> dict:
    """Load HR zone configuration from a Garmin heartRateZones JSON file.

    Returns:
        Dict with keys zone1Floor–zone5Floor, restingHeartRateUsed, maxHeartRateUsed.
    """
    with open(Path(zones_path)) as f:
        data = json.load(f)
    return next((z for z in data if z.get("sport") == "DEFAULT"), data[0])


def assign_hr_zone(
    hr_series: pd.Series,
    zones: dict | None = None,
    zones_path: str | Path | None = None,
) -> pd.Series:
    """Map per-second heart rate values to HR zones 1–5 (for FIT record DataFrames).

    Args:
        hr_series: Series of integer bpm values.
        zones: Dict from load_hr_zones(). Takes precedence over zones_path.
        zones_path: Path to heartRateZones.json if zones not provided.

    Returns:
        Int64 Series of zone labels (1–5). NaN where hr_series is NaN.
    """
    if zones is None:
        if zones_path is None:
            raise ValueError("Provide either zones or zones_path.")
        zones = load_hr_zones(zones_path)

    floors = [zones[f"zone{i}Floor"] for i in range(1, 6)]

    def _zone(hr):
        if pd.isna(hr):
            return pd.NA
        for i, floor in enumerate(floors):
            if hr < floor:
                return i + 1
        return 5

    return hr_series.map(_zone).astype("Int64")


# ── 1. Time features ───────────────────────────────────────────────────────────

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar and time-of-day columns from datetime_local.

    New columns: year, month, week_of_year, day_of_week (0=Mon), day_name,
    hour, is_weekend, season, days_into_training (days since first run).
    """
    df = df.copy()
    dt = df["datetime_local"]

    df["year"] = dt.dt.year
    df["month"] = dt.dt.month
    df["week_of_year"] = dt.dt.isocalendar().week.astype(int)
    df["day_of_week"] = dt.dt.dayofweek        # 0 = Monday
    df["day_name"] = dt.dt.day_name()
    df["hour"] = dt.dt.hour
    df["is_weekend"] = dt.dt.dayofweek >= 5

    season_map = {
        12: "Winter", 1: "Winter", 2: "Winter",
        3: "Spring", 4: "Spring", 5: "Spring",
        6: "Summer", 7: "Summer", 8: "Summer",
        9: "Fall", 10: "Fall", 11: "Fall",
    }
    df["season"] = dt.dt.month.map(season_map)

    # Days elapsed since the athlete's first recorded run — shows training age
    first_run = dt.min()
    df["days_into_training"] = (dt - first_run).dt.days

    return df


# ── 2. Run metrics ─────────────────────────────────────────────────────────────

def add_run_metrics(
    df: pd.DataFrame,
    rest_hr: int = 49,
    max_hr: int = 198,
) -> pd.DataFrame:
    """Add per-run derived metrics: numeric pace, effort zone, grade-adjusted pace.

    New columns:
      pace_sec_per_mile   — pace as a plain float (seconds/mile) for regression
      hr_intensity_ratio  — (avgHr - rest_hr) / (max_hr - rest_hr); 0–1 scale
      effort_zone         — Easy / Moderate / Hard / Very Hard / Maximum
      net_grade_pct       — net elevation gradient (positive = net uphill)
      grade_adj_pace      — pace adjusted for hills; comparable across routes
    """
    df = df.copy()

    # Numeric pace (easier to use as a regression target than min:sec strings)
    df["pace_sec_per_mile"] = df["avg_pace_min_per_mile"] * 60

    # HR intensity: fraction of usable HR range; basis for TRIMP and effort zone
    df["hr_intensity_ratio"] = (df["avgHr"] - rest_hr) / (max_hr - rest_hr)
    df["hr_intensity_ratio"] = df["hr_intensity_ratio"].clip(0, 1)

    # Effort zone from HR intensity ratio
    # Thresholds match Karvonen zones: easy < 60%, mod 60–70%, hard 70–80%, etc.
    bins = [0, 0.60, 0.70, 0.80, 0.90, 1.01]
    labels = ["Easy", "Moderate", "Hard", "Very Hard", "Maximum"]
    df["effort_zone"] = pd.cut(
        df["hr_intensity_ratio"], bins=bins, labels=labels, right=False
    )

    # Grade-adjusted pace: correct for net elevation to compare hilly vs flat runs
    # Uses Minetti's simplified linear approximation: ~4% pace shift per 1% grade
    if "elevationGain" in df.columns and "elevationLoss" in df.columns:
        dist_m = df["distance_miles"] * 1609.34
        net_rise_m = df["elevationGain"] - df["elevationLoss"].fillna(0)
        df["net_grade_pct"] = (net_rise_m / dist_m * 100).where(dist_m > 0)
        df["grade_adj_pace"] = df["avg_pace_min_per_mile"] / (
            1 + 0.04 * df["net_grade_pct"].fillna(0)
        )

    return df


# ── 3. HR zone time distribution ───────────────────────────────────────────────

def add_hr_zone_pcts(df: pd.DataFrame) -> pd.DataFrame:
    """Convert raw hrTimeInZone columns (ms) into percentages and minutes.

    Garmin stores hrTimeInZone_1–5 in milliseconds. Zones 0 and 6 (edge noise)
    are excluded from the denominator since they were dropped in clean.py.

    New columns: zone{n}_pct (% of total zone time), zone{n}_min,
    polarization_index (fraction of time in zones 1–2 vs zone 3).
    """
    df = df.copy()
    zone_cols = [f"hrTimeInZone_{i}" for i in range(1, 6)]
    present = [c for c in zone_cols if c in df.columns]
    if not present:
        return df

    total_zone_ms = df[present].sum(axis=1).replace(0, np.nan)

    for col in present:
        n = col.split("_")[-1]
        df[f"zone{n}_pct"] = df[col] / total_zone_ms * 100
        df[f"zone{n}_min"] = df[col] / 60_000  # ms → minutes

    # Polarization index: time at extremes (Z1+Z2) vs middle (Z3)
    # High values suggest polarized training; low suggests pyramidal/threshold
    z_low = df.get("zone1_pct", 0) + df.get("zone2_pct", 0)
    z_mid = df.get("zone3_pct", pd.Series(np.nan, index=df.index)).replace(0, np.nan)
    df["polarization_index"] = z_low / z_mid

    return df


# ── 4. Efficiency features ─────────────────────────────────────────────────────

def add_efficiency_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add aerobic efficiency and running economy proxies.

    New columns:
      efficiency_factor — speed (mph) / avgHr; higher = faster for same cardiac cost
      aerobic_decoupling_proxy — pace vs HR ratio vs cadence; rough drift indicator
    """
    df = df.copy()

    # Efficiency factor: speed in mph divided by average HR
    # Tracks fitness over time — a rising EF at the same effort means improved economy
    speed_mph = df["distance_miles"] / (df["duration_min"] / 60)
    df["efficiency_factor"] = (speed_mph / df["avgHr"] * 1000).round(3)

    # Cadence-normalised stride length: stride length at a given cadence
    # Useful for detecting form changes independent of pace
    if "avgStrideLength" in df.columns:
        df["stride_index"] = (
            df["avgStrideLength"] / (df["cadence_spm"] / 160)
        ).round(3)  # length normalised to 160 spm reference cadence

    return df


# ── 5. Training load (TRIMP, ATL, CTL, TSB) ────────────────────────────────────

def add_training_load(
    df: pd.DataFrame,
    rest_hr: int = 49,
    max_hr: int = 198,
) -> pd.DataFrame:
    """Add per-run TRIMP and cumulative training load metrics (ATL, CTL, TSB).

    TRIMP (Training Impulse) = duration_min × hr_intensity_ratio
    ATL  (Acute Training Load, "fitness in last 7 days")  — τ = 7 days
    CTL  (Chronic Training Load, "base fitness")           — τ = 42 days
    TSB  (Training Stress Balance = CTL − ATL, "form")
      Positive TSB → fresh/recovered; negative → fatigued

    Runs hr_intensity_ratio must already exist (call add_run_metrics first),
    or will be computed here using the provided rest_hr / max_hr.

    New columns: trimp, atl, ctl, tsb.
    """
    df = df.copy().sort_values("datetime_local").reset_index(drop=True)

    if "hr_intensity_ratio" not in df.columns:
        df["hr_intensity_ratio"] = (
            (df["avgHr"] - rest_hr) / (max_hr - rest_hr)
        ).clip(0, 1)

    df["trimp"] = df["duration_min"] * df["hr_intensity_ratio"]

    # Build a daily TRIMP series (fill gaps with 0 on rest days)
    df["_date"] = df["datetime_local"].dt.normalize()
    daily_trimp = df.groupby("_date")["trimp"].sum()

    all_dates = pd.date_range(daily_trimp.index.min(), daily_trimp.index.max(), freq="D")
    daily_trimp = daily_trimp.reindex(all_dates, fill_value=0.0)

    # Exponentially weighted ATL (τ=7) and CTL (τ=42)
    # EWM span s gives τ ≈ s/2, so span=14 → τ≈7, span=84 → τ≈42
    atl_series = daily_trimp.ewm(span=14, adjust=False).mean()
    ctl_series = daily_trimp.ewm(span=84, adjust=False).mean()

    # Map daily values back to each activity by date
    df["atl"] = df["_date"].map(atl_series).round(2)
    df["ctl"] = df["_date"].map(ctl_series).round(2)
    df["tsb"] = (df["ctl"] - df["atl"]).round(2)

    df = df.drop(columns=["_date"])
    return df


# ── 6. Rolling and temporal trend features ─────────────────────────────────────

def add_rolling_features(
    df: pd.DataFrame,
    windows: tuple[int, ...] = (7, 28),
) -> pd.DataFrame:
    """Add rolling mileage, run frequency, and recovery gap features.

    New columns:
      miles_{n}d         — total miles run in preceding n days
      runs_{n}d          — number of runs in preceding n days
      avg_pace_{n}d      — average pace over preceding n days
      weekly_miles       — total miles in the ISO calendar week of each run
      weekly_miles_chg   — % change vs previous week (injury-risk signal)
      days_since_last_run — rest days between this and the prior run
    """
    df = df.copy().sort_values("datetime_local").reset_index(drop=True)
    df["_date"] = df["datetime_local"].dt.normalize()

    # Daily aggregates (handles double-run days correctly)
    daily_miles = df.groupby("_date")["distance_miles"].sum()
    daily_runs = df.groupby("_date")["activityId"].count()
    daily_pace = df.groupby("_date")["avg_pace_min_per_mile"].mean()

    all_dates = pd.date_range(daily_miles.index.min(), daily_miles.index.max(), freq="D")
    daily_miles = daily_miles.reindex(all_dates, fill_value=0.0)
    daily_runs = daily_runs.reindex(all_dates, fill_value=0)
    daily_pace = daily_pace.reindex(all_dates)  # NaN on rest days

    for w in windows:
        # .shift(1) so each run sees the preceding window, not including today
        roll_miles = daily_miles.rolling(w, min_periods=1).sum().shift(1)
        roll_runs = daily_runs.rolling(w, min_periods=1).sum().shift(1)
        roll_pace = daily_pace.rolling(w, min_periods=1).mean().shift(1)
        df[f"miles_{w}d"] = df["_date"].map(roll_miles).round(2)
        df[f"runs_{w}d"] = df["_date"].map(roll_runs).astype("Int64")
        df[f"avg_pace_{w}d"] = df["_date"].map(roll_pace).round(3)

    # Weekly mileage and week-over-week % change
    df["_yw"] = df["datetime_local"].dt.strftime("%G-%V")  # ISO year-week
    weekly_miles = df.groupby("_yw")["distance_miles"].sum()
    df["weekly_miles"] = df["_yw"].map(weekly_miles).round(2)

    yw_index = weekly_miles.index.tolist()
    prev_week = {yw: yw_index[i - 1] if i > 0 else None for i, yw in enumerate(yw_index)}
    df["weekly_miles_chg"] = df["_yw"].apply(
        lambda yw: (
            (weekly_miles[yw] - weekly_miles[prev_week[yw]]) / weekly_miles[prev_week[yw]] * 100
            if prev_week[yw] and weekly_miles[prev_week[yw]] > 0
            else np.nan
        )
    ).round(1)

    # Days since last run (rest gap — recovery signal)
    run_dates = np.sort(df["_date"].unique())
    date_to_prev = {}
    for i, d in enumerate(run_dates):
        date_to_prev[d] = int((d - run_dates[i - 1]) / np.timedelta64(1, "D")) if i > 0 else np.nan
    df["days_since_last_run"] = df["_date"].map(date_to_prev)

    df = df.drop(columns=["_date", "_yw"])
    return df


# ── Pipeline ───────────────────────────────────────────────────────────────────

def build_features(
    df: pd.DataFrame,
    rest_hr: int = 49,
    max_hr: int = 198,
) -> pd.DataFrame:
    """Apply the full feature engineering pipeline to a cleaned runs DataFrame.

    Drops columns that are 100% null in running activity data (respiration,
    stress) to reduce noise before analysis.

    Args:
        df: Output of clean.filter_runs().
        rest_hr: Resting heart rate (from Garmin HR zones config).
        max_hr: Max heart rate (from Garmin HR zones config).

    Returns:
        Feature-rich DataFrame ready for EDA and modeling.
    """
    # Drop columns that are always null for running activities
    always_null = [
        "minRespirationRate", "maxRespirationRate", "avgRespirationRate",
        "startStress", "endStress", "differenceStress", "avgStress", "maxStress",
    ]
    df = df.drop(columns=[c for c in always_null if c in df.columns])

    df = add_time_features(df)
    df = add_run_metrics(df, rest_hr=rest_hr, max_hr=max_hr)
    df = add_hr_zone_pcts(df)
    df = add_efficiency_features(df)
    df = add_training_load(df, rest_hr=rest_hr, max_hr=max_hr)
    df = add_rolling_features(df)

    return df
