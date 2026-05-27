"""
01_eda.py — Exploratory Data Analysis: Garmin running data (Dec 2024 – Apr 2026)

Run this script directly to generate and save all plots:
    python notebooks/01_eda.py

Plots are saved to outputs/.
"""

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

# ── Load data ──────────────────────────────────────────────────────────────────

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.clean import clean_activities, filter_runs
from src.features import build_features

DATA = Path(__file__).resolve().parents[1] / "data"
OUT  = Path(__file__).resolve().parents[1] / "outputs"
OUT.mkdir(exist_ok=True)

raw  = clean_activities(DATA / "garmin_activities.json")
runs = filter_runs(raw)
df   = build_features(runs)

# Drop the 4 outlier walks (pace > 15 min/mile) from visual plots
df_runs = df[df["avg_pace_min_per_mile"] < 15].copy()

# ── Style ──────────────────────────────────────────────────────────────────────

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
BLUE   = "#2563EB"
RED    = "#DC2626"
GREEN  = "#16A34A"
ORANGE = "#EA580C"
GRAY   = "#6B7280"

ZONE_COLORS = {
    "Easy":      "#22C55E",
    "Moderate":  "#84CC16",
    "Hard":      "#F59E0B",
    "Very Hard": "#EF4444",
    "Maximum":   "#7C3AED",
}

def save(name: str):
    plt.tight_layout()
    plt.savefig(OUT / name, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved → outputs/{name}")


# ══════════════════════════════════════════════════════════════════════════════
# Plot 1 — Weekly Mileage Over Time
# ══════════════════════════════════════════════════════════════════════════════
print("Plot 1: Weekly mileage")

weekly = (
    df.groupby(df["datetime_local"].dt.to_period("W"))["distance_miles"]
    .sum()
    .reset_index()
)
weekly["week_start"] = weekly["datetime_local"].dt.start_time
weekly_ma = weekly["distance_miles"].rolling(4, min_periods=1).mean()

fig, ax = plt.subplots(figsize=(14, 4))
ax.bar(weekly["week_start"], weekly["distance_miles"],
       width=5, color=BLUE, alpha=0.55, label="Weekly miles")
ax.plot(weekly["week_start"], weekly_ma,
        color=RED, lw=2, label="4-week avg")

ax.set_title("Weekly Running Mileage  (Dec 2024 – Apr 2026)", fontsize=14, fontweight="bold")
ax.set_xlabel("")
ax.set_ylabel("Miles")
ax.yaxis.set_major_locator(mticker.MultipleLocator(5))
ax.legend(frameon=False)
ax.annotate(f"Peak: {weekly['distance_miles'].max():.1f} mi",
            xy=(weekly.loc[weekly['distance_miles'].idxmax(), 'week_start'],
                weekly['distance_miles'].max()),
            xytext=(10, 8), textcoords="offset points",
            fontsize=9, color=RED)

save("01_weekly_mileage.png")


# ══════════════════════════════════════════════════════════════════════════════
# Plot 2 — Pace Distribution
# ══════════════════════════════════════════════════════════════════════════════
print("Plot 2: Pace distribution")

fig, ax = plt.subplots(figsize=(9, 5))

sns.histplot(df_runs["avg_pace_min_per_mile"], bins=25, kde=True,
             color=BLUE, alpha=0.55, ax=ax, line_kws={"lw": 2})

for pct, label, ls in [
    (df_runs["avg_pace_min_per_mile"].quantile(0.25), "Q1", "--"),
    (df_runs["avg_pace_min_per_mile"].median(),        "Median", "-"),
    (df_runs["avg_pace_min_per_mile"].quantile(0.75), "Q3", "--"),
]:
    ax.axvline(pct, color=RED, ls=ls, lw=1.5,
               label=f"{label}: {pct:.2f} min/mi")

ax.set_title("Run Pace Distribution", fontsize=14, fontweight="bold")
ax.set_xlabel("Avg Pace (min/mile)")
ax.set_ylabel("Count")

# Format x-axis as M:SS
def pace_fmt(x, _):
    mins = int(x)
    secs = int(round((x - mins) * 60))
    return f"{mins}:{secs:02d}"

ax.xaxis.set_major_formatter(mticker.FuncFormatter(pace_fmt))
ax.legend(frameon=False, fontsize=9)

save("02_pace_distribution.png")


# ══════════════════════════════════════════════════════════════════════════════
# Plot 3 — Pace vs. Heart Rate (coloured by effort zone)
# ══════════════════════════════════════════════════════════════════════════════
print("Plot 3: Pace vs HR")

fig, ax = plt.subplots(figsize=(9, 6))

for zone, grp in df_runs.groupby("effort_zone", observed=True):
    ax.scatter(grp["avgHr"], grp["avg_pace_min_per_mile"],
               color=ZONE_COLORS.get(str(zone), GRAY),
               label=str(zone), s=60, alpha=0.8, edgecolors="white", lw=0.4)

# Regression line
valid = df_runs[["avgHr", "avg_pace_min_per_mile"]].dropna()
m, b = np.polyfit(valid["avgHr"], valid["avg_pace_min_per_mile"], 1)
x_line = np.linspace(valid["avgHr"].min(), valid["avgHr"].max(), 100)
ax.plot(x_line, m * x_line + b, color="black", lw=1.2, ls="--", alpha=0.5, label="Trend")

ax.invert_yaxis()   # lower pace value = faster
ax.yaxis.set_major_formatter(mticker.FuncFormatter(pace_fmt))
ax.set_title("Avg Pace vs. Heart Rate by Effort Zone", fontsize=14, fontweight="bold")
ax.set_xlabel("Avg Heart Rate (bpm)")
ax.set_ylabel("Avg Pace (min/mile)  ← faster")
ax.legend(frameon=False, fontsize=9, loc="upper left")

save("03_pace_vs_hr.png")


# ══════════════════════════════════════════════════════════════════════════════
# Plot 4 — VO2Max and Efficiency Factor Over Time
# ══════════════════════════════════════════════════════════════════════════════
print("Plot 4: VO2Max + efficiency factor")

vo2_df = df[df["vO2MaxValue"].notna()].copy()
monthly_vo2 = (
    vo2_df.set_index("datetime_local")["vO2MaxValue"]
    .resample("ME").mean()
    .dropna()
)
monthly_ef = (
    df.set_index("datetime_local")["efficiency_factor"]
    .resample("ME").mean()
    .dropna()
)

fig, ax1 = plt.subplots(figsize=(13, 4))
ax2 = ax1.twinx()

ax1.plot(monthly_vo2.index, monthly_vo2.values,
         color=BLUE, lw=2.5, marker="o", ms=5, label="VO₂Max")
ax1.fill_between(monthly_vo2.index, monthly_vo2.values,
                 monthly_vo2.min() - 1, alpha=0.15, color=BLUE)

ax2.plot(monthly_ef.index, monthly_ef.values,
         color=ORANGE, lw=2, ls="--", marker="s", ms=4, label="Efficiency Factor")

# Peak annotation
peak_date = monthly_vo2.idxmax()
ax1.annotate(f"Peak VO₂Max\n{monthly_vo2.max():.1f}",
             xy=(peak_date, monthly_vo2.max()),
             xytext=(-40, 12), textcoords="offset points",
             fontsize=8.5, color=BLUE,
             arrowprops=dict(arrowstyle="->", color=BLUE, lw=0.8))

ax1.set_title("VO₂Max and Aerobic Efficiency Over Time", fontsize=14, fontweight="bold")
ax1.set_ylabel("VO₂Max (ml/kg/min)", color=BLUE)
ax1.tick_params(axis="y", colors=BLUE)
ax2.set_ylabel("Efficiency Factor (×10⁻³ mph/bpm)", color=ORANGE)
ax2.tick_params(axis="y", colors=ORANGE)
ax1.set_xlabel("")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, frameon=False, fontsize=9, loc="lower left")

save("04_vo2max_efficiency.png")


# ══════════════════════════════════════════════════════════════════════════════
# Plot 5 — Total Training Time by HR Zone
# ══════════════════════════════════════════════════════════════════════════════
print("Plot 5: HR zone totals")

zone_hours = {
    f"Zone {i}": df[f"zone{i}_min"].sum() / 60
    for i in range(1, 6)
}
zone_labels = list(zone_hours.keys())
zone_vals   = list(zone_hours.values())
zone_pals   = ["#22C55E", "#84CC16", "#F59E0B", "#EF4444", "#7C3AED"]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(zone_labels, zone_vals, color=zone_pals, width=0.55, edgecolor="white")

# Annotate each bar with hours + pct
total_hours = sum(zone_vals)
for bar, val in zip(bars, zone_vals):
    pct = val / total_hours * 100
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{val:.1f} h\n({pct:.0f}%)",
            ha="center", va="bottom", fontsize=9)

ax.set_title("Total Training Time by HR Zone", fontsize=14, fontweight="bold")
ax.set_ylabel("Hours")
ax.set_ylim(0, max(zone_vals) * 1.25)
ax.text(0.98, 0.97,
        f"Total: {total_hours:.0f} hours",
        transform=ax.transAxes, ha="right", va="top", fontsize=9, color=GRAY)

save("05_hr_zone_totals.png")


# ══════════════════════════════════════════════════════════════════════════════
# Plot 6 — ATL / CTL / TSB (Training Load Over Time)
# ══════════════════════════════════════════════════════════════════════════════
print("Plot 6: ATL / CTL / TSB")

load = df.sort_values("datetime_local")
dates = load["datetime_local"]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True,
                                gridspec_kw={"height_ratios": [2, 1]})

ax1.plot(dates, load["ctl"], color=BLUE,  lw=2.5, label="CTL – Chronic Load (Fitness)")
ax1.plot(dates, load["atl"], color=RED,   lw=2,   label="ATL – Acute Load (Fatigue)", alpha=0.8)
ax1.fill_between(dates, load["ctl"], load["atl"],
                 where=load["ctl"] >= load["atl"],
                 alpha=0.15, color=GREEN, label="Positive form (CTL > ATL)")
ax1.fill_between(dates, load["ctl"], load["atl"],
                 where=load["ctl"] < load["atl"],
                 alpha=0.15, color=RED)
ax1.set_ylabel("Training Load (TRIMP)")
ax1.legend(frameon=False, fontsize=9)
ax1.set_title("Training Load: Fitness, Fatigue & Form  (Banister Model)", fontsize=14, fontweight="bold")

ax2.plot(dates, load["tsb"], color=GREEN, lw=2, label="TSB – Form (CTL − ATL)")
ax2.axhline(0, color="black", lw=0.8, ls="--")
ax2.fill_between(dates, load["tsb"], 0,
                 where=load["tsb"] >= 0, color=GREEN, alpha=0.25, label="Fresh / recovered")
ax2.fill_between(dates, load["tsb"], 0,
                 where=load["tsb"] < 0, color=RED,   alpha=0.20, label="Fatigued")
ax2.set_ylabel("TSB")
ax2.legend(frameon=False, fontsize=9)

save("06_training_load.png")


# ══════════════════════════════════════════════════════════════════════════════
# Plot 7 — Runs by Day of Week
# ══════════════════════════════════════════════════════════════════════════════
print("Plot 7: Day-of-week patterns")

dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
dow = (df.groupby("day_name")
         .agg(count=("activityId", "count"), avg_miles=("distance_miles", "mean"))
         .reindex(dow_order))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

bars = ax1.bar(dow.index, dow["count"], color=BLUE, alpha=0.75, edgecolor="white")
ax1.set_title("Run Frequency by Day", fontsize=13, fontweight="bold")
ax1.set_ylabel("Number of Runs")
ax1.set_xticklabels(dow.index, rotation=30, ha="right")
for bar, val in zip(bars, dow["count"]):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             str(int(val)), ha="center", fontsize=9)

bars2 = ax2.bar(dow.index, dow["avg_miles"], color=ORANGE, alpha=0.75, edgecolor="white")
ax2.set_title("Avg Distance by Day", fontsize=13, fontweight="bold")
ax2.set_ylabel("Avg Miles")
ax2.set_xticklabels(dow.index, rotation=30, ha="right")
for bar, val in zip(bars2, dow["avg_miles"]):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
             f"{val:.1f}", ha="center", fontsize=9)

save("07_day_of_week.png")


print("\nAll plots saved to outputs/")
