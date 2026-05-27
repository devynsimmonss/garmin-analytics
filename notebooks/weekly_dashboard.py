"""
weekly_dashboard.py — Weekly training dashboard (HTML output).

Run from the project root:
    /opt/anaconda3/bin/python notebooks/weekly_dashboard.py

Opens an interactive HTML dashboard in your default browser.
Requires GARMIN_EMAIL and GARMIN_PASSWORD in .env (see .env.example).
"""

from pathlib import Path
import sys
import webbrowser
import warnings

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.garmin_api import authenticate, update_from_api
from src.features import build_features

# ── Design tokens ──────────────────────────────────────────────────────────────

NAVY   = "#0F172A"
BLUE   = "#2563EB"
LBLUE  = "#DBEAFE"
RED    = "#DC2626"
LRED   = "#FEE2E2"
GREEN  = "#16A34A"
LGREEN = "#DCFCE7"
ORANGE = "#EA580C"
YELLOW = "#CA8A04"
GRAY   = "#6B7280"
LGRAY  = "#F1F5F9"
WHITE  = "#FFFFFF"
CARD   = "#FFFFFF"

ZONE_COLORS      = ["#22C55E", "#84CC16", "#F59E0B", "#EF4444", "#7C3AED"]
ZONE_COLORS_FADE = ["rgba(34,197,94,0.45)", "rgba(132,204,22,0.45)",
                    "rgba(245,158,11,0.45)", "rgba(239,68,68,0.45)",
                    "rgba(124,58,237,0.45)"]
ZONE_LABELS = ["Z1 Recovery", "Z2 Aerobic", "Z3 Tempo", "Z4 Threshold", "Z5 Max"]

REST_HR    = 49
MAX_HR     = 198
CACHE_PATH = ROOT / "data" / "_api_cache.parquet"

QUOTES = [
    ("Pain is inevitable. Suffering is optional.", "Haruki Murakami"),
    ("Run often. Run long. But never outrun your joy of running.", "Julie Isphording"),
    ("It's not about the miles. It's about the smile.", "Unknown"),
    ("Every run is a gift. Not everyone gets to do this.", "Unknown"),
    ("Consistency beats perfection every single time.", "Unknown"),
    ("The body achieves what the mind believes.", "Unknown"),
    ("Champions are made in the moments they want to quit.", "Unknown"),
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def fmt_pace(p: float) -> str:
    if pd.isna(p) or p <= 0:
        return "—"
    m = int(p)
    s = int(round((p - m) * 60))
    return f"{m}:{s:02d}"


def get_quote() -> tuple[str, str]:
    idx = pd.Timestamp.now().day_of_year % len(QUOTES)
    return QUOTES[idx]


def compute_injury_risk(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    risk = pd.Series(0.0, index=df.index)
    if "weekly_miles_chg" in df.columns:
        risk += (df["weekly_miles_chg"].fillna(0).clip(lower=0) / 50 * 30).clip(upper=30)
    if "zone5_pct" in df.columns:
        risk += (df["zone5_pct"].fillna(0) / 100 * 25).clip(upper=25)
    if "days_since_last_run" in df.columns:
        risk += ((2 - df["days_since_last_run"].fillna(2)).clip(lower=0) / 2 * 20).clip(upper=20)
    if "tsb" in df.columns:
        risk += ((-df["tsb"].fillna(0)).clip(lower=0) / 20 * 25).clip(upper=25)
    df["injury_risk_score"] = risk.clip(upper=100).round(1)
    return df


def load_data() -> pd.DataFrame:
    existing = pd.DataFrame()
    if CACHE_PATH.exists():
        existing = pd.read_parquet(CACHE_PATH)
        print(f"  Loaded {len(existing)} cached activities.")
    print("  Connecting to Garmin Connect...")
    client = authenticate()
    print("  Authenticated. Fetching recent activities...")
    raw_df = update_from_api(existing, client, lookback_days=90)
    print(f"  Total activities: {len(raw_df)}")
    if not raw_df.empty:
        CACHE_PATH.parent.mkdir(exist_ok=True)
        raw_df.to_parquet(CACHE_PATH, index=False)
    return raw_df


# ── Plotly chart builders ──────────────────────────────────────────────────────

def chart_training_load(df: pd.DataFrame) -> go.Figure:
    recent = df.sort_values("datetime_local").tail(60).copy()
    dates  = recent["datetime_local"]

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.65, 0.35],
        vertical_spacing=0.04,
    )

    fig.add_trace(go.Scatter(
        x=dates, y=recent["ctl"], name="CTL – Fitness",
        line=dict(color=BLUE, width=2.5),
        fill=None,
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=dates, y=recent["atl"], name="ATL – Fatigue",
        line=dict(color=RED, width=2, dash="dot"),
        fill="tonexty", fillcolor="rgba(220,38,38,0.08)",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=dates, y=recent["tsb"], name="TSB – Form",
        line=dict(color=GREEN, width=2),
        fill="tozeroy",
        fillcolor=recent["tsb"].apply(
            lambda v: "rgba(22,163,74,0.15)" if v >= 0 else "rgba(220,38,38,0.10)"
        ).iloc[0],
    ), row=2, col=1)

    fig.add_hline(y=0,   line=dict(color=GRAY, width=1, dash="dash"), row=2, col=1)
    fig.add_hline(y=-10, line=dict(color=YELLOW, width=1, dash="dot"),
                  annotation_text="Caution", annotation_font_color=YELLOW, row=2, col=1)
    fig.add_hline(y=-20, line=dict(color=RED, width=1, dash="dot"),
                  annotation_text="Overreach", annotation_font_color=RED, row=2, col=1)

    fig.update_layout(
        title=dict(text="Training Load · Last 60 Days", font=dict(size=14, color=NAVY), x=0),
        height=320, margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor=LGRAY, paper_bgcolor=WHITE,
        legend=dict(orientation="h", y=1.12, x=0, font=dict(size=11)),
        font=dict(family="Inter, sans-serif"),
    )
    fig.update_yaxes(title_text="TRIMP", row=1, col=1, title_font=dict(size=10))
    fig.update_yaxes(title_text="TSB", row=2, col=1, title_font=dict(size=10))
    return fig


def chart_weekly_mileage(df: pd.DataFrame) -> go.Figure:
    weekly = (
        df.groupby(df["datetime_local"].dt.to_period("W"))["distance_miles"]
        .sum()
        .reset_index()
    )
    weekly["week_start"] = weekly["datetime_local"].dt.start_time
    weekly = weekly.tail(16)
    pct_chg = weekly["distance_miles"].pct_change() * 100
    bar_colors = [RED if abs(c) > 10 else BLUE for c in pct_chg.fillna(0)]

    ma = weekly["distance_miles"].rolling(4, min_periods=1).mean()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=weekly["week_start"], y=weekly["distance_miles"],
        marker_color=bar_colors, marker_opacity=0.82,
        name="Weekly miles",
        hovertemplate="%{x|%b %d}<br>%{y:.1f} miles<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=weekly["week_start"], y=ma,
        name="4-week avg", line=dict(color=ORANGE, width=2.5),
        hovertemplate="%{y:.1f} mi avg<extra></extra>",
    ))

    fig.update_layout(
        title=dict(text="Weekly Mileage · Last 16 Weeks", font=dict(size=14, color=NAVY), x=0),
        height=320, margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor=LGRAY, paper_bgcolor=WHITE,
        legend=dict(orientation="h", y=1.12, x=0, font=dict(size=11)),
        font=dict(family="Inter, sans-serif"),
        yaxis_title="Miles",
        bargap=0.25,
    )
    return fig


def chart_zone_distribution(week_df: pd.DataFrame, df: pd.DataFrame) -> go.Figure:
    zone_cols = [f"zone{i}_min" for i in range(1, 6)]

    def zone_pcts(frame):
        if frame.empty:
            return [0.0] * 5
        totals = [frame[c].sum() if c in frame.columns else 0.0 for c in zone_cols]
        s = sum(totals) or 1
        return [t / s * 100 for t in totals]

    cutoff      = df["datetime_local"].max() - pd.Timedelta(weeks=8)
    baseline_df = df[df["datetime_local"] >= cutoff]
    week_pcts     = zone_pcts(week_df)
    baseline_pcts = zone_pcts(baseline_df)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="8-week avg", x=ZONE_LABELS, y=baseline_pcts,
        marker_color=ZONE_COLORS_FADE,
        hovertemplate="%{x}: %{y:.1f}%<extra>8-wk avg</extra>",
    ))
    fig.add_trace(go.Bar(
        name="This week", x=ZONE_LABELS, y=week_pcts,
        marker_color=ZONE_COLORS,
        hovertemplate="%{x}: %{y:.1f}%<extra>This week</extra>",
    ))
    fig.add_hline(y=80, line=dict(color=BLUE, width=1, dash="dash"),
                  annotation_text="80% Z1/Z2 target", annotation_font_color=BLUE,
                  annotation_font_size=10)

    fig.update_layout(
        title=dict(text="HR Zone Distribution", font=dict(size=14, color=NAVY), x=0),
        height=320, margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor=LGRAY, paper_bgcolor=WHITE,
        barmode="group", bargap=0.2,
        legend=dict(orientation="h", y=1.12, x=0, font=dict(size=11)),
        font=dict(family="Inter, sans-serif"),
        yaxis=dict(title="% of zone time", range=[0, 105]),
    )
    return fig


def chart_recent_runs(df: pd.DataFrame) -> go.Figure:
    recent = df.sort_values("datetime_local").tail(8).copy()
    recent = recent.sort_values("datetime_local", ascending=False)

    dates  = recent["datetime_local"].dt.strftime("%b %-d")
    miles  = recent["distance_miles"].round(1).astype(str) + " mi"
    paces  = recent["avg_pace_min_per_mile"].apply(fmt_pace) + " /mi"
    hrs    = recent["avgHr"].fillna(0).astype(int).astype(str) + " bpm"
    zones  = recent["effort_zone"].astype(str) if "effort_zone" in recent.columns else ["—"] * len(recent)

    zone_color_map = {
        "Easy":      "rgba(22,163,74,0.18)",
        "Moderate":  "rgba(132,204,22,0.18)",
        "Hard":      "rgba(234,88,12,0.18)",
        "Very Hard": "rgba(220,38,38,0.18)",
        "Maximum":   "rgba(124,58,237,0.18)",
    }
    row_colors = [[zone_color_map.get(str(z), "rgba(248,250,252,1)") for z in zones]] * 5

    fig = go.Figure(go.Table(
        header=dict(
            values=["<b>Date</b>", "<b>Distance</b>", "<b>Pace</b>",
                    "<b>Avg HR</b>", "<b>Effort Zone</b>"],
            fill_color=NAVY, font=dict(color=WHITE, size=12, family="Inter, sans-serif"),
            align="left", height=32,
        ),
        cells=dict(
            values=[dates, miles, paces, hrs, zones],
            fill_color=row_colors,
            font=dict(color=NAVY, size=11, family="Inter, sans-serif"),
            align="left", height=28,
            line_color="#E2E8F0",
        ),
    ))
    fig.update_layout(
        title=dict(text="Recent Runs", font=dict(size=14, color=NAVY), x=0),
        height=320, margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor=WHITE, font=dict(family="Inter, sans-serif"),
    )
    return fig


# ── Written recommendation ─────────────────────────────────────────────────────

def build_coach_note(df: pd.DataFrame, week_df: pd.DataFrame) -> str:
    latest     = df.sort_values("datetime_local").iloc[-1]
    risk_score = float(latest.get("injury_risk_score", 0) or 0)
    ctl        = float(latest.get("ctl", 0) or 0)
    tsb        = float(latest.get("tsb", 0) or 0)
    miles_chg  = float(latest.get("weekly_miles_chg", 0) or 0)
    week_miles = float(week_df["distance_miles"].sum()) if not week_df.empty else 0.0
    avg_pace   = week_df["avg_pace_min_per_mile"].mean() if not week_df.empty else np.nan
    last_run   = df.sort_values("datetime_local")["datetime_local"].iloc[-1]
    days_off   = (pd.Timestamp.now() - last_run).days

    parts = []

    if ctl >= 30:
        parts.append(f"Your fitness (CTL {ctl:.1f}) is in solid shape — you've built a meaningful aerobic base.")
    elif ctl >= 15:
        parts.append(f"Your fitness base (CTL {ctl:.1f}) is developing — consistent easy mileage will continue to lift it.")
    else:
        parts.append(f"Your CTL ({ctl:.1f}) is still building. Patience and consistency are the key right now.")

    if tsb < -15:
        parts.append(f"Fatigue is running high (TSB {tsb:+.1f}). A recovery week with 40% less volume is overdue.")
    elif tsb < -5:
        parts.append(f"You're carrying moderate fatigue (TSB {tsb:+.1f}). An extra rest day this week will help.")
    elif tsb >= 5:
        parts.append(f"Your form is positive (TSB {tsb:+.1f}) — you're fresh and recovered.")
    else:
        parts.append(f"Your training stress balance (TSB {tsb:+.1f}) is near neutral.")

    if days_off >= 7:
        parts.append(f"You've had {days_off} days off since your last run. When you return, start with a short 2-mile test at full Zone 1 effort and monitor your knee for 24 hours.")
    elif week_miles == 0:
        parts.append("You haven't run yet this week — listen to your body before lacing up.")
    else:
        pace_str = f" at {fmt_pace(avg_pace)} min/mi" if not pd.isna(avg_pace) else ""
        parts.append(f"This week you've covered {week_miles:.1f} miles{pace_str}.")

    if miles_chg > 10:
        parts.append(f"Your mileage jumped {miles_chg:+.0f}% last week — that kind of spike is the most common cause of overuse injuries like runner's knee. Hold mileage flat or reduce it this week.")

    if risk_score >= 60:
        parts.append("Given the current risk level, stick to non-impact cross-training only — pool running, cycling, or elliptical — until the knee is fully pain-free on stairs and easy walks.")
    elif risk_score >= 35:
        parts.append("Keep all runs at a fully conversational Zone 1–2 effort. No tempo, no intervals until you've had two symptom-free weeks.")
    else:
        parts.append("You're in a safe zone to train. Aim for 80% of your time in Zone 1–2 and limit week-over-week mileage increases to 10% or less.")

    parts.append("For runner's knee specifically: prioritize hip abductor and glute strengthening (clamshells, lateral band walks, single-leg deadlifts) 3× per week — weak hips are the root cause in most cases.")

    return "  ".join(parts)


# ── HTML builder ───────────────────────────────────────────────────────────────

def build_html(df: pd.DataFrame, week_df: pd.DataFrame, week_start: pd.Timestamp) -> str:
    latest     = df.sort_values("datetime_local").iloc[-1]
    risk_score = float(latest.get("injury_risk_score", 0) or 0)
    ctl        = float(latest.get("ctl", 0) or 0)
    tsb        = float(latest.get("tsb", 0) or 0)
    week_miles = week_df["distance_miles"].sum() if not week_df.empty else 0.0
    week_runs  = len(week_df)
    avg_pace   = week_df["avg_pace_min_per_mile"].mean() if not week_df.empty else np.nan

    tsb_color = GREEN if tsb >= -5 else (YELLOW if tsb >= -15 else RED)
    tsb_bg    = LGREEN if tsb >= -5 else ("#FEF9C3" if tsb >= -15 else LRED)

    if risk_score >= 60:
        risk_bg, risk_fg, risk_label = LRED,    RED,    "HIGH RISK"
    elif risk_score >= 35:
        risk_bg, risk_fg, risk_label = "#FEF9C3", YELLOW, "MODERATE RISK"
    else:
        risk_bg, risk_fg, risk_label = LGREEN, GREEN, "LOW RISK"

    quote_text, quote_author = get_quote()
    today_str   = pd.Timestamp.now().strftime("%A, %B %-d %Y")
    week_str    = week_start.strftime("%b %-d, %Y")
    coach_note  = build_coach_note(df, week_df)

    # ── Charts ─────────────────────────────────────────────────────────────────
    cfg = dict(responsive=True, displayModeBar=False)
    def to_div(fig, first=False):
        return fig.to_html(
            full_html=False,
            include_plotlyjs="cdn" if first else False,
            config=cfg,
            div_id=None,
        )

    c_load   = to_div(chart_training_load(df), first=True)
    c_miles  = to_div(chart_weekly_mileage(df))
    c_zones  = to_div(chart_zone_distribution(week_df, df))
    c_recent = to_div(chart_recent_runs(df))

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Devyn's Training Dashboard</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: 'Inter', sans-serif;
    background: #F8FAFC;
    color: {NAVY};
    min-height: 100vh;
  }}

  /* ── Header ── */
  .header {{
    background: {NAVY};
    padding: 28px 40px 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 24px;
  }}
  .header-left h1 {{
    font-size: 28px;
    font-weight: 800;
    color: {WHITE};
    letter-spacing: -0.5px;
  }}
  .header-left .date {{
    font-size: 13px;
    color: #94A3B8;
    margin-top: 4px;
  }}
  .header-quote {{
    flex: 1;
    text-align: center;
    max-width: 500px;
  }}
  .header-quote .q-text {{
    font-size: 13.5px;
    color: #CBD5E1;
    font-style: italic;
    line-height: 1.5;
  }}
  .header-quote .q-author {{
    font-size: 11.5px;
    color: #64748B;
    margin-top: 5px;
  }}

  /* ── Main layout ── */
  .main {{ padding: 28px 36px; max-width: 1300px; margin: 0 auto; }}

  /* ── Week label ── */
  .week-label {{
    font-size: 13px;
    font-weight: 600;
    color: {GRAY};
    letter-spacing: 0.8px;
    text-transform: uppercase;
    margin-bottom: 14px;
  }}

  /* ── Stat cards ── */
  .cards {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-bottom: 24px;
  }}
  .card {{
    background: {CARD};
    border-radius: 14px;
    padding: 20px 22px 16px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.07);
    border: 1px solid #E2E8F0;
  }}
  .card .c-label {{
    font-size: 11px;
    font-weight: 600;
    color: {GRAY};
    letter-spacing: 0.6px;
    text-transform: uppercase;
    margin-bottom: 8px;
  }}
  .card .c-value {{
    font-size: 30px;
    font-weight: 800;
    letter-spacing: -1px;
    line-height: 1;
  }}
  .card .c-sub {{
    font-size: 11.5px;
    color: {GRAY};
    margin-top: 6px;
    font-style: italic;
  }}

  /* ── Charts grid ── */
  .charts-row {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
    margin-bottom: 20px;
  }}
  .chart-card {{
    background: {CARD};
    border-radius: 14px;
    padding: 18px 18px 10px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.07);
    border: 1px solid #E2E8F0;
    overflow: hidden;
  }}

  /* ── Coach's note ── */
  .rec-card {{
    background: {CARD};
    border-radius: 14px;
    padding: 24px 28px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.07);
    border: 1px solid #E2E8F0;
    margin-bottom: 32px;
  }}
  .rec-card h2 {{
    font-size: 15px;
    font-weight: 700;
    color: {NAVY};
    margin-bottom: 14px;
  }}
  .risk-badge {{
    display: inline-block;
    background: {risk_bg};
    color: {risk_fg};
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.5px;
    padding: 5px 14px;
    border-radius: 999px;
    margin-bottom: 14px;
    border: 1.5px solid {risk_fg}44;
  }}
  .coach-text {{
    font-size: 14px;
    line-height: 1.75;
    color: #1E293B;
    max-width: 900px;
  }}
  .coach-text span {{
    display: block;
    margin-bottom: 8px;
  }}
</style>
</head>
<body>

<div class="header">
  <div class="header-left">
    <h1>Hello, Devyn! 👋</h1>
    <div class="date">{today_str}</div>
  </div>
  <div class="header-quote">
    <div class="q-text">"{quote_text}"</div>
    <div class="q-author">— {quote_author}</div>
  </div>
</div>

<div class="main">
  <div class="week-label">Week of {week_str}</div>

  <!-- Stat cards -->
  <div class="cards">
    <div class="card">
      <div class="c-label">Miles This Week</div>
      <div class="c-value" style="color:{BLUE}">{week_miles:.1f}</div>
      <div class="c-sub">{week_runs} run{'s' if week_runs != 1 else ''}</div>
    </div>
    <div class="card">
      <div class="c-label">Avg Pace</div>
      <div class="c-value" style="color:{NAVY}">{fmt_pace(avg_pace)}</div>
      <div class="c-sub">min / mile this week</div>
    </div>
    <div class="card">
      <div class="c-label">CTL · Fitness</div>
      <div class="c-value" style="color:{GREEN}">{ctl:.1f}</div>
      <div class="c-sub">chronic training load</div>
    </div>
    <div class="card" style="background:{tsb_bg}">
      <div class="c-label">TSB · Form</div>
      <div class="c-value" style="color:{tsb_color}">{tsb:+.1f}</div>
      <div class="c-sub" style="color:{tsb_color}">{'fresh ↑' if tsb >= -5 else ('fatigued ↓' if tsb < -15 else 'moderate')}</div>
    </div>
  </div>

  <!-- Charts row 1 -->
  <div class="charts-row">
    <div class="chart-card">{c_load}</div>
    <div class="chart-card">{c_miles}</div>
  </div>

  <!-- Charts row 2 -->
  <div class="charts-row">
    <div class="chart-card">{c_zones}</div>
    <div class="chart-card">{c_recent}</div>
  </div>

  <!-- Coach's note -->
  <div class="rec-card">
    <h2>Coach's Note</h2>
    <div class="risk-badge">Injury Risk Score: {risk_score:.0f} / 100 &nbsp;·&nbsp; {risk_label}</div>
    <div class="coach-text">
      {''.join(f'<span>{s.strip()}</span>' for s in coach_note.split('  ') if s.strip())}
    </div>
  </div>
</div>

</body>
</html>"""


# ── Entry point ────────────────────────────────────────────────────────────────

def build_dashboard():
    print("\nLoading data...")
    raw_df = load_data()

    if raw_df.empty:
        print("No activities found.")
        return

    runs_df = raw_df[raw_df["activityType"].isin({"running", "treadmill_running"})].copy()
    if runs_df.empty:
        print("No running activities found.")
        return

    print("  Building features...")
    df      = build_features(runs_df, rest_hr=REST_HR, max_hr=MAX_HR)
    df      = compute_injury_risk(df)

    today    = pd.Timestamp.now().normalize()
    wk_start = today - pd.Timedelta(days=today.dayofweek)
    week_df  = df[df["datetime_local"] >= wk_start].copy()

    print("  Generating HTML dashboard...")
    html = build_html(df, week_df, wk_start)

    out_path = ROOT / "outputs" / "weekly_dashboard.html"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"  Saved → outputs/weekly_dashboard.html")

    webbrowser.open(out_path.as_uri())
    print("  Opened in browser.")


if __name__ == "__main__":
    build_dashboard()
