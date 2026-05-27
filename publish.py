"""
publish.py — Build the GitHub Pages portfolio site.

Run this before pushing to GitHub:
    python publish.py

What it does:
  1. Loads the run data and computes key stats
  2. Copies outputs/ charts → docs/assets/
  3. Copies the weekly dashboard HTML → docs/dashboard.html
  4. Generates docs/index.html with real stats filled in

Then commit and push:
    git add docs/
    git commit -m "Update portfolio site"
    git push
"""

from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import pandas as pd

# ── Config ─────────────────────────────────────────────────────────────────────

GITHUB_USERNAME = "devynsimmonss"
REPO_NAME       = "garmin-analytics"       # ← replace if your repo name differs

DOCS      = ROOT / "docs"
ASSETS    = DOCS / "assets"
OUTPUTS   = ROOT / "outputs"
CACHE     = ROOT / "data" / "_api_cache.parquet"
DATA_JSON = ROOT / "data" / "garmin_activities.json"


# ── Load data ──────────────────────────────────────────────────────────────────

def load_stats() -> dict:
    """Compute portfolio headline stats from cached or raw data."""
    try:
        from src.clean import clean_activities, filter_runs
        from src.features import build_features

        if DATA_JSON.exists():
            raw  = clean_activities(DATA_JSON)
            runs = filter_runs(raw)
        elif CACHE.exists():
            raw  = pd.read_parquet(CACHE)
            runs = raw[raw["activityType"].isin({"running", "treadmill_running"})].copy()
        else:
            print("  No data found — using placeholder stats.")
            return _placeholder_stats()

        df = build_features(runs)

        date_min  = df["datetime_local"].min()
        date_max  = df["datetime_local"].max()
        months    = round((date_max - date_min).days / 30.4)

        def fmt_pace(p):
            if pd.isna(p) or p <= 0:
                return "—"
            m = int(p); s = int(round((p - m) * 60))
            return f"{m}:{s:02d}"

        return {
            "total_runs":    len(df),
            "total_miles":   f"{df['distance_miles'].sum():.0f}",
            "months":        months,
            "date_range":    f"{date_min.strftime('%b %Y')} – {date_max.strftime('%b %Y')}",
            "avg_weekly":    f"{df['weekly_miles'].mean():.1f}" if "weekly_miles" in df.columns else "—",
            "peak_vo2":      f"{df['vO2MaxValue'].max():.0f}" if "vO2MaxValue" in df.columns else "—",
            "median_pace":   fmt_pace(df["avg_pace_min_per_mile"].median()),
            "peak_ctl":      f"{df['ctl'].max():.1f}" if "ctl" in df.columns else "—",
        }
    except Exception as e:
        print(f"  Warning: could not compute stats ({e}). Using placeholders.")
        return _placeholder_stats()


def _placeholder_stats() -> dict:
    return {
        "total_runs": "70+", "total_miles": "500+", "months": 16,
        "date_range": "Dec 2024 – Apr 2026", "avg_weekly": "—",
        "peak_vo2": "48", "median_pace": "9:30", "peak_ctl": "—",
    }


# ── Asset copying ──────────────────────────────────────────────────────────────

def copy_assets():
    ASSETS.mkdir(parents=True, exist_ok=True)
    copied = 0
    for img in sorted(OUTPUTS.glob("*.png")):
        shutil.copy2(img, ASSETS / img.name)
        copied += 1
    dashboard_src = OUTPUTS / "weekly_dashboard.html"
    if dashboard_src.exists():
        shutil.copy2(dashboard_src, DOCS / "dashboard.html")
        print(f"  Copied dashboard.html")
    print(f"  Copied {copied} chart images → docs/assets/")


# ── HTML generation ────────────────────────────────────────────────────────────

CHART_INFO = [
    ("01_weekly_mileage.png",   "Weekly Mileage",          "Total miles run per week with 4-week rolling average. Red bars flag >10% week-over-week spikes — the leading predictor of overuse injury."),
    ("02_pace_distribution.png","Pace Distribution",        "Histogram of average pace across all runs. Q1/median/Q3 marked. The spread reflects training variety from recovery jogs to race-effort runs."),
    ("03_pace_vs_hr.png",       "Pace vs. Heart Rate",     "Each dot is one run, colored by effort zone. The trend line shows the expected pace–HR tradeoff; outliers reveal unusually good or bad days."),
    ("04_vo2max_efficiency.png","VO₂Max & Efficiency",     "Monthly VO₂Max estimate alongside aerobic efficiency factor (speed/HR). Both track fitness improvement independently of pace."),
    ("05_hr_zone_totals.png",   "HR Zone Distribution",    "Total training time by heart rate zone. Zone 1–2 dominance reflects a polarized base-building approach (~80/20 training)."),
    ("06_training_load.png",    "ATL / CTL / TSB",         "The Banister model: Chronic Training Load (fitness), Acute Training Load (fatigue), and Training Stress Balance (form). Negative TSB at injury onset confirms the overreaching diagnosis."),
    ("07_day_of_week.png",      "Day-of-Week Patterns",    "Run frequency and average distance by day of week. Saturday long runs and midweek recovery runs emerge as the dominant training structure."),
]

STAT_ANALYSIS_CHARTS = [
    ("08_correlation_matrix.png",  "Spearman Correlation Matrix",  "Non-parametric correlations between all key running metrics."),
    ("09_pace_trend.png",          "Pace Trend Over Time",          "OLS regression with 95% CI. Pace slowed +0.024 min/mi per week — driven by volume, not fitness decline."),
    ("10_recovery_vs_pace.png",    "Recovery Gap vs. Pace",         "Mann-Whitney U: runs within 1 day of a prior run are significantly slower (p = 0.003)."),
    ("11_season_pace.png",         "Pace by Season",                "Kruskal-Wallis: seasonal pace differences are statistically significant (p < 0.05)."),
]


def generate_html(stats: dict) -> str:
    repo_url   = f"https://github.com/{GITHUB_USERNAME}/{REPO_NAME}"
    dash_url   = "dashboard.html"
    today_year = pd.Timestamp.now().year

    chart_cards = "\n".join(
        f"""
        <div class="chart-card">
          <img src="assets/{fn}" alt="{title}" loading="lazy">
          <div class="chart-info">
            <h3>{title}</h3>
            <p>{desc}</p>
          </div>
        </div>"""
        for fn, title, desc in CHART_INFO
        if (ASSETS / fn).exists()
    )

    stat_cards = "\n".join(
        f"""
        <div class="stat-chart-card">
          <img src="assets/{fn}" alt="{title}" loading="lazy">
          <div class="chart-info">
            <h3>{title}</h3>
            <p>{desc}</p>
          </div>
        </div>"""
        for fn, title, desc in STAT_ANALYSIS_CHARTS
        if (ASSETS / fn).exists()
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Garmin Running Analytics · Devyn Simmons</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  :root {{
    --navy:  #0F172A;
    --blue:  #2563EB;
    --green: #16A34A;
    --gray:  #6B7280;
    --lgray: #F1F5F9;
    --white: #FFFFFF;
    --border:#E2E8F0;
  }}
  body {{ font-family: 'Inter', sans-serif; background: #F8FAFC; color: var(--navy); }}
  a {{ color: var(--blue); text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}

  /* ── Hero ── */
  .hero {{
    background: var(--navy);
    padding: 56px 40px 48px;
    text-align: center;
  }}
  .hero-tag {{
    display: inline-block;
    background: rgba(37,99,235,0.25);
    color: #93C5FD;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    padding: 5px 14px;
    border-radius: 999px;
    margin-bottom: 20px;
  }}
  .hero h1 {{
    font-size: clamp(28px, 5vw, 46px);
    font-weight: 800;
    color: var(--white);
    letter-spacing: -1px;
    line-height: 1.15;
    margin-bottom: 16px;
  }}
  .hero p {{
    font-size: 16px;
    color: #94A3B8;
    max-width: 580px;
    margin: 0 auto 32px;
    line-height: 1.7;
  }}
  .hero-btns {{ display: flex; gap: 14px; justify-content: center; flex-wrap: wrap; }}
  .btn {{
    display: inline-block;
    padding: 11px 24px;
    border-radius: 8px;
    font-size: 14px;
    font-weight: 600;
    transition: opacity 0.15s;
  }}
  .btn:hover {{ opacity: 0.85; text-decoration: none; }}
  .btn-primary {{ background: var(--blue); color: var(--white); }}
  .btn-outline {{ background: transparent; color: var(--white); border: 1.5px solid #475569; }}

  /* ── Stats strip ── */
  .stats-strip {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    background: var(--white);
    border-bottom: 1px solid var(--border);
  }}
  .stat-item {{
    padding: 28px 24px;
    text-align: center;
    border-right: 1px solid var(--border);
  }}
  .stat-item:last-child {{ border-right: none; }}
  .stat-val {{
    font-size: 34px;
    font-weight: 800;
    color: var(--blue);
    letter-spacing: -1px;
    line-height: 1;
  }}
  .stat-label {{
    font-size: 11.5px;
    color: var(--gray);
    margin-top: 6px;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }}

  /* ── Main layout ── */
  .main {{ max-width: 1180px; margin: 0 auto; padding: 56px 28px; }}
  .section {{ margin-bottom: 64px; }}
  .section-header {{ margin-bottom: 32px; }}
  .section-tag {{
    display: inline-block;
    background: #EFF6FF;
    color: var(--blue);
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    padding: 4px 12px;
    border-radius: 999px;
    margin-bottom: 10px;
  }}
  .section-header h2 {{
    font-size: 26px;
    font-weight: 800;
    letter-spacing: -0.5px;
  }}
  .section-header p {{
    font-size: 15px;
    color: var(--gray);
    margin-top: 8px;
    line-height: 1.65;
    max-width: 680px;
  }}

  /* ── Dashboard CTA ── */
  .dashboard-cta {{
    background: linear-gradient(135deg, #1E40AF 0%, #1D4ED8 100%);
    border-radius: 16px;
    padding: 36px 40px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 24px;
    flex-wrap: wrap;
    margin-bottom: 64px;
  }}
  .dashboard-cta h2 {{
    font-size: 22px;
    font-weight: 800;
    color: var(--white);
    margin-bottom: 8px;
  }}
  .dashboard-cta p {{
    font-size: 14px;
    color: #BFDBFE;
    max-width: 480px;
    line-height: 1.6;
  }}
  .btn-white {{
    background: var(--white);
    color: #1D4ED8;
    padding: 12px 28px;
    border-radius: 8px;
    font-size: 14px;
    font-weight: 700;
    white-space: nowrap;
  }}

  /* ── Findings ── */
  .findings-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
    gap: 20px;
  }}
  .finding-card {{
    background: var(--white);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 24px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
  }}
  .finding-icon {{ font-size: 26px; margin-bottom: 12px; }}
  .finding-card h3 {{ font-size: 15px; font-weight: 700; margin-bottom: 8px; }}
  .finding-card p {{ font-size: 13.5px; color: #374151; line-height: 1.65; }}

  /* ── Charts ── */
  .charts-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
    gap: 24px;
  }}
  .chart-card, .stat-chart-card {{
    background: var(--white);
    border: 1px solid var(--border);
    border-radius: 14px;
    overflow: hidden;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    transition: box-shadow 0.2s;
  }}
  .chart-card:hover, .stat-chart-card:hover {{
    box-shadow: 0 4px 16px rgba(0,0,0,0.10);
  }}
  .chart-card img, .stat-chart-card img {{
    width: 100%;
    display: block;
    border-bottom: 1px solid var(--border);
  }}
  .chart-info {{ padding: 16px 18px 18px; }}
  .chart-info h3 {{ font-size: 14px; font-weight: 700; margin-bottom: 6px; }}
  .chart-info p  {{ font-size: 12.5px; color: var(--gray); line-height: 1.6; }}

  /* ── Tech stack ── */
  .tech-grid {{
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
  }}
  .tech-pill {{
    background: var(--white);
    border: 1px solid var(--border);
    border-radius: 999px;
    padding: 7px 16px;
    font-size: 13px;
    font-weight: 500;
    color: var(--navy);
  }}

  /* ── Footer ── */
  footer {{
    text-align: center;
    padding: 32px 24px;
    background: var(--navy);
    color: #64748B;
    font-size: 13px;
  }}
  footer a {{ color: #94A3B8; }}

  @media (max-width: 640px) {{
    .stats-strip {{ grid-template-columns: repeat(2, 1fr); }}
    .hero {{ padding: 40px 20px 36px; }}
    .main {{ padding: 36px 16px; }}
    .dashboard-cta {{ padding: 28px 20px; }}
  }}
</style>
</head>
<body>

<!-- Hero -->
<div class="hero">
  <div class="hero-tag">Sports Analytics · Personal Project</div>
  <h1>Garmin Running Analytics</h1>
  <p>A {stats['months']}-month analysis of personal GPS running data — exploring pace trends, heart rate zones, training load, and the data story behind a runner's knee injury.</p>
  <div class="hero-btns">
    <a href="{dash_url}" class="btn btn-primary">Live Dashboard</a>
    <a href="{repo_url}" class="btn btn-outline" target="_blank">View on GitHub</a>
  </div>
</div>

<!-- Stats strip -->
<div class="stats-strip">
  <div class="stat-item">
    <div class="stat-val">{stats['total_runs']}</div>
    <div class="stat-label">Total Runs</div>
  </div>
  <div class="stat-item">
    <div class="stat-val">{stats['total_miles']}</div>
    <div class="stat-label">Miles Logged</div>
  </div>
  <div class="stat-item">
    <div class="stat-val">{stats['peak_vo2']}</div>
    <div class="stat-label">Peak VO₂Max</div>
  </div>
  <div class="stat-item">
    <div class="stat-val">{stats['median_pace']}</div>
    <div class="stat-label">Median Pace (min/mi)</div>
  </div>
</div>

<div class="main">

  <!-- Dashboard CTA -->
  <div class="dashboard-cta">
    <div>
      <h2>Weekly Training Dashboard</h2>
      <p>An auto-refreshing dashboard pulling live data from Garmin Connect. Shows CTL/ATL/TSB training load, HR zone distribution, weekly mileage, injury risk score, and a personalized coach's note.</p>
    </div>
    <a href="{dash_url}" class="btn btn-white">Open Dashboard →</a>
  </div>

  <!-- Key Findings -->
  <div class="section">
    <div class="section-header">
      <div class="section-tag">Key Findings</div>
      <h2>What the data revealed</h2>
      <p>Statistical analysis using non-parametric tests, OLS regression, and the Banister training load model.</p>
    </div>
    <div class="findings-grid">
      <div class="finding-card">
        <div class="finding-icon">📉</div>
        <h3>Pace Trend Paradox</h3>
        <p>Average pace slowed significantly over the study period (ρ = +0.40, p &lt; 0.001) — but OLS regression showed distance was not the confounder. The shift reflects a transition from short, high-intensity runs to longer aerobic base work.</p>
      </div>
      <div class="finding-card">
        <div class="finding-icon">🫀</div>
        <h3>VO₂Max Progression</h3>
        <p>VO₂Max peaked at {stats['peak_vo2']} in early 2026 before declining as injury-related mileage drops reduced aerobic stimulus — consistent with the principle that fitness requires sustained load to maintain.</p>
      </div>
      <div class="finding-card">
        <div class="finding-icon">⚠️</div>
        <h3>Injury Signature</h3>
        <p>Runner's knee onset (Feb 2026) was preceded by an 18-mile week (+112% mileage spike), TSB of −9.2, and a 9-mile "Very Hard" run three weeks before the half marathon. The data tells the injury story clearly.</p>
      </div>
      <div class="finding-card">
        <div class="finding-icon">😴</div>
        <h3>Recovery Gap Effect</h3>
        <p>Runs within 1 day of a prior run were significantly slower than those with 2+ days rest (Mann-Whitney U, p = 0.003). Adequate recovery is measurably reflected in pace performance.</p>
      </div>
    </div>
  </div>

  <!-- EDA Charts -->
  <div class="section">
    <div class="section-header">
      <div class="section-tag">Exploratory Analysis</div>
      <h2>Training patterns & physiological trends</h2>
      <p>{stats['date_range']} · {stats['total_runs']} runs · {stats['total_miles']} miles</p>
    </div>
    <div class="charts-grid">
      {chart_cards}
    </div>
  </div>

  <!-- Statistical Analysis -->
  <div class="section">
    <div class="section-header">
      <div class="section-tag">Statistical Analysis</div>
      <h2>Significance testing & regression</h2>
      <p>Spearman correlations, Kruskal-Wallis H-tests, Mann-Whitney U with Bonferroni correction, and OLS regression with confidence intervals.</p>
    </div>
    <div class="charts-grid">
      {stat_cards}
    </div>
  </div>

  <!-- Tech Stack -->
  <div class="section">
    <div class="section-header">
      <div class="section-tag">Tech Stack</div>
      <h2>Built with</h2>
    </div>
    <div class="tech-grid">
      {"".join(f'<span class="tech-pill">{t}</span>' for t in [
        "Python 3.13", "pandas", "NumPy", "Matplotlib", "Seaborn",
        "Plotly", "SciPy", "statsmodels", "fitparse", "garminconnect",
        "Jupyter", "GitHub Pages",
      ])}
    </div>
  </div>

</div>

<footer>
  <p>Built by <strong style="color:#CBD5E1">Devyn Simmons</strong> &nbsp;·&nbsp;
  <a href="{repo_url}" target="_blank">GitHub</a> &nbsp;·&nbsp; {today_year}</p>
</footer>

</body>
</html>"""


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\nBuilding portfolio site...")

    print("  Computing stats...")
    stats = load_stats()
    print(f"    {stats['total_runs']} runs · {stats['total_miles']} miles · {stats['date_range']}")

    copy_assets()

    print("  Generating docs/index.html...")
    html = generate_html(stats)
    (DOCS / "index.html").write_text(html, encoding="utf-8")

    print("\nDone! docs/ is ready to push.")
    print(f"\nNext steps:")
    print(f"  1. Replace 'YOUR-GITHUB-USERNAME' in publish.py and README.md")
    print(f"  2. git add docs/ README.md && git commit -m 'Add portfolio site'")
    print(f"  3. git push")
    print(f"  4. On GitHub: Settings → Pages → Source: main branch, /docs folder")
