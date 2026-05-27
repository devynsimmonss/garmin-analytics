# Garmin Running Analytics

A personal sports analytics project analyzing 16+ months of GPS running data exported from Garmin Connect. Built as a portfolio piece targeting sports analytics roles.

**[Live Portfolio Site](https://devynsimmonss.github.io/garmin-analytics)** &nbsp;|&nbsp; **[Interactive Dashboard](https://devynsimmonss.github.io/garmin-analytics/dashboard.html)**

---

## Overview

Parses, cleans, and analyzes raw Garmin Connect data (JSON export + FIT files) to surface training trends, physiological progression, and injury risk signals across ~70 runs from December 2024 through April 2026 — including a half marathon build and a runner's knee injury recovery.

## Key Findings

- **Pace slowed +0.024 min/mile per week** over the study period (ρ = +0.40, p < 0.001) — driven by increasing run volume, not declining fitness
- **VO₂Max peaked at 48** in early 2026 before declining as mileage dropped during injury
- **80% of training time in Zone 1–2** (aerobic base), consistent with polarized training theory
- **Injury preceded by an 18-mile week** (+112% mileage spike) and TSB of −9.2, flagging the exact mechanism of runner's knee onset
- **Recovery gap analysis** showed runs within 1 day of each other correlated with significantly slower pace (Mann-Whitney p = 0.003)

## Project Structure

```
garmin-analytics/
├── src/
│   ├── parse.py          — FIT file parsing (semicircle GPS, enhanced speed)
│   ├── clean.py          — Garmin JSON normalization (cm → miles, cm/ms → pace)
│   ├── features.py       — Feature engineering (TRIMP, ATL/CTL/TSB, HR zones)
│   └── garmin_api.py     — Garmin Connect API integration
├── notebooks/
│   ├── 01_eda.py                  — EDA script (7 plots)
│   ├── 01_eda.ipynb               — EDA Jupyter notebook
│   ├── 02_statistical_analysis.ipynb — Significance testing + OLS regression
│   └── weekly_dashboard.py        — Live HTML dashboard (opens in browser)
├── outputs/              — Generated charts and dashboard HTML
├── docs/                 — GitHub Pages portfolio site
├── data_sample/          — Anonymized sample data (safe to share)
├── publish.py            — Copies outputs → docs/assets, regenerates site
└── requirements.txt
```

## Feature Engineering

| Feature | Description |
|---|---|
| `trimp` | Training Impulse = duration × HR intensity ratio |
| `atl` | Acute Training Load (7-day EWM, fatigue proxy) |
| `ctl` | Chronic Training Load (42-day EWM, fitness proxy) |
| `tsb` | Training Stress Balance = CTL − ATL (form) |
| `efficiency_factor` | Speed (mph) / avg HR × 1000 (aerobic economy) |
| `grade_adj_pace` | Pace corrected for net elevation (Minetti model) |
| `weekly_miles_chg` | % mileage change week-over-week (injury risk signal) |
| `polarization_index` | Time at extremes (Z1+Z2) vs middle (Z3) |

## Statistical Analysis

- **Normality**: Shapiro-Wilk (all key metrics non-normal → non-parametric tests)
- **Correlations**: Spearman rank correlation matrix
- **Pace trend**: OLS regression with 95% confidence intervals
- **Group comparisons**: Kruskal-Wallis H-test + Mann-Whitney U with Bonferroni correction
- **VO₂Max predictors**: Multiple OLS regression (TSB, efficiency factor, zone distribution)

## Setup

```bash
git clone https://github.com/devynsimmonss/garmin-analytics.git
cd garmin-analytics
pip install -r requirements.txt
```

To run the EDA:
```bash
python notebooks/01_eda.py
```

To run the weekly dashboard (requires Garmin Connect credentials):
```bash
cp .env.example .env   # fill in GARMIN_EMAIL and GARMIN_PASSWORD
python notebooks/weekly_dashboard.py
```

To publish the portfolio site:
```bash
python publish.py
git add docs/ && git commit -m "Update portfolio site" && git push
```

## Tech Stack

Python · pandas · NumPy · Matplotlib · Seaborn · Plotly · SciPy · statsmodels · fitparse · garminconnect

## Data

Raw Garmin data is **not** included in this repository (personal GPS and health data). A small anonymized sample is available in `data_sample/`. To use your own data, export from [Garmin Connect](https://www.garmin.com/en-US/account/datamanagement/exportdata/) and place in `data/`.

---

*Built by Devyn Simmons · Sports Analytics Portfolio*
