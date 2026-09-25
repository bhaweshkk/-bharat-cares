# Energy & Demand Business Intelligence Pipeline

A lightweight, modular Python pipeline that turns raw smart-meter and
system-demand data into executive-ready insights, following the BI flow:

**Data → Information → Insights → Decision → Action**

---

## 1. Project Architecture & Data Flow

```
energy_bi_project/
├── data/                                  # Raw input files (place here)
│   ├── Sites Energy Consumption Part_001.csv
│   ├── Sites Energy Consumption Part_002.csv
│   ├── Sites Energy Consumption Part_003.csv
│   └── Demand.xlsx
├── src/
│   ├── data_ingestion.py                  # Step 1: Ingest & clean
│   ├── eda.py                             # Step 2: EDA & visualizations
│   ├── modeling.py                        # Step 3: Forecasting & risk model
│   └── insights.py                        # Step 4: KPIs & executive summary
├── outputs/
│   ├── charts/                            # Generated PNG charts
│   └── executive_summary.txt              # Generated text dashboard
├── .streamlit/
│   └── config.toml                        # Forces the dark theme for app.py
├── main.py                                # CLI pipeline (console + PNG charts)
├── app.py                                 # Interactive Streamlit judge dashboard
├── requirements.txt
└── README.md
```

**Data flow:**

| Stage | Module | What happens |
|---|---|---|
| **Data** | `data_ingestion.py` | Loads 3 half-hourly household consumption CSVs (`kWh` per meter) + 1 system demand-status Excel file. Parses a non-standard timestamp encoding, fixes typo'd category labels, removes duplicates, and caps sensor-glitch outliers. |
| **Information** | `eda.py` | Aggregates and visualizes consumption over time, by region, by hour, and against system demand status. |
| **Insights** | `modeling.py` | A regression model forecasts near-term total demand; a classifier quantifies *when* high-demand risk is highest and *why* (feature importance = drivers). |
| **Decision / Action** | `insights.py` | Converts KPIs + model outputs into a plain-English executive summary with concrete recommended actions. |

The two raw datasets are household-level half-hourly energy consumption
readings (`cell_id` = meter, tagged by `site_id` and `region`) and a
system-wide demand-status label (`Low` / `Normal` / `High`) for the same
half-hour periods across 2013. The pipeline merges them to connect
**local consumption behavior** with **system-level demand stress**.

---

## 2. Core KPIs, Drivers, Risks & Recommended Actions

*(Exact figures are computed dynamically from your data each run — the
values below are what this pipeline produced on the included dataset.)*

**Core KPIs**
- Total portfolio consumption, average consumption per meter, peak demand
  day, and the % of time the system spends in "High" demand.

**Identified Drivers** (from the trained risk classifier's feature
importances)
- **Seasonality (month of year)** — the strongest predictor of high-demand
  risk; consumption swings ~70%+ between the heaviest and lightest months.
- **Time of day** — a sharp evening peak (roughly 5–8 PM) dominates the
  daily load curve on both weekdays and weekends.
- **Regional variation** — one region consistently runs ~30–35% higher
  average consumption per meter than the lightest region, pointing to a
  structural (not just seasonal) driver.

**Risks**
- Concentrated high-demand windows around seasonal peaks and evening
  hours create capacity-strain and peak-tariff exposure.
- Provisioning capacity flat across the year over- or under-shoots given
  the large seasonal swing.

**Opportunities**
- The lowest-consuming region's usage profile is a benchmark that could
  be replicated in the heaviest region.
- A short-term forecast with single-digit % error gives a reliable
  planning baseline for near-term capacity/procurement decisions.

**Recommended Actions**
1. Target demand-response programs at the seasonal/time-of-day windows
   the model flags as highest-risk, to flatten the load curve.
2. Prioritize efficiency outreach in the heaviest-consuming region for
   the fastest portfolio-wide reduction.
3. Use the forecast to provision capacity ahead of seasonal peaks rather
   than reactively.
4. Operationalize the risk classifier as an early-warning score to
   trigger pre-emptive load-shedding or customer alerts.

The full, data-driven version of this summary is (re)generated on every
run at `outputs/executive_summary.txt`.

---

## 3. Quickstart

### Setup
```bash
# 1. Clone/copy this project, then from the project root:
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place the raw data files in data/ (already included if you received
#    this project pre-populated):
#    - Sites Energy Consumption Part_001.csv
#    - Sites Energy Consumption Part_002.csv
#    - Sites Energy Consumption Part_003.csv
#    - Demand.xlsx
```

### Run
```bash
python main.py
```

This runs the full pipeline and:
- Prints cleaning/model diagnostics and the executive dashboard to the console.
- Saves 4 charts to `outputs/charts/`.
- Saves the full executive summary to `outputs/executive_summary.txt`.

### Run the interactive dashboard (recommended for presentations/demos)
```bash
streamlit run app.py
```
This launches a browser-based, dark-themed executive dashboard at
`http://localhost:8501` with:
- **Sidebar filters** — region multiselect + date-range slider, applied live to every KPI and chart.
- **Hero KPI cards** — Active Meters, Total Load, Average Load, Peak Reading.
- **Tab 1 — Executive Overview**: core KPIs, demand-status composition donut, plain-English "what this means" callouts.
- **Tab 2 — Trends & Regional Drivers**: side-by-side seasonality trend + 24-hour load curve, plus a regional driver bar chart.
- **Tab 3 — AI Risk Simulator**: feature-importance chart from the trained classifier, plus live sliders (hour, month, day of week, region, meter load) that recompute the model's predicted "High demand" risk score in real time via a gauge.
- **Tab 4 — Strategic Action Plan**: forecast outlook chart and auto-generated recommended actions, risks, and opportunities — computed from the live model outputs, not hardcoded text.

The data pipeline and both models are wrapped in `@st.cache_data` /
`@st.cache_resource`, so after the first load (a few seconds), tab
switches and slider/filter changes are instant — safe for live judging.

`app.py` imports `data_ingestion.py`, `modeling.py`, and `insights.py`
directly and does not duplicate any cleaning or modeling logic; it reuses
`eda.py`'s color palette for visual consistency but renders its own
Plotly charts (rather than `eda.py`'s static matplotlib figures) so
judges get hover tooltips, zoom, and live filtering.

### What each chart shows
| File | Business question answered |
|---|---|
| `01_daily_trend.png` | Where is total consumption heading, and how does it relate to system demand stress? |
| `02_consumption_by_region.png` | Which region drives the heaviest load? |
| `03_hourly_load_curve.png` | When during the day is load concentrated (peak-hour risk)? |
| `04_demand_status_distribution.png` | Does per-meter consumption actually differ under Low/Normal/High system demand? |

---

## 4. Design Notes

- **Modeling choices are intentionally simple** (Random Forest regressor/classifier on
  calendar + lag features) — the goal is a fast, interpretable, dependency-light
  pipeline suited to strategic decision-making, not a production forecasting system.
- **Forecast validation** uses a chronological train/test split (the model must
  predict genuinely unseen future days).
- **Risk-classifier validation** uses a stratified random split with cyclical
  (sin/cos) encoding of hour/month — this model learns *recurring calendar
  patterns* rather than sequential future dependence, so a random split is the
  statistically appropriate choice and avoids tree-model extrapolation errors at
  year-end.
- **Outlier handling**: household half-hourly readings above 5 kWh are treated as
  sensor/logging glitches (physically implausible for a single domestic meter) and
  are capped rather than dropped, to preserve a continuous timeline for time-series
  modeling.
