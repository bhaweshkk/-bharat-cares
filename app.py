"""
app.py
------
Executive, judge-facing Streamlit dashboard for the Energy & Demand BI
project. This is a presentation layer only — all data cleaning,
forecasting, and risk modeling logic lives in src/ and is imported here
unchanged (data_ingestion.py, modeling.py, insights.py). eda.py's color
palette is reused for visual consistency, but charts here are rebuilt in
Plotly (rather than eda.py's static matplotlib figures) so judges get
hover tooltips, zoom, and live filtering.

Run with:  streamlit run app.py
"""

import os
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from data_ingestion import run_ingestion              # noqa: E402
from eda import PALETTE                               # noqa: E402
from insights import compute_kpis, _fmt_month         # noqa: E402
from modeling import forecast_consumption, train_demand_risk_model  # noqa: E402

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Extend the existing eda.py palette with a couple of UI-only accents.
ACCENT = PALETTE["Normal"]        # #3F6AB5 - primary blue
RISK_LOW = PALETTE["Low"]         # #4C9A8E - teal (safe)
RISK_HIGH = PALETTE["High"]       # #D65F5F - red (risk)
RISK_MID = "#E3A93A"              # amber (caution) - UI-only, not in eda.py
BG_DARK = "#0B0F19"
CARD_BG = "#141B2D"
CARD_BORDER = "#242E45"
TEXT_MUTED = "#93A0BA"

REGION_COLOR_MAP = {"A": RISK_HIGH, "B": ACCENT, "C": "#8E6FD6", "D": RISK_LOW}


# ----------------------------------------------------------------------
# Page config & global styling
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Energy & Demand BI | Executive Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(f"""
<style>
    .stApp {{
        background: radial-gradient(circle at top left, #101828 0%, {BG_DARK} 45%);
        color: #E6E9F0;
    }}
    section[data-testid="stSidebar"] {{
        background-color: #0E1424;
        border-right: 1px solid {CARD_BORDER};
    }}
    h1, h2, h3, h4 {{ color: #F4F6FB !important; font-family: 'Inter', 'Segoe UI', sans-serif; }}
    p, span, label, div {{ font-family: 'Inter', 'Segoe UI', sans-serif; }}

    /* Hero */
    .hero-title {{
        font-size: 2.4rem; font-weight: 800; letter-spacing: -0.02em;
        background: linear-gradient(90deg, #FFFFFF 0%, #9EB6FF 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.1rem;
    }}
    .hero-subtitle {{ color: {TEXT_MUTED}; font-size: 1.05rem; margin-bottom: 1.2rem; }}
    .badge {{
        display: inline-block; background: rgba(63,106,181,0.18); color: {ACCENT};
        border: 1px solid rgba(63,106,181,0.4); border-radius: 999px;
        padding: 3px 12px; font-size: 0.78rem; font-weight: 600; margin-bottom: 10px;
    }}

    /* KPI cards */
    .kpi-card {{
        background: linear-gradient(160deg, {CARD_BG} 0%, #101728 100%);
        border: 1px solid {CARD_BORDER}; border-radius: 14px;
        padding: 18px 20px; height: 118px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.25);
    }}
    .kpi-label {{ color: {TEXT_MUTED}; font-size: 0.8rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; }}
    .kpi-value {{ color: #FFFFFF; font-size: 1.7rem; font-weight: 800; margin-top: 6px; }}
    .kpi-delta {{ font-size: 0.82rem; font-weight: 600; margin-top: 4px; }}

    /* Section headers */
    .section-tag {{
        color: {ACCENT}; font-weight: 700; font-size: 0.82rem;
        text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: -6px;
    }}

    /* Insight / action cards */
    .insight-card {{
        background: {CARD_BG}; border-left: 4px solid {ACCENT};
        border-radius: 8px; padding: 14px 18px; margin-bottom: 10px;
        font-size: 0.95rem; color: #DCE2F0;
    }}
    .insight-card.risk {{ border-left-color: {RISK_HIGH}; }}
    .insight-card.opportunity {{ border-left-color: {RISK_LOW}; }}
    .action-card {{
        background: linear-gradient(135deg, {CARD_BG} 0%, #10182A 100%);
        border: 1px solid {CARD_BORDER}; border-radius: 12px;
        padding: 16px 18px; margin-bottom: 12px;
    }}
    .action-number {{
        display: inline-flex; align-items: center; justify-content: center;
        width: 26px; height: 26px; border-radius: 50%; background: {ACCENT};
        color: white; font-weight: 700; font-size: 0.85rem; margin-right: 10px;
    }}

    div[data-testid="stMetricValue"] {{ color: #FFFFFF; }}
    hr {{ border-color: {CARD_BORDER}; }}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------
# Cached data / model loading — kept instant across judge Q&A
# ----------------------------------------------------------------------
@st.cache_data(show_spinner="Ingesting and cleaning half-hourly meter data…")
def load_data() -> pd.DataFrame:
    return run_ingestion(DATA_DIR)


@st.cache_resource(show_spinner="Training the 14-day consumption forecaster…")
def get_forecast(_df: pd.DataFrame) -> dict:
    return forecast_consumption(_df)


@st.cache_resource(show_spinner="Training the AI high-demand risk classifier…")
def get_risk_model(_df: pd.DataFrame) -> dict:
    return train_demand_risk_model(_df)


@st.cache_data(show_spinner=False)
def get_kpis(_df: pd.DataFrame) -> dict:
    return compute_kpis(_df)


def build_risk_features(feature_cols, hour, month, day_of_week, avg_kwh) -> pd.DataFrame:
    """Build a single-row feature vector matching modeling.py's exact
    cyclical encoding, so the live simulator stays consistent with the
    trained model regardless of future changes to that encoding."""
    is_weekend = int(day_of_week >= 5)
    row = {
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "month_sin": np.sin(2 * np.pi * month / 12),
        "month_cos": np.cos(2 * np.pi * month / 12),
        "day_of_week": day_of_week,
        "is_weekend": is_weekend,
        "avg_kwh": avg_kwh,
    }
    return pd.DataFrame([row])[feature_cols]


# ----------------------------------------------------------------------
# Load everything once (cached)
# ----------------------------------------------------------------------
df = load_data()
forecast = get_forecast(df)
risk = get_risk_model(df)

MIN_DATE, MAX_DATE = df["timestamp"].min().date(), df["timestamp"].max().date()
ALL_REGIONS = sorted(df["region"].dropna().unique().tolist())

# ----------------------------------------------------------------------
# Sidebar — filters & project badge
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown('<span class="badge">⚡ HACKATHON DEMO BUILD</span>', unsafe_allow_html=True)
    st.markdown("### Energy & Demand BI")
    st.caption(
        "Household smart-meter consumption merged with system-wide demand "
        "status, turned into forecasts, risk scores, and action plans."
    )
    st.divider()

    st.markdown("**Filters**")
    selected_regions = st.multiselect("Region", options=ALL_REGIONS, default=ALL_REGIONS)
    date_range = st.slider(
        "Date range",
        min_value=MIN_DATE, max_value=MAX_DATE,
        value=(MIN_DATE, MAX_DATE), format="YYYY-MM-DD",
    )

    st.divider()
    st.markdown("**Pipeline snapshot**")
    st.caption(f"📅 {MIN_DATE} → {MAX_DATE}")
    st.caption(f"🔌 {df['cell_id'].nunique()} meters · {len(ALL_REGIONS)} regions")
    st.caption(f"🌲 Forecast MAPE: {forecast['test_mape']:.1f}%  ·  Risk model AUC: {risk['auc']:.2f}")

if not selected_regions:
    st.warning("Select at least one region in the sidebar to see data.")
    st.stop()

mask = (
    df["region"].isin(selected_regions)
    & (df["timestamp"].dt.date >= date_range[0])
    & (df["timestamp"].dt.date <= date_range[1])
)
fdf = df[mask]

if fdf.empty:
    st.warning("No data in the selected filter range. Widen the date range or region selection.")
    st.stop()

# ----------------------------------------------------------------------
# Hero header
# ----------------------------------------------------------------------
st.markdown('<span class="badge">DATA → INFORMATION → INSIGHTS → DECISION → ACTION</span>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">Energy & Demand Business Intelligence</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">From raw half-hourly smart-meter readings to forecasted risk '
    'and strategic action — live, interactive, judge-ready.</div>',
    unsafe_allow_html=True,
)

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
hero_metrics = [
    (kpi1, "Active Meters", f"{fdf['cell_id'].nunique():,}", "monitored in current selection"),
    (kpi2, "Total Load", f"{fdf['kwh'].sum():,.0f} kWh", "cumulative selected consumption"),
    (kpi3, "Average Load", f"{fdf['kwh'].mean():.3f} kWh", "per meter, per half-hour"),
    (kpi4, "Peak Reading", f"{fdf['kwh'].max():.2f} kWh", "single highest half-hour reading"),
]
for col, label, value, sub in hero_metrics:
    with col:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-delta" style="color:{TEXT_MUTED};">{sub}</div>
        </div>
        """, unsafe_allow_html=True)

st.write("")

# ----------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊  Executive Overview",
    "📈  Trends & Regional Drivers",
    "🤖  AI Risk Simulator",
    "🎯  Strategic Action Plan",
])

# ======================================================================
# TAB 1 — Executive Overview & KPIs
# ======================================================================
with tab1:
    kpis = get_kpis(fdf)

    st.markdown('<div class="section-tag">WHAT IS HAPPENING</div>', unsafe_allow_html=True)
    st.markdown("### Core KPIs at a glance")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Avg. annual consumption / meter", f"{kpis['avg_kwh_per_meter']:,.0f} kWh")
        st.metric("Heaviest region", f"Region {kpis['top_region']}", f"{kpis['top_region_avg_kwh']:,.0f} kWh/meter/yr")
    with c2:
        st.metric("Peak single day", str(kpis["peak_day"]), f"{kpis['peak_day_kwh']:,.0f} kWh")
        st.metric("Time in 'High' system demand", f"{kpis['pct_time_high_demand']*100:.1f}%")
    with c3:
        st.metric("Seasonal swing (peak vs. trough month)", f"{kpis['seasonal_swing_pct']:.0f}%")
        st.metric("Heaviest / lightest month", f"{_fmt_month(kpis['winter_month'])} / {_fmt_month(kpis['summer_month'])}")

    st.write("")
    left, right = st.columns([1.3, 1])

    with left:
        st.markdown("##### Demand status composition")
        status_counts = fdf["demand_status"].value_counts().reindex(["Low", "Normal", "High"]).dropna()
        fig = px.pie(
            values=status_counts.values, names=status_counts.index,
            color=status_counts.index, color_discrete_map=PALETTE, hole=0.55,
        )
        fig.update_traces(
            textinfo="percent+label",
            hovertemplate="<b>%{label}</b><br>%{value:,} periods (%{percent})<extra></extra>",
        )
        fig.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False, margin=dict(t=10, b=10, l=10, r=10), height=320,
        )
        st.plotly_chart(fig, width='stretch')

    with right:
        st.markdown("##### What this means")
        st.markdown(f"""
        <div class="insight-card">
            The portfolio holds <b>{kpis['n_meters']} meters</b> across
            <b>{len(ALL_REGIONS)} regions</b>, totalling
            <b>{kpis['total_kwh']:,.0f} kWh</b> over the reporting window.
        </div>
        <div class="insight-card risk">
            <b>Risk:</b> the system spends <b>{kpis['pct_time_high_demand']*100:.1f}%</b> of all
            half-hour periods in "High" demand — a small but structurally
            recurring exposure window (see the AI Risk Simulator tab).
        </div>
        <div class="insight-card opportunity">
            <b>Opportunity:</b> Region <b>{kpis['lowest_region']}</b> runs the
            lightest per-meter load — a natural efficiency benchmark for
            Region <b>{kpis['top_region']}</b>.
        </div>
        """, unsafe_allow_html=True)

# ======================================================================
# TAB 2 — Demand Trends & Regional Drivers
# ======================================================================
with tab2:
    st.markdown('<div class="section-tag">WHERE IS IT GOING & WHY</div>', unsafe_allow_html=True)
    st.markdown("### Seasonality, daily rhythm, and regional drivers")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("##### Daily consumption trend (seasonality)")
        daily = fdf.groupby(fdf["timestamp"].dt.floor("D")).agg(
            total_kwh=("kwh", "sum"),
            high_share=("demand_status", lambda s: (s == "High").mean() * 100),
        ).reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=daily["timestamp"], y=daily["total_kwh"], mode="lines", name="Total kWh",
            line=dict(color=ACCENT, width=2), fill="tozeroy", fillcolor="rgba(63,106,181,0.12)",
            hovertemplate="%{x|%b %d, %Y}<br>Total: %{y:,.0f} kWh<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=daily["timestamp"], y=daily["high_share"], mode="lines", name="% High demand",
            line=dict(color=RISK_HIGH, width=1.5, dash="dot"), yaxis="y2",
            hovertemplate="%{x|%b %d, %Y}<br>High demand: %{y:.0f}%<extra></extra>",
        ))
        fig.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=10, b=10, l=10, r=10), height=380,
            legend=dict(orientation="h", y=1.12, x=0),
            yaxis=dict(title="Total daily kWh"),
            yaxis2=dict(title="% High demand", overlaying="y", side="right", range=[0, 100], showgrid=False),
            hovermode="x unified",
        )
        st.plotly_chart(fig, width='stretch')
        st.caption("The classic winter-high / summer-low 'U-shape' — the strongest recurring pattern in the data.")

    with c2:
        st.markdown("##### 24-hour load curve")
        tmp = fdf.copy()
        tmp["hour"] = tmp["timestamp"].dt.hour
        tmp["day_type"] = np.where(tmp["timestamp"].dt.dayofweek >= 5, "Weekend", "Weekday")
        hourly = tmp.groupby(["day_type", "hour"], observed=True)["kwh"].mean().reset_index()
        fig = px.line(
            hourly, x="hour", y="kwh", color="day_type", markers=True,
            color_discrete_map={"Weekday": ACCENT, "Weekend": RISK_MID},
            labels={"hour": "Hour of day", "kwh": "Avg. kWh per meter", "day_type": ""},
        )
        fig.update_traces(hovertemplate="Hour %{x}:00<br>%{y:.3f} kWh<extra></extra>")
        fig.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=10, b=10, l=10, r=10), height=380,
            legend=dict(orientation="h", y=1.12, x=0),
            xaxis=dict(dtick=2),
        )
        st.plotly_chart(fig, width='stretch')
        st.caption("A sharp evening peak (~5–8 PM) dominates both weekdays and weekends.")

    st.write("")
    st.markdown("##### Regional driver: average annual consumption per meter")
    per_meter = fdf.groupby(["region", "cell_id"], observed=True)["kwh"].sum().reset_index()
    region_avg = per_meter.groupby("region", observed=True)["kwh"].mean().sort_values(ascending=False).reset_index()
    fig = px.bar(
        region_avg, x="region", y="kwh", color="region",
        color_discrete_map=REGION_COLOR_MAP, text_auto=".0f",
        labels={"region": "Region", "kwh": "Avg. kWh per meter"},
    )
    fig.update_traces(hovertemplate="Region %{x}<br>%{y:,.0f} kWh/meter<extra></extra>", textfont_color="white")
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=10, b=10, l=10, r=10), height=340, showlegend=False,
    )
    st.plotly_chart(fig, width='stretch')

# ======================================================================
# TAB 3 — AI Risk Prediction & Interactive Simulator
# ======================================================================
with tab3:
    st.markdown('<div class="section-tag">WHAT COULD GO WRONG</div>', unsafe_allow_html=True)
    st.markdown("### AI-powered high-demand risk model")

    m1, m2, m3 = st.columns(3)
    m1.metric("Model", "Random Forest Classifier")
    m2.metric("Held-out ROC-AUC", f"{risk['auc']:.3f}")
    m3.metric("Baseline 'High' rate", f"{risk['high_demand_share']*100:.1f}%")

    left, right = st.columns([1, 1.2])

    with left:
        st.markdown("##### What drives high-demand risk")
        imp = risk["feature_importances"].sort_values(ascending=True)
        labels = {
            "hour_of_day": "Time of day", "month_of_year": "Season (month)",
            "day_of_week": "Day of week", "is_weekend": "Weekend/weekday",
            "avg_meter_load": "Underlying meter load",
        }
        fig = px.bar(
            x=imp.values, y=[labels.get(i, i) for i in imp.index], orientation="h",
            color=imp.values, color_continuous_scale=[RISK_LOW, RISK_MID, RISK_HIGH],
            labels={"x": "Relative importance", "y": ""},
        )
        fig.update_traces(hovertemplate="%{y}: %{x:.1%}<extra></extra>")
        fig.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=10, b=10, l=10, r=10), height=300, coloraxis_showscale=False,
            xaxis_tickformat=".0%",
        )
        st.plotly_chart(fig, width='stretch')

        with st.expander("Held-out classification report"):
            st.code(risk["classification_report"])

    with right:
        st.markdown("##### 🎛️ Live risk simulator — try your own scenario")
        st.caption(
            "Adjust the sliders to simulate a half-hour period and see the "
            "model's live predicted risk of 'High' system demand."
        )

        region_avg_lookup = df.groupby("region", observed=True)["kwh"].mean()
        default_region = region_avg_lookup.idxmax()

        s1, s2 = st.columns(2)
        with s1:
            sim_hour = st.slider("Hour of day", 0, 23, 18)
            sim_month = st.select_slider(
                "Month", options=list(range(1, 13)), value=1,
                format_func=lambda m: _fmt_month(m)[:3],
            )
        with s2:
            sim_day = st.selectbox(
                "Day of week", options=list(range(7)), index=0,
                format_func=lambda d: ["Monday", "Tuesday", "Wednesday", "Thursday",
                                        "Friday", "Saturday", "Sunday"][d],
            )
            sim_region = st.selectbox(
                "Region (sets typical load)", options=ALL_REGIONS,
                index=ALL_REGIONS.index(default_region) if default_region in ALL_REGIONS else 0,
            )

        default_load = float(region_avg_lookup.get(sim_region, df["kwh"].mean()))
        sim_avg_kwh = st.slider(
            "Avg. meter load for this period (kWh)", 0.0, float(df["kwh"].quantile(0.99)),
            value=round(default_load, 3), step=0.01,
        )

        feats = build_risk_features(risk["feature_cols"], sim_hour, sim_month, sim_day, sim_avg_kwh)
        risk_score = float(risk["model"].predict_proba(feats)[:, 1][0])

        if risk_score < 0.33:
            risk_color, risk_label = RISK_LOW, "LOW RISK"
        elif risk_score < 0.66:
            risk_color, risk_label = RISK_MID, "MODERATE RISK"
        else:
            risk_color, risk_label = RISK_HIGH, "ELEVATED RISK"

        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_score * 100,
            number={"suffix": "%", "font": {"color": "#FFFFFF", "size": 40}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": TEXT_MUTED},
                "bar": {"color": risk_color, "thickness": 0.35},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 33], "color": "rgba(76,154,142,0.18)"},
                    {"range": [33, 66], "color": "rgba(227,169,58,0.18)"},
                    {"range": [66, 100], "color": "rgba(214,95,95,0.18)"},
                ],
            },
        ))
        gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", height=240,
            margin=dict(t=30, b=10, l=30, r=30),
            font={"color": "#E6E9F0"},
        )
        st.plotly_chart(gauge, width='stretch')
        st.markdown(
            f"<div style='text-align:center; margin-top:-15px;'>"
            f"<span style='background:{risk_color}22; color:{risk_color}; padding:4px 14px; "
            f"border-radius:999px; font-weight:700; font-size:0.85rem;'>{risk_label}</span></div>",
            unsafe_allow_html=True,
        )

        st.write("")
        if risk_score < 0.33:
            st.success(
                f"Low exposure ({risk_score*100:.0f}%) for this scenario — normal operating conditions, "
                "no special action needed."
            )
        elif risk_score < 0.66:
            st.warning(
                f"Moderate exposure ({risk_score*100:.0f}%) — worth monitoring; consider soft demand-response "
                "nudges (pricing signals, notifications) if this window recurs."
            )
        else:
            st.error(
                f"Elevated exposure ({risk_score*100:.0f}%) — this scenario closely matches historical "
                "'High' demand conditions. Consider pre-emptive load-shedding or capacity buffers."
            )

# ======================================================================
# TAB 4 — Strategic Action Plan
# ======================================================================
with tab4:
    kpis_full = get_kpis(df)
    direction = "up" if forecast["trend_pct"] > 0 else "down"
    top_driver_key = risk["feature_importances"].index[0]
    driver_labels = {
        "hour_of_day": "time-of-day", "month_of_year": "seasonal (month-of-year)",
        "day_of_week": "day-of-week", "is_weekend": "weekend/weekday",
        "avg_meter_load": "underlying meter load",
    }
    top_driver = driver_labels.get(top_driver_key, top_driver_key.replace("_", " "))

    st.markdown('<div class="section-tag">WHAT SHOULD WE DO NEXT</div>', unsafe_allow_html=True)
    st.markdown("### Strategic action plan")
    st.caption("Derived automatically from the current model outputs and full-year KPIs — not hardcoded.")

    st.markdown("##### 14-day forward outlook")
    f1, f2, f3 = st.columns(3)
    f1.metric("Trailing 14-day avg.", f"{forecast['recent_avg_daily_kwh']:,.0f} kWh/day")
    f2.metric("Forecast 14-day avg.", f"{forecast['forecast_avg_daily_kwh']:,.0f} kWh/day",
              f"{'+' if direction == 'up' else '-'}{abs(forecast['trend_pct']):.1f}%")
    f3.metric("Forecast accuracy (test MAPE)", f"{forecast['test_mape']:.1f}%")

    fc_df = forecast["forecast_df"]
    fig = px.line(
        fc_df, x="date", y="forecast_kwh", markers=True,
        labels={"date": "", "forecast_kwh": "Forecast kWh/day"},
    )
    fig.update_traces(line=dict(color=ACCENT, width=2.5), hovertemplate="%{x|%b %d}<br>%{y:,.0f} kWh<extra></extra>")
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=10, b=10, l=10, r=10), height=260,
    )
    st.plotly_chart(fig, width='stretch')

    st.write("")
    st.markdown("##### Recommended actions")

    actions = [
        f"Target demand-response programs at <b>{top_driver}</b> peaks — the single largest "
        f"predictor of High-demand risk — to flatten the load curve.",
        f"Prioritize efficiency/audit outreach in <b>Region {kpis_full['top_region']}</b>, the highest "
        f"per-meter consumer, for the fastest portfolio-wide reduction.",
        f"Provision {'additional' if direction == 'up' else 'more flexible'} capacity ahead of "
        f"<b>{_fmt_month(kpis_full['winter_month'])}</b>-level seasonal peaks, using the forecast "
        f"(±{forecast['test_mape']:.0f}% MAPE) as a planning baseline.",
        "Monitor the 'High' demand risk score in near-real time (via the trained classifier) to trigger "
        "pre-emptive load-shedding or customer alerts before strain occurs.",
    ]
    for i, action in enumerate(actions, start=1):
        st.markdown(f"""
        <div class="action-card">
            <span class="action-number">{i}</span>{action}
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    r1, r2 = st.columns(2)
    with r1:
        st.markdown(f"""
        <div class="insight-card risk">
            <b>Risk:</b> Consumption swings <b>{kpis_full['seasonal_swing_pct']:.0f}%</b> between
            {_fmt_month(kpis_full['summer_month'])} and {_fmt_month(kpis_full['winter_month'])},
            requiring seasonal capacity buffers rather than flat provisioning.
        </div>
        """, unsafe_allow_html=True)
    with r2:
        st.markdown(f"""
        <div class="insight-card opportunity">
            <b>Opportunity:</b> The near-term trend is {direction} <b>{abs(forecast['trend_pct']):.1f}%</b> —
            {"a window to pre-position capacity/procurement." if direction == "up"
             else "room for a controlled pull-back in reserve procurement."}
        </div>
        """, unsafe_allow_html=True)

st.write("")
st.divider()
st.caption(
    "Energy & Demand BI · Random Forest forecasting + risk classification · "
    "Built on cleaned half-hourly smart-meter data merged with system demand status."
)
