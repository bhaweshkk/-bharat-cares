"""
insights.py
-----------
Converts cleaned data + model outputs into an executive-ready summary:
KPIs, drivers, risks/opportunities, and concrete recommended actions.

This is the "so what" layer of the BI flow (Data -> Information ->
Insights -> Decision -> Action) — everything computed here is derived
dynamically from the current dataset, not hardcoded, so the summary
stays accurate if the underlying data changes.
"""

import pandas as pd


def compute_kpis(df: pd.DataFrame) -> dict:
    """Compute the core portfolio KPIs from the cleaned, merged dataset."""
    total_kwh = df["kwh"].sum()
    n_meters = df["cell_id"].nunique()
    date_range = (df["timestamp"].min().date(), df["timestamp"].max().date())

    per_meter_total = df.groupby("cell_id", observed=True)["kwh"].sum()
    region_avg = df.groupby(["region", "cell_id"], observed=True)["kwh"].sum().groupby("region", observed=True).mean()
    top_region = region_avg.idxmax()

    daily_total = df.groupby(df["timestamp"].dt.floor("D"))["kwh"].sum()
    peak_day = daily_total.idxmax()

    demand_share = df["demand_status"].value_counts(normalize=True)
    high_share = float(demand_share.get("High", 0.0))

    # Seasonal swing: compare the heaviest vs lightest consumption month.
    monthly = df.groupby(df["timestamp"].dt.month)["kwh"].sum()
    seasonal_swing_pct = (monthly.max() - monthly.min()) / monthly.min() * 100

    return {
        "total_kwh": total_kwh,
        "n_meters": n_meters,
        "date_range": date_range,
        "avg_kwh_per_meter": per_meter_total.mean(),
        "top_region": top_region,
        "top_region_avg_kwh": region_avg.max(),
        "lowest_region": region_avg.idxmin(),
        "peak_day": peak_day.date(),
        "peak_day_kwh": daily_total.max(),
        "pct_time_high_demand": high_share,
        "seasonal_swing_pct": seasonal_swing_pct,
        "winter_month": monthly.idxmax(),
        "summer_month": monthly.idxmin(),
    }


def _fmt_month(month_num: int) -> str:
    return pd.Timestamp(2013, month_num, 1).strftime("%B")


# Human-readable phrasing for each risk-model driver, used in narrative text.
_DRIVER_PHRASES = {
    "hour_of_day": "time-of-day",
    "month_of_year": "seasonal (month-of-year)",
    "day_of_week": "day-of-week",
    "is_weekend": "weekend/weekday",
    "avg_meter_load": "underlying meter load",
}


def _driver_phrase(feature_name: str) -> str:
    return _DRIVER_PHRASES.get(feature_name, feature_name.replace("_", " "))


def build_executive_summary(kpis: dict, forecast: dict, risk: dict) -> str:
    """Assemble the full console/text executive dashboard."""
    lines = []
    add = lines.append

    add("=" * 78)
    add("EXECUTIVE ENERGY BI DASHBOARD")
    add(f"Reporting period: {kpis['date_range'][0]} to {kpis['date_range'][1]}")
    add("=" * 78)

    # ---- 1. What is happening? (KPIs) ----
    add("\n1. WHAT IS HAPPENING — CORE KPIs")
    add("-" * 78)
    add(f"  Total portfolio consumption      : {kpis['total_kwh']:,.0f} kWh")
    add(f"  Meters monitored                 : {kpis['n_meters']}")
    add(f"  Avg. annual consumption / meter  : {kpis['avg_kwh_per_meter']:,.0f} kWh")
    add(f"  Peak single day                  : {kpis['peak_day']} ({kpis['peak_day_kwh']:,.0f} kWh)")
    add(f"  Time spent in 'High' system demand: {kpis['pct_time_high_demand']*100:.1f}% of all periods")

    # ---- 2. Where is it going? (Trends) ----
    add("\n2. WHERE IS IT GOING — TREND & FORECAST")
    add("-" * 78)
    direction = "up" if forecast["trend_pct"] > 0 else "down"
    add(f"  Seasonal swing (peak vs. trough month): {kpis['seasonal_swing_pct']:.0f}%")
    add(f"    Heaviest month: {_fmt_month(kpis['winter_month'])}  |  "
        f"Lightest month: {_fmt_month(kpis['summer_month'])}")
    add(f"  14-day forecast vs. trailing 14-day average: {direction} {abs(forecast['trend_pct']):.1f}%")
    add(f"    Trailing avg: {forecast['recent_avg_daily_kwh']:,.0f} kWh/day  ->  "
        f"Forecast avg: {forecast['forecast_avg_daily_kwh']:,.0f} kWh/day")
    add(f"    Forecast model accuracy (test MAPE): {forecast['test_mape']:.1f}%")

    # ---- 3. Why is it happening? (Drivers) ----
    add("\n3. WHY IS IT HAPPENING — KEY DRIVERS")
    add("-" * 78)
    add(f"  Heaviest-consuming region: Region {kpis['top_region']} "
        f"({kpis['top_region_avg_kwh']:,.0f} kWh/meter/yr, vs. "
        f"lightest Region {kpis['lowest_region']})")
    add("  Top drivers of 'High' system demand periods (model feature importance):")
    for feature, importance in risk["feature_importances"].head(3).items():
        add(f"    - {_driver_phrase(feature)}: {importance*100:.0f}% relative importance")

    # ---- 4. What could go wrong/right? (Risks & Opportunities) ----
    add("\n4. RISKS & OPPORTUNITIES")
    add("-" * 78)
    add(f"  RISK: 'High' demand periods occur {kpis['pct_time_high_demand']*100:.1f}% of the time, "
        f"driven mainly by {_driver_phrase(risk['feature_importances'].index[0])} patterns — "
        "unplanned exposure here risks capacity strain or peak tariff costs.")
    add(f"  RISK: Consumption swings {kpis['seasonal_swing_pct']:.0f}% between "
        f"{_fmt_month(kpis['summer_month'])} and {_fmt_month(kpis['winter_month'])}, "
        "requiring seasonal capacity buffers rather than flat provisioning.")
    add(f"  OPPORTUNITY: Region {kpis['lowest_region']}'s lower per-meter load profile could be "
        f"benchmarked and applied to Region {kpis['top_region']} to curb the heaviest draw.")
    add(f"  OPPORTUNITY: The near-term trend is {direction} {abs(forecast['trend_pct']):.1f}%, "
        + ("giving a window to pre-position capacity/procurement." if direction == "up"
           else "allowing a controlled pull-back in reserve procurement."))

    # ---- 5. What should we do next? (Recommendations) ----
    add("\n5. RECOMMENDED ACTIONS")
    add("-" * 78)
    top_driver = _driver_phrase(risk["feature_importances"].index[0])
    add(f"  1. Target demand-response programs at {top_driver} peaks — "
        "the single largest predictor of High-demand risk — to flatten the load curve.")
    add(f"  2. Prioritize efficiency/audit outreach in Region {kpis['top_region']}, the highest "
        "per-meter consumer, for the fastest portfolio-wide reduction.")
    add(f"  3. Provision {'additional' if direction == 'up' else 'more flexible'} capacity ahead of "
        f"{_fmt_month(kpis['winter_month'])}-level seasonal peaks, using the forecast "
        f"(±{forecast['test_mape']:.0f}% MAPE) as a planning baseline.")
    add("  4. Monitor the 'High' demand risk score in near-real time (via the trained classifier) "
        "to trigger pre-emptive load-shedding or customer alerts before strain occurs.")

    add("\n" + "=" * 78)
    return "\n".join(lines)
