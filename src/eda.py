"""
eda.py
------
Exploratory Data Analysis: produces a small set of high-impact,
business-oriented charts rather than an exhaustive dashboard.

Each chart is chosen to answer one specific business question:
  1. daily_trend.png        -> Where is total consumption heading, and how
                                does it track system demand stress?
  2. consumption_by_region.png -> Which regions/sites drive the load?
  3. hourly_load_curve.png  -> When during the day is load concentrated
                                (peak-hour risk)?
  4. demand_status_distribution.png -> How does per-meter consumption
                                behave under Low/Normal/High system demand?
"""

import os

import matplotlib
matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Minimal, modern, consistent visual identity for every chart.
PALETTE = {"Low": "#4C9A8E", "Normal": "#3F6AB5", "High": "#D65F5F"}
ACCENT = "#3F6AB5"
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#444444",
    "axes.titleweight": "bold",
    "axes.titlesize": 15,
    "axes.labelsize": 12,
    "font.family": "sans-serif",
})


def _save(fig, output_dir: str, filename: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_daily_trend(df: pd.DataFrame, output_dir: str) -> str:
    """Total daily consumption across the whole portfolio, with the daily
    share of time spent in 'High' system demand overlaid."""
    daily = df.groupby(df["timestamp"].dt.date).agg(
        total_kwh=("kwh", "sum"),
        high_share=("demand_status", lambda s: (s == "High").mean()),
    )
    daily.index = pd.to_datetime(daily.index)

    fig, ax1 = plt.subplots(figsize=(11, 5.5))
    ax1.plot(daily.index, daily["total_kwh"], color=ACCENT, linewidth=1.6, label="Total consumption")
    ax1.fill_between(daily.index, daily["total_kwh"], color=ACCENT, alpha=0.08)
    ax1.set_ylabel("Total daily consumption (kWh)")
    ax1.set_title("Portfolio Energy Consumption Trend (2013)")

    ax2 = ax1.twinx()
    ax2.plot(daily.index, daily["high_share"] * 100, color=PALETTE["High"],
              linewidth=1.2, alpha=0.7, label="% time in High demand")
    ax2.set_ylabel("% of day in High system demand", color=PALETTE["High"])
    ax2.tick_params(axis="y", colors=PALETTE["High"])
    ax2.grid(False)

    lines, labels = [], []
    for ax in (ax1, ax2):
        l, lab = ax.get_legend_handles_labels()
        lines += l
        labels += lab
    ax1.legend(lines, labels, loc="upper left", frameon=False, fontsize=10)

    return _save(fig, output_dir, "01_daily_trend.png")


def plot_consumption_by_region(df: pd.DataFrame, output_dir: str) -> str:
    """Average per-meter consumption by region — identifies which regions
    are structurally heavier consumers (a demand driver)."""
    per_meter = df.groupby(["region", "cell_id"], observed=True)["kwh"].sum().reset_index()
    region_avg = (
        per_meter.groupby("region", observed=True)["kwh"]
        .mean()
        .sort_values(ascending=False)
    )

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.bar(region_avg.index.astype(str), region_avg.values, color=ACCENT, width=0.6)
    bars[0].set_color(PALETTE["High"])  # highlight the top-driving region
    ax.set_title("Average Annual Consumption per Meter, by Region")
    ax.set_xlabel("Region")
    ax.set_ylabel("Avg. annual consumption per meter (kWh)")
    for bar in bars:
        ax.annotate(f"{bar.get_height():,.0f}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center", va="bottom", fontsize=10, color="#333333")

    return _save(fig, output_dir, "02_consumption_by_region.png")


def plot_hourly_load_curve(df: pd.DataFrame, output_dir: str) -> str:
    """Average load by hour of day, split weekday vs weekend — reveals
    peak-hour risk windows for capacity planning."""
    tmp = df.copy()
    tmp["hour"] = tmp["timestamp"].dt.hour
    tmp["day_type"] = tmp["timestamp"].dt.dayofweek.apply(
        lambda d: "Weekend" if d >= 5 else "Weekday"
    )
    hourly = tmp.groupby(["day_type", "hour"], observed=True)["kwh"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for day_type, color in [("Weekday", ACCENT), ("Weekend", "#E39B3A")]:
        subset = hourly[hourly["day_type"] == day_type]
        ax.plot(subset["hour"], subset["kwh"], marker="o", markersize=4,
                linewidth=2, label=day_type, color=color)
    ax.set_title("Average Load Curve by Hour of Day")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Avg. consumption per meter (kWh)")
    ax.set_xticks(range(0, 24, 2))
    ax.legend(frameon=False)

    return _save(fig, output_dir, "03_hourly_load_curve.png")


def plot_demand_status_distribution(df: pd.DataFrame, output_dir: str) -> str:
    """Per-reading consumption distribution split by system demand status —
    shows whether high-demand periods coincide with genuinely higher
    per-meter load or are driven by other system factors."""
    ordered = ["Low", "Normal", "High"]
    plot_df = df[df["demand_status"].isin(ordered)]

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    sns.boxplot(
        data=plot_df, x="demand_status", y="kwh", order=ordered,
        hue="demand_status", palette=PALETTE, legend=False,
        showfliers=False, ax=ax,
    )
    ax.set_title("Per-Meter Consumption by System Demand Status")
    ax.set_xlabel("System demand status")
    ax.set_ylabel("Consumption per half hour (kWh)")

    return _save(fig, output_dir, "04_demand_status_distribution.png")


def generate_all_charts(df: pd.DataFrame, output_dir: str) -> dict:
    """Generate the full chart set and return a name -> filepath mapping."""
    charts = {
        "daily_trend": plot_daily_trend(df, output_dir),
        "consumption_by_region": plot_consumption_by_region(df, output_dir),
        "hourly_load_curve": plot_hourly_load_curve(df, output_dir),
        "demand_status_distribution": plot_demand_status_distribution(df, output_dir),
    }
    print(f"[EDA] {len(charts)} charts saved to '{output_dir}'")
    return charts
