"""
main.py
-------
Orchestrates the full Energy BI pipeline end to end:

    Data -> Information -> Insights -> Decision -> Action

    1. Ingestion & Cleaning   (src/data_ingestion.py)
    2. EDA & Visualization    (src/eda.py)
    3. Predictive Modeling    (src/modeling.py)
    4. Executive Dashboard    (src/insights.py)

Run with:  python main.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_ingestion import run_ingestion          # noqa: E402
from eda import generate_all_charts               # noqa: E402
from modeling import forecast_consumption, train_demand_risk_model  # noqa: E402
from insights import build_executive_summary, compute_kpis          # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
CHARTS_DIR = os.path.join(OUTPUT_DIR, "charts")
SUMMARY_PATH = os.path.join(OUTPUT_DIR, "executive_summary.txt")


def main():
    print("\n" + "#" * 78)
    print("# ENERGY & DEMAND BUSINESS INTELLIGENCE PIPELINE")
    print("#" * 78 + "\n")

    # 1. Data Ingestion & Cleaning
    print("--- Step 1/4: Data Ingestion & Cleaning ---")
    df = run_ingestion(DATA_DIR)

    # 2. Exploratory Data Analysis & Visualizations
    print("\n--- Step 2/4: Exploratory Data Analysis ---")
    generate_all_charts(df, CHARTS_DIR)

    # 3. Predictive Modeling
    print("\n--- Step 3/4: Predictive Modeling ---")
    forecast = forecast_consumption(df)
    risk = train_demand_risk_model(df)

    # 4. Executive Dashboard / Insight Summary
    print("\n--- Step 4/4: Executive Insight Summary ---")
    kpis = compute_kpis(df)
    summary = build_executive_summary(kpis, forecast, risk)

    print("\n" + summary)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(SUMMARY_PATH, "w") as f:
        f.write(summary)
    print(f"\n[Done] Executive summary saved to '{SUMMARY_PATH}'")
    print(f"[Done] Charts saved to '{CHARTS_DIR}'")


if __name__ == "__main__":
    main()
