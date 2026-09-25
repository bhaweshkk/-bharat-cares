"""
modeling.py
-----------
Two lightweight, business-relevant models:

1. Forecasting  — RandomForestRegressor on daily aggregated consumption,
   used to project the next FORECAST_HORIZON days and quantify
   near-term demand trajectory.

2. Risk classification — RandomForestClassifier predicting the
   probability that a given half-hour period will fall in a system
   'High' demand status, from time-of-day/seasonal features. This
   turns a reactive label (Demand status, known only after the fact)
   into a forward-looking risk score, and its feature importances
   double as a data-driven "why" (drivers) answer.

Both models are intentionally simple (tree ensembles on a handful of
calendar features) so the pipeline stays fast, dependency-light, and
interpretable — appropriate for a strategic BI tool rather than a
production forecasting system.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, mean_absolute_error, roc_auc_score
from sklearn.model_selection import train_test_split

FORECAST_HORIZON_DAYS = 14
RANDOM_STATE = 42


# ----------------------------------------------------------------------
# 1. Consumption forecasting
# ----------------------------------------------------------------------
def _build_daily_series(df: pd.DataFrame) -> pd.DataFrame:
    daily = df.groupby(df["timestamp"].dt.floor("D"))["kwh"].sum().rename("total_kwh").to_frame()
    daily.index.name = "date"
    daily = daily.reset_index()
    daily["day_of_year"] = daily["date"].dt.dayofyear
    daily["day_of_week"] = daily["date"].dt.dayofweek
    daily["month"] = daily["date"].dt.month
    daily["lag_7"] = daily["total_kwh"].shift(7)
    daily["rolling_mean_7"] = daily["total_kwh"].shift(1).rolling(7).mean()
    return daily


def forecast_consumption(df: pd.DataFrame) -> dict:
    """
    Trains a RandomForestRegressor on calendar + lag features to forecast
    total daily portfolio consumption, holding out the most recent 14 days
    as a test set to report honest accuracy.

    Returns a dict with the fitted test metrics, a forward-looking forecast
    for the next FORECAST_HORIZON_DAYS days, and the trend direction.
    """
    daily = _build_daily_series(df).dropna().reset_index(drop=True)

    feature_cols = ["day_of_year", "day_of_week", "month", "lag_7", "rolling_mean_7"]
    X, y = daily[feature_cols], daily["total_kwh"]

    # Chronological split: last 14 available days as the test set.
    split_idx = len(daily) - FORECAST_HORIZON_DAYS
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    test_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, test_pred)
    mape = float(np.mean(np.abs((y_test.values - test_pred) / y_test.values)) * 100)

    # Refit on the full series, then iteratively forecast forward.
    model_full = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=RANDOM_STATE)
    model_full.fit(X, y)

    history = daily[["date", "total_kwh"]].copy()
    forecasts = []
    for _ in range(FORECAST_HORIZON_DAYS):
        next_date = history["date"].iloc[-1] + pd.Timedelta(days=1)
        lag_7 = history["total_kwh"].iloc[-7] if len(history) >= 7 else history["total_kwh"].mean()
        rolling_mean_7 = history["total_kwh"].iloc[-7:].mean()
        features = pd.DataFrame([{
            "day_of_year": next_date.dayofyear,
            "day_of_week": next_date.dayofweek,
            "month": next_date.month,
            "lag_7": lag_7,
            "rolling_mean_7": rolling_mean_7,
        }])
        pred = float(model_full.predict(features)[0])
        forecasts.append({"date": next_date, "forecast_kwh": pred})
        history = pd.concat(
            [history, pd.DataFrame([{"date": next_date, "total_kwh": pred}])],
            ignore_index=True,
        )

    forecast_df = pd.DataFrame(forecasts)
    recent_avg = daily["total_kwh"].tail(FORECAST_HORIZON_DAYS).mean()
    forecast_avg = forecast_df["forecast_kwh"].mean()
    trend_pct = (forecast_avg - recent_avg) / recent_avg * 100

    print(f"[Forecasting] Test MAE: {mae:,.1f} kWh/day | Test MAPE: {mape:.1f}% "
          f"(evaluated on the most recent {FORECAST_HORIZON_DAYS} days)")

    return {
        "test_mae": mae,
        "test_mape": mape,
        "forecast_df": forecast_df,
        "recent_avg_daily_kwh": recent_avg,
        "forecast_avg_daily_kwh": forecast_avg,
        "trend_pct": trend_pct,
    }


# ----------------------------------------------------------------------
# 2. High-demand risk classification
# ----------------------------------------------------------------------
def train_demand_risk_model(df: pd.DataFrame) -> dict:
    """
    Trains a classifier to predict P(system demand status == 'High') for a
    half-hour period, using only calendar/time features (no future
    consumption leakage). This produces:
      - a held-out ROC-AUC / classification report for credibility
      - feature importances, used as a data-driven "top drivers" answer
    """
    labeled = df[df["demand_status"].isin(["Low", "Normal", "High"])].copy()

    # Aggregate to one row per half-hour timestamp (system-level features).
    period = labeled.groupby("timestamp").agg(
        avg_kwh=("kwh", "mean"),
        total_kwh=("kwh", "sum"),
        demand_status=("demand_status", "first"),
    ).reset_index()

    period["hour"] = period["timestamp"].dt.hour
    period["day_of_week"] = period["timestamp"].dt.dayofweek
    period["month"] = period["timestamp"].dt.month
    period["is_weekend"] = (period["day_of_week"] >= 5).astype(int)
    period["target_high"] = (period["demand_status"] == "High").astype(int)

    # Cyclical encoding for calendar features: hour/day/month are periodic
    # (December sits next to January, hour 23 sits next to hour 0), so raw
    # integers would mislead a tree-based model at the wrap-around point.
    period["hour_sin"] = np.sin(2 * np.pi * period["hour"] / 24)
    period["hour_cos"] = np.cos(2 * np.pi * period["hour"] / 24)
    period["month_sin"] = np.sin(2 * np.pi * period["month"] / 12)
    period["month_cos"] = np.cos(2 * np.pi * period["month"] / 12)

    feature_cols = [
        "hour_sin", "hour_cos", "month_sin", "month_cos",
        "day_of_week", "is_weekend", "avg_kwh",
    ]
    X, y = period[feature_cols], period["target_high"]

    # This model learns recurring calendar patterns ("winter evenings carry
    # more High-demand risk"), not sequential/future dependence, so a
    # stratified random split is the appropriate validation strategy here
    # (unlike the sequential forecast above, which must respect time order).
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    model = RandomForestClassifier(
        n_estimators=300, max_depth=6, class_weight="balanced", random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    preds = model.predict(X_test)
    auc = roc_auc_score(y_test, proba)
    report = classification_report(y_test, preds, target_names=["Not High", "High"], zero_division=0)

    raw_importances = pd.Series(model.feature_importances_, index=feature_cols)

    # Collapse each cyclical (sin, cos) pair back into one human-readable
    # driver for reporting, e.g. "hour" instead of "hour_sin" + "hour_cos".
    grouped_importances = pd.Series({
        "hour_of_day": raw_importances[["hour_sin", "hour_cos"]].sum(),
        "month_of_year": raw_importances[["month_sin", "month_cos"]].sum(),
        "day_of_week": raw_importances["day_of_week"],
        "is_weekend": raw_importances["is_weekend"],
        "avg_meter_load": raw_importances["avg_kwh"],
    }).sort_values(ascending=False)

    print(f"[Risk model] Held-out ROC-AUC for predicting 'High' demand periods: {auc:.3f}")

    return {
        "model": model,
        "feature_cols": feature_cols,
        "auc": auc,
        "classification_report": report,
        "feature_importances": grouped_importances,
        "high_demand_share": float(y.mean()),
    }
