"""
02_forecasting.py
------------------
Builds a per-product demand forecast and compares two approaches:

  1. Baseline  : 7-day moving average (naive but standard industry baseline)
  2. ML model  : Random Forest Regressor using lag + rolling + calendar features

Features used by the ML model:
  lag_1, lag_7, lag_14, lag_28
  rolling_mean_7, rolling_mean_14, rolling_mean_28
  day_of_week, month, promotion

Train/test split: last 60 days per product held out as the test set
(time-based split -- no shuffling, no leakage).

Metrics: MAE, RMSE, MAPE, computed per product and averaged overall.

Output:
  - outputs/forecast_results.csv   (actual vs baseline vs ML predictions, test period)
  - outputs/model_comparison.csv   (per-product + overall metrics for both models)
  - Printed summary with the ACTUAL improvement % (not hardcoded)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA_PATH = Path(__file__).parent / "data" / "sales_data.csv"
OUT_DIR = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)

TEST_DAYS = 60
FEATURES = [
    "lag_1", "lag_7", "lag_14", "lag_28",
    "rolling_mean_7", "rolling_mean_14", "rolling_mean_28",
    "day_of_week", "month", "promotion",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["product_id", "date"]).copy()
    g = df.groupby("product_id")["demand"]

    df["lag_1"] = g.shift(1)
    df["lag_7"] = g.shift(7)
    df["lag_14"] = g.shift(14)
    df["lag_28"] = g.shift(28)

    df["rolling_mean_7"] = g.shift(1).rolling(7).mean().reset_index(level=0, drop=True)
    df["rolling_mean_14"] = g.shift(1).rolling(14).mean().reset_index(level=0, drop=True)
    df["rolling_mean_28"] = g.shift(1).rolling(28).mean().reset_index(level=0, drop=True)

    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month

    # baseline forecast = trailing 7-day moving average (using data up to t-1, no leakage)
    df["baseline_pred"] = df["rolling_mean_7"]

    df = df.dropna(subset=FEATURES + ["baseline_pred"]).reset_index(drop=True)
    return df


def mape(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def train_and_evaluate(df: pd.DataFrame):
    results_rows = []
    metrics_rows = []

    for product_id, group in df.groupby("product_id"):
        group = group.sort_values("date").reset_index(drop=True)
        split_idx = len(group) - TEST_DAYS
        train, test = group.iloc[:split_idx], group.iloc[split_idx:]

        # ---- Baseline: trailing 7-day moving average ----
        baseline_pred = test["baseline_pred"].values

        # ---- ML model: Random Forest ----
        model = RandomForestRegressor(
            n_estimators=300, max_depth=8, min_samples_leaf=3,
            random_state=42, n_jobs=-1,
        )
        model.fit(train[FEATURES], train["demand"])
        ml_pred = model.predict(test[FEATURES])
        ml_pred = np.clip(ml_pred, 0, None)

        actual = test["demand"].values

        # ---- Metrics ----
        base_mae = mean_absolute_error(actual, baseline_pred)
        base_rmse = np.sqrt(mean_squared_error(actual, baseline_pred))
        base_mape = mape(actual, baseline_pred)

        ml_mae = mean_absolute_error(actual, ml_pred)
        ml_rmse = np.sqrt(mean_squared_error(actual, ml_pred))
        ml_mape = mape(actual, ml_pred)

        metrics_rows.append({
            "product_id": product_id,
            "product_name": test["product_name"].iloc[0],
            "baseline_MAE": base_mae, "ml_MAE": ml_mae,
            "baseline_RMSE": base_rmse, "ml_RMSE": ml_rmse,
            "baseline_MAPE": base_mape, "ml_MAPE": ml_mape,
        })

        for d, a, b, m in zip(test["date"], actual, baseline_pred, ml_pred):
            results_rows.append({
                "date": d, "product_id": product_id,
                "actual": a, "baseline_pred": b, "ml_pred": m,
            })

    results_df = pd.DataFrame(results_rows)
    metrics_df = pd.DataFrame(metrics_rows)

    metrics_df["mae_improvement_pct"] = (
        (metrics_df["baseline_MAE"] - metrics_df["ml_MAE"]) / metrics_df["baseline_MAE"] * 100
    )

    return results_df, metrics_df


def main():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = build_features(df)
    results_df, metrics_df = train_and_evaluate(df)

    results_df.to_csv(OUT_DIR / "forecast_results.csv", index=False)
    metrics_df.round(3).to_csv(OUT_DIR / "model_comparison.csv", index=False)

    print("=" * 70)
    print("PER-PRODUCT MODEL COMPARISON (test = last 60 days per product)")
    print("=" * 70)
    print(metrics_df.round(2).to_string(index=False))

    overall_base_mae = metrics_df["baseline_MAE"].mean()
    overall_ml_mae = metrics_df["ml_MAE"].mean()
    overall_improvement = (overall_base_mae - overall_ml_mae) / overall_base_mae * 100

    overall_base_mape = metrics_df["baseline_MAPE"].mean()
    overall_ml_mape = metrics_df["ml_MAPE"].mean()

    print("\n" + "=" * 70)
    print("OVERALL RESULT")
    print("=" * 70)
    print(f"Baseline avg MAE : {overall_base_mae:.2f} units | avg MAPE: {overall_base_mape:.2f}%")
    print(f"ML model avg MAE : {overall_ml_mae:.2f} units | avg MAPE: {overall_ml_mape:.2f}%")
    print(f"\n>>> ACTUAL forecast accuracy improvement (MAE basis): {overall_improvement:.1f}% <<<")
    print("\nUse this real number on your resume/interview -- not a hardcoded guess.")
    print(f"\nSaved -> {OUT_DIR / 'forecast_results.csv'}")
    print(f"Saved -> {OUT_DIR / 'model_comparison.csv'}")


if __name__ == "__main__":
    main()
