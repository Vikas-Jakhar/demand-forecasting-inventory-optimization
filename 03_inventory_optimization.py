"""
03_inventory_optimization.py
------------------------------
Takes the forecast outputs from 02_forecasting.py and turns them into an
actual inventory policy, then simulates it day-by-day to measure stockouts.

Steps:
  1. EOQ (Economic Order Quantity)   = sqrt(2 * D * S / H)
  2. Safety Stock                    = z * sigma_demand * sqrt(lead_time)
  3. Reorder Point (ROP)             = (avg_daily_demand * lead_time) + safety_stock
  4. Inventory simulation ("Before"): reorder policy driven by the BASELINE
     forecast (7-day moving average)
  5. Inventory simulation ("After") : reorder policy driven by the ML forecast
  6. Compare stockout days / stockout units / service level between the two

Assumptions (clearly stated so they're defensible in an interview):
  - Ordering cost (S)      = Rs. 500 per order
  - Holding cost (H)       = 20% of unit cost per year
  - Unit cost              = Rs. 300 (assumed, editable below)
  - Lead time              = 5 days
  - Service level target   = 95% -> z = 1.645
  - Starting inventory     = 20 days of average demand

Output:
  - outputs/inventory_policy.csv     (EOQ, safety stock, ROP per product)
  - outputs/simulation_before.csv    (day-by-day sim using baseline forecast)
  - outputs/simulation_after.csv     (day-by-day sim using ML forecast)
  - outputs/stockout_comparison.csv  (before vs after KPIs per product)
  - Printed summary with ACTUAL stockout reduction % (not hardcoded)
"""

import numpy as np
import pandas as pd
from pathlib import Path

DATA_PATH = Path(__file__).parent / "data" / "sales_data.csv"
FORECAST_PATH = Path(__file__).parent / "outputs" / "forecast_results.csv"
OUT_DIR = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)

# ---- Business assumptions (edit these to match a real scenario) ----
ORDERING_COST = 500        # Rs per order (S)
HOLDING_COST_PCT = 0.20    # % of unit cost held per year (H = pct * unit_cost)
UNIT_COST = 300            # Rs per unit
LEAD_TIME_DAYS = 5
SERVICE_LEVEL_Z = 1.645    # 95% service level
STARTING_INVENTORY_DAYS = 20


def compute_policy(sales_df: pd.DataFrame) -> pd.DataFrame:
    """EOQ, Safety Stock, Reorder Point per product using full-year historical stats."""
    stats = sales_df.groupby("product_id").agg(
        product_name=("product_name", "first"),
        avg_daily_demand=("demand", "mean"),
        std_daily_demand=("demand", "std"),
    ).reset_index()

    annual_demand = stats["avg_daily_demand"] * 365
    holding_cost_per_unit = HOLDING_COST_PCT * UNIT_COST

    stats["annual_demand_D"] = annual_demand.round(0)
    stats["EOQ"] = np.sqrt(2 * annual_demand * ORDERING_COST / holding_cost_per_unit).round(0)
    stats["safety_stock"] = (
        SERVICE_LEVEL_Z * stats["std_daily_demand"] * np.sqrt(LEAD_TIME_DAYS)
    ).round(0)
    stats["reorder_point"] = (
        stats["avg_daily_demand"] * LEAD_TIME_DAYS + stats["safety_stock"]
    ).round(0)
    stats["starting_inventory"] = (
        stats["avg_daily_demand"] * STARTING_INVENTORY_DAYS
    ).round(0)

    return stats


def simulate(product_id, actual_demand, forecast_demand, policy_row):
    """
    Simple periodic-review (s, Q) simulation:
      - Every day, subtract actual demand from inventory (stockout if inventory < demand)
      - If inventory position (on-hand + on-order) <= reorder point, place an EOQ order
      - Orders arrive after LEAD_TIME_DAYS
      - Forecast is used only to decide WHEN inventory looks low relative to expected
        near-term usage (forecast-informed reorder trigger), actual demand is what
        depletes stock.
    """
    inventory = policy_row["starting_inventory"]
    rop = policy_row["reorder_point"]
    eoq = policy_row["EOQ"]

    on_order = []  # list of (arrival_day_index, qty)
    records = []
    stockout_days = 0
    stockout_units = 0

    for day_idx in range(len(actual_demand)):
        # receive any arriving orders
        arriving = [qty for arr_day, qty in on_order if arr_day == day_idx]
        inventory += sum(arriving)
        on_order = [(arr_day, qty) for arr_day, qty in on_order if arr_day != day_idx]

        demand_today = actual_demand[day_idx]

        if inventory < demand_today:
            stockout_units += (demand_today - inventory)
            stockout_days += 1
            inventory = 0
        else:
            inventory -= demand_today

        # inventory position = on-hand + on-order
        inventory_position = inventory + sum(qty for _, qty in on_order)

        # forecast-informed trigger: expected demand over the lead time
        expected_near_term = forecast_demand[day_idx] * LEAD_TIME_DAYS if day_idx < len(forecast_demand) else policy_row["avg_daily_demand"] * LEAD_TIME_DAYS

        if inventory_position <= max(rop, expected_near_term):
            on_order.append((day_idx + LEAD_TIME_DAYS, eoq))

        records.append({
            "product_id": product_id,
            "day": day_idx,
            "demand": demand_today,
            "inventory_end_of_day": inventory,
            "stockout": int(inventory == 0 and demand_today > 0 and stockout_units > 0 and day_idx == day_idx),
        })

    return records, stockout_days, stockout_units


def main():
    sales_df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    forecast_df = pd.read_csv(FORECAST_PATH, parse_dates=["date"])

    policy_df = compute_policy(sales_df)
    policy_df.round(2).to_csv(OUT_DIR / "inventory_policy.csv", index=False)

    print("=" * 70)
    print("INVENTORY POLICY (EOQ / Safety Stock / Reorder Point)")
    print("=" * 70)
    print(policy_df[["product_id", "product_name", "avg_daily_demand", "EOQ",
                      "safety_stock", "reorder_point"]].round(1).to_string(index=False))

    comparison_rows = []
    all_before, all_after = [], []

    for product_id, policy_row in policy_df.set_index("product_id").iterrows():
        prod_forecast = forecast_df[forecast_df["product_id"] == product_id].sort_values("date")
        if prod_forecast.empty:
            continue

        actual = prod_forecast["actual"].values
        baseline_pred = prod_forecast["baseline_pred"].values
        ml_pred = prod_forecast["ml_pred"].values

        before_records, before_stockout_days, before_stockout_units = simulate(
            product_id, actual, baseline_pred, policy_row
        )
        after_records, after_stockout_days, after_stockout_units = simulate(
            product_id, actual, ml_pred, policy_row
        )

        all_before.extend(before_records)
        all_after.extend(after_records)

        n_days = len(actual)
        comparison_rows.append({
            "product_id": product_id,
            "product_name": policy_row["product_name"],
            "sim_days": n_days,
            "stockout_days_before": before_stockout_days,
            "stockout_days_after": after_stockout_days,
            "stockout_units_before": before_stockout_units,
            "stockout_units_after": after_stockout_units,
            "service_level_before_pct": round((1 - before_stockout_days / n_days) * 100, 2),
            "service_level_after_pct": round((1 - after_stockout_days / n_days) * 100, 2),
        })

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df["stockout_day_reduction_pct"] = (
        (comparison_df["stockout_days_before"] - comparison_df["stockout_days_after"])
        / comparison_df["stockout_days_before"].replace(0, np.nan) * 100
    ).fillna(0).round(1)

    pd.DataFrame(all_before).to_csv(OUT_DIR / "simulation_before.csv", index=False)
    pd.DataFrame(all_after).to_csv(OUT_DIR / "simulation_after.csv", index=False)
    comparison_df.to_csv(OUT_DIR / "stockout_comparison.csv", index=False)

    print("\n" + "=" * 70)
    print("STOCKOUT COMPARISON: BASELINE-DRIVEN vs ML-DRIVEN REORDER POLICY")
    print("=" * 70)
    print(comparison_df.to_string(index=False))

    total_before = comparison_df["stockout_days_before"].sum()
    total_after = comparison_df["stockout_days_after"].sum()
    overall_reduction = (total_before - total_after) / total_before * 100 if total_before else 0

    print("\n" + "=" * 70)
    print("OVERALL RESULT")
    print("=" * 70)
    print(f"Total stockout-days (baseline policy): {total_before}")
    print(f"Total stockout-days (ML policy)      : {total_after}")
    print(f"\n>>> ACTUAL stockout reduction: {overall_reduction:.1f}% <<<")
    print("\nUse this real number on your resume/interview -- not a hardcoded guess.")


if __name__ == "__main__":
    main()
