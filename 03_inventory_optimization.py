import numpy as np
import pandas as pd
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

DATA_PATH = Path(__file__).parent / "data" / "sales_data.csv"
FORECAST_PATH = Path(__file__).parent / "outputs" / "forecast_results.csv"
OUT_DIR = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)


# ============================================================
# BUSINESS ASSUMPTIONS
# ============================================================

ORDERING_COST = 500
HOLDING_COST_PCT = 0.20
UNIT_COST = 300

LEAD_TIME_DAYS = 5
SERVICE_LEVEL_Z = 1.645

# Same starting inventory for BOTH policies
STARTING_INVENTORY_DAYS = 7


# ============================================================
# INVENTORY POLICY CALCULATION
# ============================================================

def compute_policy(sales_df):

    stats = sales_df.groupby("product_id").agg(
        product_name=("product_name", "first"),
        avg_daily_demand=("demand", "mean"),
        std_daily_demand=("demand", "std")
    ).reset_index()

    holding_cost_per_unit = HOLDING_COST_PCT * UNIT_COST

    # Annual demand for EOQ
    stats["annual_demand"] = stats["avg_daily_demand"] * 365

    # EOQ
    stats["EOQ"] = np.sqrt(
        2
        * stats["annual_demand"]
        * ORDERING_COST
        / holding_cost_per_unit
    )

    # Safety Stock
    stats["safety_stock"] = (
        SERVICE_LEVEL_Z
        * stats["std_daily_demand"]
        * np.sqrt(LEAD_TIME_DAYS)
    )

    # Traditional fixed ROP
    stats["reorder_point"] = (
        stats["avg_daily_demand"] * LEAD_TIME_DAYS
        + stats["safety_stock"]
    )

    # Same starting inventory
    stats["starting_inventory"] = (
        stats["avg_daily_demand"]
        * STARTING_INVENTORY_DAYS
    )

    return stats


# ============================================================
# SIMULATION
# ============================================================

def simulate(
    product_id,
    actual_demand,
    forecast_demand,
    policy_row,
    policy_type
):

    """
    Simulates two inventory strategies.

    BEFORE:
        Traditional fixed reorder point:
        ROP = historical average demand × lead time + safety stock

        Replenishment quantity = EOQ

    AFTER:
        Forecast-informed dynamic reorder point:

        Dynamic ROP =
        forecasted demand during lead time + safety stock

        Replenishment quantity = EOQ

    Both policies:
        - Same starting inventory
        - Same EOQ
        - Same safety stock
        - Same lead time
        - Same actual demand

    Therefore the only major difference is how demand
    is estimated for replenishment timing.
    """

    inventory = float(policy_row["starting_inventory"])
    eoq = float(policy_row["EOQ"])
    fixed_rop = float(policy_row["reorder_point"])
    safety_stock = float(policy_row["safety_stock"])
    avg_daily_demand = float(policy_row["avg_daily_demand"])

    on_order = []

    records = []

    stockout_days = 0
    stockout_units = 0


    for day_idx in range(len(actual_demand)):

        # ====================================================
        # 1. RECEIVE ARRIVING ORDERS
        # ====================================================

        arriving_qty = sum(
            qty
            for arrival_day, qty in on_order
            if arrival_day == day_idx
        )

        inventory += arriving_qty

        on_order = [
            (arrival_day, qty)
            for arrival_day, qty in on_order
            if arrival_day != day_idx
        ]


        # ====================================================
        # 2. ACTUAL DEMAND CONSUMES INVENTORY
        # ====================================================

        demand_today = float(actual_demand[day_idx])

        stockout_today = 0
        shortage = 0

        if inventory < demand_today:

            shortage = demand_today - inventory

            stockout_units += shortage
            stockout_days += 1
            stockout_today = 1

            inventory = 0

        else:

            inventory -= demand_today


        # ====================================================
        # 3. INVENTORY POSITION
        # ====================================================

        inventory_position = (
            inventory
            + sum(qty for _, qty in on_order)
        )


        # ====================================================
        # 4. DETERMINE REORDER POINT
        # ====================================================

        if policy_type == "baseline":

            # Traditional inventory policy:
            # Uses historical average demand
            reorder_trigger = fixed_rop


        elif policy_type == "ml":

            # ML-driven policy:
            # Uses forecast for the NEXT lead-time window.
            future_forecast = forecast_demand[
                day_idx:
                min(
                    day_idx + LEAD_TIME_DAYS,
                    len(forecast_demand)
                )
            ]

            if len(future_forecast) > 0:

                forecast_lead_time_demand = float(
                    np.sum(
                        np.maximum(future_forecast, 0)
                    )
                )

            else:

                forecast_lead_time_demand = (
                    avg_daily_demand
                    * LEAD_TIME_DAYS
                )

            reorder_trigger = (
                forecast_lead_time_demand
                + safety_stock
            )

        else:

            raise ValueError(
                "policy_type must be 'baseline' or 'ml'"
            )


        # ====================================================
        # 5. PLACE ORDER
        # ====================================================

        # Avoid repeatedly placing multiple EOQ orders
        # while another order is already in transit.
        if (
            inventory_position <= reorder_trigger
            and len(on_order) == 0
        ):

            on_order.append(
                (
                    day_idx + LEAD_TIME_DAYS,
                    eoq
                )
            )


        # ====================================================
        # 6. SAVE DAILY RECORD
        # ====================================================

        records.append({

            "product_id": product_id,

            "day": day_idx,

            "actual_demand": demand_today,

            "inventory_end_of_day": inventory,

            "inventory_position": inventory_position,

            "reorder_trigger": reorder_trigger,

            "stockout": stockout_today,

            "shortage_units": shortage

        })


    # ========================================================
    # RETURN RESULTS
    # ========================================================

    return (
        records,
        stockout_days,
        stockout_units
    )


# ============================================================
# MAIN
# ============================================================

def main():

    sales_df = pd.read_csv(
        DATA_PATH,
        parse_dates=["date"]
    )

    forecast_df = pd.read_csv(
        FORECAST_PATH,
        parse_dates=["date"]
    )


    # ========================================================
    # CALCULATE INVENTORY POLICY
    # ========================================================

    policy_df = compute_policy(sales_df)

    policy_df.round(2).to_csv(
        OUT_DIR / "inventory_policy.csv",
        index=False
    )

    print("=" * 70)
    print("INVENTORY POLICY (EOQ / SAFETY STOCK / REORDER POINT)")
    print("=" * 70)

    print(
        policy_df[
            [
                "product_id",
                "product_name",
                "avg_daily_demand",
                "EOQ",
                "safety_stock",
                "reorder_point"
            ]
        ].round(1).to_string(index=False)
    )


    # ========================================================
    # RUN SIMULATIONS
    # ========================================================

    comparison_rows = []

    all_before = []
    all_after = []


    for product_id, policy_row in (
        policy_df.set_index("product_id").iterrows()
    ):

        product_data = forecast_df[
            forecast_df["product_id"] == product_id
        ].sort_values("date")

        if product_data.empty:
            continue


        actual = product_data["actual"].values

        ml_forecast = product_data["ml_pred"].values


        # ----------------------------------------------------
        # BEFORE: Traditional fixed ROP
        # ----------------------------------------------------

        (
            before_records,
            before_stockout_days,
            before_stockout_units
        ) = simulate(
            product_id=product_id,
            actual_demand=actual,
            forecast_demand=ml_forecast,
            policy_row=policy_row,
            policy_type="baseline"
        )


        # ----------------------------------------------------
        # AFTER: ML forecast-informed dynamic ROP
        # ----------------------------------------------------

        (
            after_records,
            after_stockout_days,
            after_stockout_units
        ) = simulate(
            product_id=product_id,
            actual_demand=actual,
            forecast_demand=ml_forecast,
            policy_row=policy_row,
            policy_type="ml"
        )


        all_before.extend(before_records)
        all_after.extend(after_records)


        n_days = len(actual)


        comparison_rows.append({

            "product_id": product_id,

            "product_name": policy_row["product_name"],

            "sim_days": n_days,

            "stockout_days_before":
                before_stockout_days,

            "stockout_days_after":
                after_stockout_days,

            "stockout_units_before":
                before_stockout_units,

            "stockout_units_after":
                after_stockout_units,

            "service_level_before_pct":
                round(
                    (
                        1
                        - before_stockout_days / n_days
                    ) * 100,
                    2
                ),

            "service_level_after_pct":
                round(
                    (
                        1
                        - after_stockout_days / n_days
                    ) * 100,
                    2
                )

        })


    # ========================================================
    # CREATE COMPARISON
    # ========================================================

    comparison_df = pd.DataFrame(
        comparison_rows
    )


    comparison_df[
        "stockout_day_reduction_pct"
    ] = (

        (
            comparison_df[
                "stockout_days_before"
            ]
            -
            comparison_df[
                "stockout_days_after"
            ]
        )

        /

        comparison_df[
            "stockout_days_before"
        ].replace(0, np.nan)

        * 100

    ).fillna(0).round(1)


    # ========================================================
    # SAVE FILES
    # ========================================================

    pd.DataFrame(all_before).to_csv(
        OUT_DIR / "simulation_before.csv",
        index=False
    )

    pd.DataFrame(all_after).to_csv(
        OUT_DIR / "simulation_after.csv",
        index=False
    )

    comparison_df.to_csv(
        OUT_DIR / "stockout_comparison.csv",
        index=False
    )


    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print("\n" + "=" * 70)
    print("STOCKOUT COMPARISON: TRADITIONAL vs ML-DRIVEN POLICY")
    print("=" * 70)

    print(
        comparison_df.to_string(index=False)
    )


    # ========================================================
    # OVERALL RESULTS
    # ========================================================

    total_before = comparison_df[
        "stockout_days_before"
    ].sum()

    total_after = comparison_df[
        "stockout_days_after"
    ].sum()


    if total_before > 0:

        overall_reduction = (
            (
                total_before
                - total_after
            )
            /
            total_before
            * 100
        )

    else:

        overall_reduction = 0


    print("\n" + "=" * 70)
    print("OVERALL RESULT")
    print("=" * 70)

    print(
        f"Total stockout-days (Traditional): "
        f"{total_before}"
    )

    print(
        f"Total stockout-days (ML-driven)  : "
        f"{total_after}"
    )

    print()

    print(
        f">>> ACTUAL stockout reduction: "
        f"{overall_reduction:.1f}% <<<"
    )

    print("\nSaved outputs to:")
    print(OUT_DIR)


if __name__ == "__main__":
    main()