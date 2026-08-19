import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Demand Forecasting & Inventory Optimization",
    page_icon="📦",
    layout="wide"
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    sales = pd.read_csv(
        DATA_DIR / "sales_data.csv",
        parse_dates=["date"]
    )

    forecast = pd.read_csv(
        OUTPUT_DIR / "forecast_results.csv",
        parse_dates=["date"]
    )

    inventory = pd.read_csv(
        OUTPUT_DIR / "inventory_policy.csv"
    )

    model_comparison = pd.read_csv(
        OUTPUT_DIR / "model_comparison.csv"
    )

    stockout = pd.read_csv(
        OUTPUT_DIR / "stockout_comparison.csv"
    )

    return sales, forecast, inventory, model_comparison, stockout


try:
    sales, forecast, inventory, model_comparison, stockout = load_data()

except FileNotFoundError as e:

    st.error(
        f"Required output file not found: {e}"
    )

    st.info(
        "Run the following scripts first:\n\n"
        "1. python 02_forecasting.py\n"
        "2. python 03_inventory_optimization.py"
    )

    st.stop()


# ============================================================
# TITLE
# ============================================================

st.title("📦 Demand Forecasting & Inventory Optimization")

st.markdown(
    """
    ### Machine Learning-Based Supply Chain Analytics

    This dashboard analyzes historical product demand, compares a traditional
    **7-day Moving Average** against a **Random Forest forecasting model**, and
    uses the forecasts to support inventory decisions using:

    - Economic Order Quantity (EOQ)
    - Safety Stock
    - Reorder Point (ROP)
    - Traditional vs ML-driven inventory policy simulation
    """
)


st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Dashboard Controls")

products = sorted(
    sales["product_id"].unique()
)

selected_product = st.sidebar.selectbox(
    "Select Product",
    products
)


# ============================================================
# FILTER SELECTED PRODUCT
# ============================================================

product_sales = sales[
    sales["product_id"] == selected_product
].copy()

product_forecast = forecast[
    forecast["product_id"] == selected_product
].copy()

product_inventory = inventory[
    inventory["product_id"] == selected_product
].copy()

product_model = model_comparison[
    model_comparison["product_id"] == selected_product
].copy()

product_stockout = stockout[
    stockout["product_id"] == selected_product
].copy()


# ============================================================
# PRODUCT NAME
# ============================================================

if (
    "product_name" in product_sales.columns
    and not product_sales.empty
):

    product_name = product_sales["product_name"].iloc[0]

else:

    product_name = selected_product


st.header(f"📦 {product_name} ({selected_product})")


# ============================================================
# EXTRACT KPI VALUES
# ============================================================

avg_demand = product_sales["demand"].mean()


# ---------- MODEL METRICS ----------

if not product_model.empty:

    baseline_mae = product_model["baseline_MAE"].iloc[0]
    ml_mae = product_model["ml_MAE"].iloc[0]

    mae_improvement = product_model[
        "mae_improvement_pct"
    ].iloc[0]

else:

    baseline_mae = 0
    ml_mae = 0
    mae_improvement = 0


# ---------- INVENTORY METRICS ----------

if not product_inventory.empty:

    eoq = product_inventory["EOQ"].iloc[0]

    safety_stock = product_inventory[
        "safety_stock"
    ].iloc[0]

    reorder_point = product_inventory[
        "reorder_point"
    ].iloc[0]

else:

    eoq = 0
    safety_stock = 0
    reorder_point = 0


# ============================================================
# TOP KPI CARDS
# ============================================================

st.subheader("📊 Product Performance Summary")

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Average Daily Demand",
        f"{avg_demand:.1f} units"
    )


with col2:

    st.metric(
        "Baseline MAE",
        f"{baseline_mae:.2f}"
    )


with col3:

    st.metric(
        "ML Forecast MAE",
        f"{ml_mae:.2f}",
        delta=f"{mae_improvement:.1f}% improvement",
        delta_color="normal"
    )


with col4:

    st.metric(
        "Reorder Point",
        f"{reorder_point:.0f} units"
    )


st.divider()


# ============================================================
# INVENTORY METRICS
# ============================================================

st.subheader("📦 Inventory Parameters")

col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "Economic Order Quantity",
        f"{eoq:.0f} units"
    )


with col2:

    st.metric(
        "Safety Stock",
        f"{safety_stock:.0f} units"
    )


with col3:

    st.metric(
        "Reorder Point",
        f"{reorder_point:.0f} units"
    )


st.divider()


# ============================================================
# FULL HISTORICAL DEMAND
# ============================================================

st.subheader("📈 Full Historical Demand")

plot_data = product_sales.sort_values("date")

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    plot_data["date"],
    plot_data["demand"],
    label="Historical Demand"
)

ax.set_xlabel("Date")
ax.set_ylabel("Demand (Units)")
ax.set_title(
    f"Historical Demand — {product_name}"
)

ax.legend()
ax.grid(True, alpha=0.3)

plt.xticks(rotation=45)

plt.tight_layout()

st.pyplot(fig)

plt.close(fig)


# ============================================================
# ACTUAL VS FORECAST CHART
# ============================================================

st.divider()

st.subheader("🔮 Actual vs Forecasted Demand")

if not product_forecast.empty:

    product_forecast["date"] = pd.to_datetime(
        product_forecast["date"]
    )

    plot_forecast = product_forecast.sort_values(
        "date"
    )

    fig, ax = plt.subplots(figsize=(12, 5))


    # Actual demand

    ax.plot(
        plot_forecast["date"],
        plot_forecast["actual"],
        label="Actual Demand",
        marker="o",
        markersize=3
    )


    # Baseline forecast

    ax.plot(
        plot_forecast["date"],
        plot_forecast["baseline_pred"],
        label="7-Day Moving Average"
    )


    # ML forecast

    ax.plot(
        plot_forecast["date"],
        plot_forecast["ml_pred"],
        label="Random Forest Forecast"
    )


    ax.set_xlabel("Date")
    ax.set_ylabel("Demand (Units)")

    ax.set_title(
        f"Forecast Comparison — {product_name}"
    )

    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.xticks(rotation=45)

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)


else:

    st.warning(
        "No forecast data available for this product."
    )


# ============================================================
# FORECAST MODEL COMPARISON
# ============================================================

st.divider()

st.subheader("🤖 Forecast Model Comparison")

if not product_model.empty:

    comparison_display = product_model[
        [
            "baseline_MAE",
            "ml_MAE",
            "baseline_RMSE",
            "ml_RMSE",
            "baseline_MAPE",
            "ml_MAPE",
            "mae_improvement_pct"
        ]
    ].T

    comparison_display.columns = ["Value"]

    comparison_display.index = [
        "Baseline MAE",
        "ML MAE",
        "Baseline RMSE",
        "ML RMSE",
        "Baseline MAPE (%)",
        "ML MAPE (%)",
        "MAE Improvement (%)"
    ]

    st.dataframe(
        comparison_display.round(2),
        use_container_width=True
    )


else:

    st.warning(
        "Model comparison data is not available."
    )


# ============================================================
# INVENTORY POLICY TABLE
# ============================================================

st.divider()

st.subheader("📋 Inventory Policy")

if not product_inventory.empty:

    policy_columns = [
        "avg_daily_demand",
        "annual_demand_D",
        "EOQ",
        "safety_stock",
        "reorder_point"
    ]

    available_policy_columns = [
        col for col in policy_columns
        if col in product_inventory.columns
    ]

    inventory_display = product_inventory[
        available_policy_columns
    ].copy()

    inventory_display = inventory_display.round(2)

    st.dataframe(
        inventory_display,
        use_container_width=True
    )


else:

    st.warning(
        "Inventory policy data is not available."
    )


# ============================================================
# STOCKOUT PERFORMANCE
# ============================================================

st.divider()

st.subheader("⚡ Traditional vs ML-Driven Inventory Performance")

if not product_stockout.empty:

    stockout_cols = [
        "stockout_days_before",
        "stockout_days_after",
        "stockout_units_before",
        "stockout_units_after",
        "service_level_before_pct",
        "service_level_after_pct",
        "stockout_day_reduction_pct"
    ]

    available_cols = [
        col
        for col in stockout_cols
        if col in product_stockout.columns
    ]

    stockout_display = product_stockout[
        available_cols
    ].T

    stockout_display.columns = ["Value"]

    st.dataframe(
        stockout_display.round(2),
        use_container_width=True
    )


    # ---------- PRODUCT STOCKOUT KPIs ----------

    before_days = product_stockout[
        "stockout_days_before"
    ].iloc[0]

    after_days = product_stockout[
        "stockout_days_after"
    ].iloc[0]


    stockout_col1, stockout_col2, stockout_col3 = st.columns(3)


    with stockout_col1:

        st.metric(
            "Traditional Policy Stockout-Days",
            f"{before_days:.0f}"
        )


    with stockout_col2:

        st.metric(
            "ML-Driven Policy Stockout-Days",
            f"{after_days:.0f}"
        )


    with stockout_col3:

        if before_days > 0:

            reduction = (
                (before_days - after_days)
                / before_days
                * 100
            )

        else:

            reduction = 0


        st.metric(
            "Stockout Reduction",
            f"{reduction:.1f}%"
        )


else:

    st.warning(
        "Stockout simulation data is not available."
    )


# ============================================================
# OVERALL PROJECT PERFORMANCE
# ============================================================

st.divider()

st.header("🏆 Overall Project Performance")


# ---------- FORECAST PERFORMANCE ----------

baseline_avg_mae = model_comparison[
    "baseline_MAE"
].mean()

ml_avg_mae = model_comparison[
    "ml_MAE"
].mean()


forecast_improvement = (
    (
        baseline_avg_mae
        - ml_avg_mae
    )
    / baseline_avg_mae
    * 100
)


# ---------- STOCKOUT PERFORMANCE ----------

baseline_stockouts = stockout[
    "stockout_days_before"
].sum()

ml_stockouts = stockout[
    "stockout_days_after"
].sum()


if baseline_stockouts > 0:

    stockout_reduction = (
        (
            baseline_stockouts
            - ml_stockouts
        )
        / baseline_stockouts
        * 100
    )

else:

    stockout_reduction = 0


# ---------- OVERALL KPIs ----------

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Forecast MAE Improvement",
        f"{forecast_improvement:.1f}%"
    )


with col2:

    st.metric(
        "Baseline Stockout-Days",
        f"{baseline_stockouts:.0f}"
    )


with col3:

    st.metric(
        "ML-Driven Stockout-Days",
        f"{ml_stockouts:.0f}"
    )


with col4:

    st.metric(
        "Stockout Reduction",
        f"{stockout_reduction:.1f}%"
    )


# ============================================================
# OVERALL STOCKOUT CHART
# ============================================================

st.subheader("📊 Stockout Days: Traditional vs ML-Driven Policy")

chart_data = stockout[
    [
        "product_id",
        "stockout_days_before",
        "stockout_days_after"
    ]
].copy()


fig, ax = plt.subplots(figsize=(12, 5))


x = range(len(chart_data))


ax.bar(
    [i - 0.2 for i in x],
    chart_data["stockout_days_before"],
    width=0.4,
    label="Traditional Policy"
)


ax.bar(
    [i + 0.2 for i in x],
    chart_data["stockout_days_after"],
    width=0.4,
    label="ML-Driven Policy"
)


ax.set_xticks(list(x))

ax.set_xticklabels(
    chart_data["product_id"]
)

ax.set_xlabel("Product")

ax.set_ylabel("Stockout Days")

ax.set_title(
    "Stockout Performance Across Products"
)

ax.legend()

ax.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

st.pyplot(fig)

plt.close(fig)


# ============================================================
# PROJECT METHODOLOGY
# ============================================================

st.divider()

st.subheader("🔧 Project Methodology")

st.markdown(
    """
    **1. Demand Forecasting**
    
    Historical sales data is processed using lag features, rolling averages,
    calendar variables and promotion information. A Random Forest Regressor
    is trained separately for each product and evaluated against a
    7-day Moving Average baseline.

    **2. Inventory Optimization**
    
    EOQ is calculated to determine an efficient replenishment quantity.
    Safety stock is calculated using demand variability and a target
    95% service level. The Reorder Point combines expected lead-time
    demand with safety stock.

    **3. Inventory Simulation**
    
    A Traditional inventory policy is compared against an ML-driven
    inventory policy using actual test-period demand. Performance is
    evaluated using stockout-days, stockout units and service levels.
    """
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Demand Forecasting & Inventory Optimization Dashboard | "
    "Random Forest + EOQ + Safety Stock + Reorder Point | "
    "Dataset is synthetic and performance metrics represent "
    "results from the project's simulated environment."
)