"""
01_eda.py
---------
Exploratory Data Analysis on data/sales_data.csv.

Answers:
  - Which products sell the most?
  - What is the average daily demand?
  - Does demand increase on weekends?
  - What happens during promotional periods?
  - Are there seasonal patterns?
  - Which products have the most volatile demand?

Outputs:
  - outputs/eda_summary.csv          (per-product summary table)
  - outputs/eda_plots.png            (4-panel chart)
  - Printed findings to console
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

DATA_PATH = Path(__file__).parent / "data" / "sales_data.csv"
OUT_DIR = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)


def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df["day_of_week"] = df["date"].dt.day_name()
    df["is_weekend"] = df["date"].dt.dayofweek >= 5
    df["month"] = df["date"].dt.month
    return df


def run_eda(df: pd.DataFrame):
    print("=" * 70)
    print("1) TOTAL & AVERAGE DEMAND BY PRODUCT")
    print("=" * 70)
    by_product = (
        df.groupby("product_name")["demand"]
        .agg(total_demand="sum", avg_daily_demand="mean", volatility_std="std")
        .sort_values("total_demand", ascending=False)
    )
    by_product["coefficient_of_variation"] = (
        by_product["volatility_std"] / by_product["avg_daily_demand"]
    )
    print(by_product.round(2))

    print("\n" + "=" * 70)
    print("2) OVERALL AVERAGE DAILY DEMAND (ALL PRODUCTS COMBINED)")
    print("=" * 70)
    overall_avg = df.groupby("date")["demand"].sum().mean()
    print(f"Average total demand per day across all products: {overall_avg:.1f} units")

    print("\n" + "=" * 70)
    print("3) WEEKEND VS WEEKDAY DEMAND")
    print("=" * 70)
    weekend_effect = df.groupby("is_weekend")["demand"].mean()
    weekday_avg = weekend_effect.loc[False]
    weekend_avg = weekend_effect.loc[True]
    lift_pct = (weekend_avg - weekday_avg) / weekday_avg * 100
    print(f"Weekday avg demand: {weekday_avg:.2f}")
    print(f"Weekend avg demand: {weekend_avg:.2f}")
    print(f"Weekend lift: {lift_pct:+.1f}%")

    print("\n" + "=" * 70)
    print("4) PROMOTION EFFECT ON DEMAND")
    print("=" * 70)
    promo_effect = df.groupby("promotion")["demand"].mean()
    no_promo, promo = promo_effect.loc[0], promo_effect.loc[1]
    promo_lift_pct = (promo - no_promo) / no_promo * 100
    print(f"Avg demand (no promotion): {no_promo:.2f}")
    print(f"Avg demand (during promotion): {promo:.2f}")
    print(f"Promotion lift: {promo_lift_pct:+.1f}%")

    print("\n" + "=" * 70)
    print("5) SEASONAL / MONTHLY PATTERN (ALL PRODUCTS COMBINED)")
    print("=" * 70)
    monthly = df.groupby("month")["demand"].mean()
    print(monthly.round(2))
    peak_month = monthly.idxmax()
    print(f"Peak demand month: {peak_month} (festive/holiday season)")

    print("\n" + "=" * 70)
    print("6) MOST VOLATILE PRODUCTS (BY COEFFICIENT OF VARIATION)")
    print("=" * 70)
    print(by_product.sort_values("coefficient_of_variation", ascending=False)
          [["avg_daily_demand", "coefficient_of_variation"]].round(3))

    # Save summary table
    by_product.round(2).to_csv(OUT_DIR / "eda_summary.csv")
    print(f"\nSaved per-product summary -> {OUT_DIR / 'eda_summary.csv'}")

    make_plots(df, by_product, weekend_effect, promo_effect, monthly)


def make_plots(df, by_product, weekend_effect, promo_effect, monthly):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Total demand by product
    by_product["total_demand"].sort_values().plot(kind="barh", ax=axes[0, 0], color="#4C72B0")
    axes[0, 0].set_title("Total Demand by Product")
    axes[0, 0].set_xlabel("Total Units Sold")

    # Weekday vs weekend
    weekend_effect.rename({False: "Weekday", True: "Weekend"}).plot(
        kind="bar", ax=axes[0, 1], color=["#4C72B0", "#DD8452"]
    )
    axes[0, 1].set_title("Avg Demand: Weekday vs Weekend")
    axes[0, 1].set_xticklabels(["Weekday", "Weekend"], rotation=0)
    axes[0, 1].set_ylabel("Avg Demand (units)")

    # Promotion effect
    promo_effect.rename({0: "No Promotion", 1: "Promotion"}).plot(
        kind="bar", ax=axes[1, 0], color=["#4C72B0", "#55A868"]
    )
    axes[1, 0].set_title("Avg Demand: Promotion vs No Promotion")
    axes[1, 0].set_xticklabels(["No Promotion", "Promotion"], rotation=0)
    axes[1, 0].set_ylabel("Avg Demand (units)")

    # Monthly seasonality
    monthly.plot(kind="line", marker="o", ax=axes[1, 1], color="#C44E52")
    axes[1, 1].set_title("Avg Demand by Month (Seasonality)")
    axes[1, 1].set_xlabel("Month")
    axes[1, 1].set_ylabel("Avg Demand (units)")
    axes[1, 1].set_xticks(range(1, 13))

    plt.tight_layout()
    out_path = OUT_DIR / "eda_plots.png"
    plt.savefig(out_path, dpi=120)
    print(f"Saved plots -> {out_path}")


if __name__ == "__main__":
    df = load_data()
    run_eda(df)
