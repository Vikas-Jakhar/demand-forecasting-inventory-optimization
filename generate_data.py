"""
generate_data.py
-----------------
Generates a controlled, synthetic daily sales dataset for 10 products
over 2 years, with realistic demand drivers baked in on purpose so we
know exactly where every number comes from:

    - base demand per product (different price points / popularity)
    - weekly seasonality (higher demand on weekends)
    - yearly seasonality (festive/holiday bumps, e.g. Oct-Dec)
    - a slow upward demand trend (business growth)
    - random promotions (~8% of days) that lift demand 30-70%
    - Poisson demand noise (integer, realistic for unit sales)

Output: data/sales_data.csv with columns:
    date, product_id, product_name, demand, promotion
"""

import numpy as np
import pandas as pd
from pathlib import Path

RANDOM_SEED = 42
START_DATE = "2024-01-01"
NUM_DAYS = 730  # 2 years
OUTPUT_PATH = Path(__file__).parent / "data" / "sales_data.csv"

rng = np.random.default_rng(RANDOM_SEED)

PRODUCTS = [
    {"product_id": "P01", "product_name": "Wireless Mouse",       "base_demand": 45, "trend_per_day": 0.010, "volatility": 0.15},
    {"product_id": "P02", "product_name": "Bluetooth Speaker",    "base_demand": 30, "trend_per_day": 0.020, "volatility": 0.25},
    {"product_id": "P03", "product_name": "USB-C Cable",          "base_demand": 90, "trend_per_day": 0.005, "volatility": 0.10},
    {"product_id": "P04", "product_name": "Laptop Backpack",      "base_demand": 20, "trend_per_day": 0.008, "volatility": 0.20},
    {"product_id": "P05", "product_name": "Smart Watch",          "base_demand": 15, "trend_per_day": 0.030, "volatility": 0.35},
    {"product_id": "P06", "product_name": "Power Bank 10000mAh",  "base_demand": 35, "trend_per_day": 0.012, "volatility": 0.18},
    {"product_id": "P07", "product_name": "Mechanical Keyboard",  "base_demand": 12, "trend_per_day": 0.006, "volatility": 0.22},
    {"product_id": "P08", "product_name": "LED Desk Lamp",        "base_demand": 18, "trend_per_day": 0.004, "volatility": 0.15},
    {"product_id": "P09", "product_name": "Phone Case",           "base_demand": 70, "trend_per_day": 0.007, "volatility": 0.12},
    {"product_id": "P10", "product_name": "Wireless Earbuds",     "base_demand": 40, "trend_per_day": 0.025, "volatility": 0.30},
]

# Weekly pattern: Mon=0 ... Sun=6. Weekends see a demand lift.
WEEKDAY_MULTIPLIER = {0: 0.95, 1: 0.95, 2: 0.97, 3: 1.00, 4: 1.10, 5: 1.35, 6: 1.25}

def yearly_seasonality(day_of_year: int) -> float:
    """Smooth seasonal curve + an explicit festive bump in Oct-Dec (Diwali/Xmas sales)."""
    smooth = 1.0 + 0.12 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    return smooth

def festive_bump(date: pd.Timestamp) -> float:
    if date.month in (10, 11):       # Oct-Nov festive/Diwali season
        return 1.45
    if date.month == 12 and date.day <= 25:  # Christmas run-up
        return 1.30
    if date.month == 1 and date.day <= 5:    # New year clearance
        return 1.15
    return 1.0

def generate():
    dates = pd.date_range(start=START_DATE, periods=NUM_DAYS, freq="D")
    rows = []

    for prod in PRODUCTS:
        # promotion calendar: ~8% of days, in short 2-4 day bursts
        promo_days = set()
        num_promo_bursts = int(NUM_DAYS * 0.08 / 3)
        burst_starts = rng.choice(NUM_DAYS - 4, size=num_promo_bursts, replace=False)
        for s in burst_starts:
            length = rng.integers(2, 5)
            for d in range(s, s + length):
                promo_days.add(d)

        for i, date in enumerate(dates):
            trend = 1.0 + prod["trend_per_day"] * i / 30.0  # gentle monthly compounding trend
            weekday_mult = WEEKDAY_MULTIPLIER[date.dayofweek]
            season_mult = yearly_seasonality(date.dayofyear)
            fest_mult = festive_bump(date)
            is_promo = i in promo_days
            promo_mult = 1.0 + rng.uniform(0.30, 0.70) if is_promo else 1.0

            expected_demand = (
                prod["base_demand"]
                * trend
                * weekday_mult
                * season_mult
                * fest_mult
                * promo_mult
            )

            noisy = rng.normal(loc=expected_demand, scale=expected_demand * prod["volatility"])
            demand = max(0, int(round(noisy)))

            rows.append({
                "date": date.strftime("%Y-%m-%d"),
                "product_id": prod["product_id"],
                "product_name": prod["product_name"],
                "demand": demand,
                "promotion": int(is_promo),
            })

    df = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Generated {len(df):,} rows for {len(PRODUCTS)} products over {NUM_DAYS} days")
    print(f"Saved to: {OUTPUT_PATH}")
    print(df.head(10))

if __name__ == "__main__":
    generate()
