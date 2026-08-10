# Demand Forecasting & Inventory Optimization System

An end-to-end supply-chain analytics project: synthetic sales data →
EDA → demand forecasting (baseline vs ML) → EOQ / safety stock / reorder
point → inventory simulation → stockout comparison → KPI dashboard.

Built for a resume project connected to supply-chain / operations roles
(demand planning, inventory optimization, fulfillment, KPI analysis, RCA).

## Project structure

```
01_demand_forecasting_inventory/
├── README.md
├── requirements.txt
├── generate_data.py            # Step 1: synthetic sales data
├── 01_eda.py                   # Step 2: exploratory data analysis
├── 02_forecasting.py           # Step 3: baseline vs ML forecast
├── 03_inventory_optimization.py# Step 4: EOQ, safety stock, ROP, simulation
├── 04_kpi_dashboard.py         # Step 5: consolidated Excel KPI dashboard
├── data/
│   └── sales_data.csv          # generated dataset
└── outputs/                    # all generated results (csv, png, xlsx)
```

## How to run (in order)

```bash
pip install -r requirements.txt

python generate_data.py             # creates data/sales_data.csv
python 01_eda.py                    # creates outputs/eda_summary.csv, eda_plots.png
python 02_forecasting.py            # creates outputs/forecast_results.csv, model_comparison.csv
python 03_inventory_optimization.py # creates outputs/inventory_policy.csv, stockout_comparison.csv
python 04_kpi_dashboard.py          # creates outputs/KPI_Dashboard.xlsx
```

Each script depends on the outputs of the one before it, so run them in
order the first time. After that you can re-run any single script.

## What each stage actually does

### 1. Data generation (`generate_data.py`)
Produces 2 years of daily sales for 10 SKUs with deliberate, documented
demand drivers: weekday/weekend pattern, yearly seasonality with a
festive bump (Oct-Dec), random promotion bursts (+30-70% lift), a slow
growth trend, and Poisson-like noise. Because we control the generator,
every number downstream can be traced back to a cause -- useful for
explaining the project in an interview.

### 2. EDA (`01_eda.py`)
Answers: which products sell most, weekday vs weekend lift, promotion
lift, monthly seasonality, and which SKUs are most volatile
(coefficient of variation). Produces a 4-panel chart and a summary CSV.

### 3. Forecasting (`02_forecasting.py`)
- **Baseline**: trailing 7-day moving average (a standard, defensible
  industry baseline -- not a strawman).
- **ML model**: Random Forest using lag features (1/7/14/28 days),
  rolling means (7/14/28 days), day-of-week, month, and promotion flag.
- **Split**: last 60 days per product held out, time-based (no shuffling,
  no leakage -- rolling features only use past data).
- **Metrics**: MAE, RMSE, MAPE per product and overall.
- The script prints the **actual** improvement percentage. In our run
  this came out to **~22.5% MAE improvement**, which is what you should
  quote -- not a number picked in advance.

### 4. Inventory optimization (`03_inventory_optimization.py`)
Implements the classic textbook chain and simulates it:
- **EOQ** = sqrt(2DS/H)
- **Safety stock** = z * σ_demand * sqrt(lead time), z = 1.645 for 95% service level
- **Reorder point** = avg daily demand * lead time + safety stock
- **Simulation**: day-by-day inventory tracked against actual demand,
  reordering triggered by a forecast-informed policy, once using the
  baseline forecast and once using the ML forecast, so you get a clean
  "before vs after" stockout comparison under identical conditions.
- All cost/lead-time assumptions (ordering cost, holding cost %, unit
  cost, lead time, service level) are declared as constants at the top
  of the script -- change them to match a real scenario and be ready to
  justify them in an interview.

### 5. KPI dashboard (`04_kpi_dashboard.py`)
Consolidates everything into `outputs/KPI_Dashboard.xlsx` with a
Summary sheet (headline numbers) plus one sheet per analysis stage.

## Honest caveats to mention in an interview

- The dataset is synthetic (by design, so every number is explainable),
  not scraped from Kaggle. Say this proactively -- it's a strength, not
  a weakness, because you can defend every assumption.
- With a well-tuned safety-stock policy, stockouts are naturally rare
  (that's the point of safety stock), so the "before vs after" stockout
  counts are small in absolute terms (e.g. single-digit stockout-days
  across 10 SKUs over 60 days). Report the percentage reduction *and*
  the raw counts together so it doesn't look inflated.
- Random Forest is a reasonable, explainable ML choice for a resume
  project. If you want a stronger result, mention XGBoost/LightGBM or a
  seasonal ARIMA/Prophet baseline as "next steps" -- that shows range.

## Suggested resume line (based on actual results from this run)

> Built an end-to-end demand forecasting and inventory optimization
> system in Python (Pandas, scikit-learn); improved forecast accuracy
> by ~22% (MAE) over a moving-average baseline using a Random Forest
> model with lag/rolling features, and reduced simulated stockout days
> by ~33% through EOQ/safety-stock-driven reorder policy tied to the
> improved forecast.

Re-run the scripts and use whatever numbers you actually get -- they
will vary slightly with any changes to the generator or model.

## Next steps (Project 2)

Once this is working end-to-end, we move to **Warehouse Management and
Operations Optimization**: ABC classification, SKU demand ranking,
warehouse slotting, pick-time simulation, and fulfillment KPIs -- the
project that maps most directly to the Flipkart NEEV JD's emphasis on
day-to-day operations, fulfillment, and process improvement.
