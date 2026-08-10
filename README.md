# Demand Forecasting & Inventory Optimization System

A machine-learning-based supply chain analytics project that combines demand forecasting with inventory optimization to improve replenishment decisions and reduce simulated stockouts.

## Project Overview

The objective of this project is to build an end-to-end demand forecasting and inventory optimization pipeline for a multi-SKU retail environment.

The system:

- Generates a reproducible synthetic retail demand dataset
- Performs exploratory data analysis
- Identifies seasonality, promotion effects, and demand variability
- Compares a 7-day moving-average forecasting baseline with a Random Forest model
- Calculates EOQ, safety stock, and reorder points
- Simulates inventory performance under baseline and ML-driven policies
- Produces a consolidated Excel KPI dashboard

## Project Pipeline

```text
Synthetic Demand Data
        │
        ▼
Exploratory Data Analysis
        │
        ├── Product Demand
        ├── Seasonality
        ├── Weekend Effects
        └── Promotion Impact
        │
        ▼
Demand Forecasting
        │
        ├── 7-Day Moving Average
        └── Random Forest
        │
        ▼
Forecast Evaluation
        │
        ├── MAE
        ├── RMSE
        └── MAPE
        │
        ▼
Inventory Optimization
        │
        ├── EOQ
        ├── Safety Stock
        └── Reorder Point
        │
        ▼
60-Day Inventory Simulation
        │
        ▼
KPI Dashboard

Dataset

The project uses a synthetically generated retail demand dataset because proprietary company sales and inventory data were not available.

The dataset contains:

10 products/SKUs
730 days of historical demand
7,300 total observations
Product-level demand variation
Weekday/weekend effects
Promotional demand changes
Monthly seasonality
Random demand variability

The synthetic data is generated using generate_data.py, making the complete project reproducible.

Exploratory Data Analysis

The EDA stage identified several demand patterns.

Key Findings
Metric	Result
Average total demand	520.1 units/day
Weekend demand lift	+30.6%
Promotion demand lift	+44.9%
Peak demand month	November

Demand variability also differed substantially between products.

For example:

USB-C Cable had relatively stable demand with a coefficient of variation of approximately 0.25.
Wireless Earbuds had significantly higher relative variability with a coefficient of variation of approximately 0.45.

These patterns were considered when developing the forecasting and inventory strategy.

Demand Forecasting

Two forecasting approaches were evaluated:

Baseline

A 7-day moving average was used as a simple benchmark.

Machine Learning Model

A Random Forest regression model was trained using historical demand and engineered temporal features.

Features include:

Lagged demand
Rolling demand statistics
Day of week
Month
Promotion indicator
Product-level demand history
Evaluation Method

The last 60 days of demand for each product were held out as the test period.

The model was evaluated using:

Mean Absolute Error (MAE)
Root Mean Squared Error (RMSE)
Mean Absolute Percentage Error (MAPE)

Forecasting Results
Metric                        Baseline	                     Random Forest
Average MAE	                   14.28	                           11.07
Average MAPE	                 26.98%	                           21.81%

Result

Random Forest reduced average MAE by 22.5% compared with the 7-day moving-average baseline.

Performance varied by SKU. For example, the model improved MAE by more than 30% for several high-volume products, while the Smart Watch SKU showed a small deterioration.

This variation highlights that a single forecasting approach may not be optimal for every demand pattern.

Inventory Optimization

The demand information was then used to construct an inventory policy.

Economic Order Quantity

EOQ was calculated to determine an economically appropriate replenishment quantity based on demand, ordering cost, and holding cost assumptions.

EOQ=
H
2DS
	​

	​


where:

D = annual demand
S = ordering cost
H = annual holding cost per unit
Safety Stock

Safety stock was calculated to provide a buffer against demand variability.

Reorder Point

The reorder point determines when replenishment should be triggered.

ROP=Lead Time Demand + Safety Stock
Inventory Simulation

A 60-day inventory simulation was performed to compare:

 1.Baseline inventory policy
 2.ML-driven inventory policy
Results
KPI	Baseline	ML Policy
Stockout-days	9	6
Stockout-day reduction	—	33.3%

The ML-driven policy reduced aggregate simulated stockout-days from 9 to 6, corresponding to a 33.3% reduction.

The impact varied across SKUs. Some products experienced complete elimination of stockout-days, while others showed no change during the simulation period.

Dashboard

The project generates an Excel KPI dashboard containing:

Forecast accuracy metrics
Model comparison
Inventory policy parameters
Stockout performance
Service-level metrics

Output:

outputs/KPI_Dashboard.xlsx
Project Structure
demand-forecasting-inventory-optimization/
│
├── data/
│   └── sales_data.csv
│
├── outputs/
│   ├── KPI_Dashboard.xlsx
│   ├── eda_plots.png
│   ├── eda_summary.csv
│   ├── forecast_results.csv
│   ├── inventory_policy.csv
│   ├── model_comparison.csv
│   ├── simulation_before.csv
│   ├── simulation_after.csv
│   └── stockout_comparison.csv
│
├── generate_data.py
├── 01_eda.py
├── 02_forecasting.py
├── 03_inventory_optimization.py
├── 04_kpi_dashboard.py
├── requirements.txt
├── README.md
└── .gitignore
How to Run
1. Clone the repository
git clone https://github.com/Vikas-Jakhar/demand-forecasting-inventory-optimization.git
cd demand-forecasting-inventory-optimization
2. Create a virtual environment
python -m venv .venv
3. Activate the environment

Windows PowerShell:

.venv\Scripts\Activate.ps1
4. Install dependencies
pip install -r requirements.txt
5. Run the pipeline

Run the scripts in the following order:

python generate_data.py
python 01_eda.py
python 02_forecasting.py
python 03_inventory_optimization.py
python 04_kpi_dashboard.py

The generated datasets and analytical outputs will be saved in the data/ and outputs/ directories.

Technologies Used
Python
Pandas
NumPy
Scikit-learn
Matplotlib
OpenPyXL
Machine Learning
Demand Forecasting
Inventory Optimization
EOQ
Safety Stock
Reorder Point
Limitations

This project uses synthetic data rather than proprietary business data.

Therefore, the reported performance metrics represent results from the project's simulated environment and should not be interpreted as real-world company performance.

The inventory simulation also uses simplified assumptions for costs, lead times, demand behavior, and replenishment.

A production system would require integration with real sales, inventory, supplier lead-time, procurement, and warehouse data.

Key Results

22.5% lower average MAE using Random Forest compared with a 7-day moving-average baseline.

33.3% reduction in simulated stockout-days using the ML-driven inventory policy.

Author
Vikas Jakhar
GitHub:https://github.com/Vikas-Jakhar
