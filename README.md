# Demand Forecasting & Inventory Optimization System

A machine-learning-based supply chain analytics project that combines demand forecasting with inventory optimization to improve replenishment decisions and reduce simulated stockouts.

The project compares a traditional 7-day moving-average forecasting approach with a Random Forest model and uses the resulting demand estimates to support inventory planning through EOQ, Safety Stock, and Reorder Point calculations.

An interactive Streamlit dashboard is included to explore product-level demand, forecasting performance, inventory policies, and simulated stockout results.

> **Note:** The dataset is synthetic, and all inventory performance results represent outcomes from the project's simulated environment.

---

## Key Results

- **22.5% improvement in forecast accuracy on an MAE basis**
- Average MAE reduced from **14.28 units to 11.07 units**
- Average MAPE improved from **26.98% to 21.81%**
- **66.7% reduction in simulated stockout-days**
- Total stockout-days reduced from **24 to 8**
- Evaluated forecasting and inventory performance across **10 retail SKUs**
- Built an interactive **Streamlit dashboard** for result exploration

---

# Project Overview

The objective of this project is to build an end-to-end demand forecasting and inventory optimization pipeline for a multi-SKU retail environment.

Poor demand forecasts can result in:

- **Stockouts**, leading to lost sales and lower service levels
- **Excess inventory**, increasing holding and storage costs

This project develops a reproducible analytics workflow that uses historical demand patterns to forecast future demand and evaluate how improved demand estimates can support better replenishment decisions.

The system:

- Generates a reproducible synthetic retail demand dataset
- Performs data validation and profiling
- Performs exploratory data analysis
- Identifies seasonality, promotion effects, and demand variability
- Compares a 7-day moving-average baseline with a Random Forest model
- Calculates EOQ, Safety Stock, and Reorder Points
- Simulates inventory performance under Traditional and ML-driven policies
- Compares stockout-days and service levels
- Provides an interactive Streamlit dashboard

---

# Project Pipeline

```text
Synthetic Demand Data
        │
        ▼
Data Validation & Profiling
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
        ├── Traditional 7-Day Moving Average
        └── Random Forest Regressor
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
        ├── Traditional Policy
        └── ML-Driven Policy
        │
        ▼
Performance Comparison
        │
        ├── Stockout-Days
        ├── Stockout Units
        └── Service Level
        │
        ▼
Streamlit Dashboard
```
## Dataset

The project uses a synthetically generated retail demand dataset because proprietary company sales and inventory data were not available.

The dataset contains:

*10 products / SKUs
*730 days of historical demand per product
*7,300 total observations
*Product-level demand variation
*Weekday/weekend effects
*Promotional demand changes
*Monthly seasonality
*Random demand variability

The synthetic data is generated using generate_data.py, making the project reproducible.

## Data Validation and Profiling

Before forecasting, the dataset is checked and profiled to support data quality and exploratory analysis.

The project includes modules for:

*Data validation
*Missing-value checks
*Duplicate checks
*Basic dataset profiling
*Demand summary analysis

These utilities are available in the src/ directory.

## Exploratory Data Analysis

The EDA stage was used to identify important demand patterns before model development.

The analysis examines:

*Product-level demand variation
*Monthly seasonality
*Weekend demand effects
*Promotional demand effects
*Demand variability across SKUs

## Key Findings
Metric                                       	Result
Average Total Demand	                        520.1 units/day
Weekend Demand Lift                         	+30.6%
Promotion Demand Lift	                        +44.9%
Peak Demand Month	                            November

Demand variability also differed substantially between products.

For example:

 *USB-C Cable showed relatively stable demand with a coefficient of variation of approximately 0.25
 *Wireless Earbuds showed significantly higher relative variability with a coefficient of variation  of approximately 0.45

These demand patterns motivated the use of lag, rolling-window, calendar, and promotion features in the forecasting model.

## Demand Forecasting

Two forecasting approaches were evaluated.

### 1. Traditional Baseline

A 7-day trailing moving average was used as the forecasting baseline.

For each day, the baseline prediction is calculated using demand from the previous seven days.

This provides a simple and interpretable benchmark against which the machine learning model can be evaluated.

### 2. Machine Learning Model

A Random Forest Regressor was trained separately for each product using historical demand and engineered temporal features.

Features Used
 *lag_1
 *lag_7
 *lag_14
 *lag_28
 *rolling_mean_7
 *rolling_mean_14
 *rolling_mean_28
 *day_of_week
 *month
 *promotion

The rolling features use only historical observations available before the prediction date, helping avoid future-data leakage.

## Evaluation Method

The final 60 days of data for each product were reserved as the test period.

A time-based train/test split was used:

 *Training data: Earlier historical observations
 *Test data: Final 60 days
 *No random shuffling
 *No future demand information used to construct lag or rolling features

The models were evaluated using:

 *Mean Absolute Error (MAE)
 *Root Mean Squared Error (RMSE)
 *Mean Absolute Percentage Error (MAPE)
 
### Forecasting Results
Metric	                         Traditional Baseline	                 Random Forest
Average MAE	                          14.28	                                 11.07
Average MAPE	                      26.98%	                             21.81%

### Result

The Random Forest model reduced average MAE by 22.5% compared with the 7-day moving-average baseline.
```text
Forecast Improvement (%) =
(Baseline MAE - ML MAE) / Baseline MAE × 100
```
Performance varied across SKUs.

The Random Forest model improved forecasting accuracy for most products, with several SKUs showing MAE improvements of approximately 30% or more.

One SKU, Smart Watch, showed a small deterioration in MAE, demonstrating that a single forecasting model may not perform equally well for every demand pattern.

## Inventory Optimization

The demand information was used to construct inventory policies for each SKU.

The project calculates:

 *Economic Order Quantity
 *Safety Stock 
 *Reorder Point 
 
### Economic Order Quantity

EOQ estimates an appropriate replenishment quantity by balancing ordering and inventory holding costs.

```text
EOQ = √(2DS / H)
```
where:

D = Annual demand
S = Ordering cost per order
H = Annual holding cost per unit

### Safety Stock

Safety Stock provides an inventory buffer against demand uncertainty and variability.

It helps reduce the probability of running out of stock while waiting for replenishment orders to arrive.

### Reorder Point

The Reorder Point determines when a replenishment order should be triggered.

```text
Reorder Point = Expected Demand During Lead Time + Safety Stock
```
The inventory policy is calculated separately for each product based on its demand characteristics.

### Inventory Simulation

A 60-day daily inventory simulation was performed using the test-period demand data.

The simulation models:

*Daily demand depletion
*Inventory on hand
*Supplier lead time
*Outstanding replenishment orders
*Order arrivals
*EOQ-based replenishment
*Safety stock
*Reorder triggers
*Stockout-days
*Stockout units
*Service level

Actual demand is used to deplete inventory, while forecast demand is used to support replenishment decisions.

## Traditional vs ML-Driven Inventory Policy

Two inventory approaches were compared.

### Traditional Policy

The Traditional policy uses the demand estimate produced by the 7-day moving-average baseline to support inventory planning and replenishment decisions.

### ML-Driven Policy

The ML-driven policy uses the Random Forest demand forecast to support forecast-informed replenishment decisions.

This allows the simulation to evaluate whether improved demand forecasting translates into better inventory performance.

### Stockout Simulation Results
Metric	                            Traditional Policy	                 ML-Driven Policy
Total Stockout-Days                        	24	                                  8

### Result

The ML-driven inventory policy reduced total simulated stockout-days from 24 to 8.

```text
Stockout Reduction (%) =
(Traditional Stockout-Days - ML Stockout-Days)
/
Traditional Stockout-Days × 100
```
66.7% simulated stockout-day reduction

The improvement is calculated directly from the simulation results and is not a hardcoded estimate.

Performance varies by SKU. Some products experienced complete elimination of simulated stockouts, while others showed smaller improvements.

### Interactive Streamlit Dashboard

The project includes an interactive Streamlit application for exploring the results.

The dashboard provides:

*Product selection
*Historical demand visualization
*Average daily demand
*Baseline forecasting MAE
*ML forecasting MAE
*EOQ
*Safety Stock
*Reorder Point
*Product-level model comparison
*Inventory policy parameters
*Stockout performance
*Overall forecast improvement
*Overall stockout reduction

## Running the Dashboard

After generating the forecasting and inventory outputs, run:
```text
streamlit run app.py
```
The application will start locally and display a URL similar to:
```text
http://localhost:8501
```
Open this URL in your browser to use the dashboard.
## Project Structure
```text
demand-forecasting-inventory-optimization/
│
├── data/
│   └── sales_data.csv
│
├── outputs/
│   ├── forecast_results.csv
│   ├── inventory_policy.csv
│   ├── model_comparison.csv
│   ├── simulation_before.csv
│   ├── simulation_after.csv
│   └── stockout_comparison.csv
│
├── src/
│   ├── __init__.py
│   ├── data_profiler.py
│   └── data_validator.py
│
├── generate_data.py
├── 01_eda.py
├── 02_forecasting.py
├── 03_inventory_optimization.py
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```
## How to Run
1. Clone the Repository
```text
git clone https://github.com/Vikas-Jakhar/demand-forecasting-inventory-optimization.git
cd demand-forecasting-inventory-optimization
```
 2.Create a Virtual Environment
 ```text
python -m venv .venv
```
3. Activate the Environment
Windows PowerShell
```text
.venv\Scripts\Activate.ps1
```
4. Install Dependencies
```text
pip install -r requirements.txt
```
5. Run the Pipeline

Run the scripts in the following order:
```text
python generate_data.py
python 01_eda.py
python 02_forecasting.py
python 03_inventory_optimization.py
```
6. Launch the Dashboard
```text
streamlit run app.py
```
The generated datasets and analytical outputs will be saved in the data/ and outputs/ directories.

### Technologies Used
Python
Pandas
NumPy
Scikit-learn
Matplotlib
Streamlit
OpenPyXL
###Concepts Used
Machine Learning
Demand Forecasting
Time-Based Train/Test Split
Feature Engineering
Lag Features
Rolling Statistics
Random Forest Regression
Forecast Accuracy Evaluation
EOQ
Safety Stock
Reorder Point
Inventory Simulation
Stockout Analysis
Service-Level Analysis
### Limitations

This project uses synthetic data rather than proprietary business data.

Therefore, the reported performance metrics represent results from the project's simulated environment and should not be interpreted as real-world company performance.

The inventory simulation also uses simplified assumptions for:

Ordering costs
Holding costs
Lead times
Demand behavior
Replenishment behavior

A production implementation would require integration with real:

Sales data
Inventory records
Supplier lead-time data
Procurement systems
Warehouse data
Product costs

Further improvements could include probabilistic forecasting and dynamic safety-stock calculation based on forecast uncertainty.

### Key Results Summary
Result	                                                        Performance
Forecast Accuracy Improvement	                                  22.5%
Baseline Average MAE	                                          14.28 units
ML Average MAE	                                                  11.07 units
Traditional Stockout-Days	                                      24
ML-Driven Stockout-Days	                                          8
Simulated Stockout-Day Reduction	                              66.7%

All results are based on the synthetic dataset and the inventory simulation implemented in this project.

### Author

Vikas Jakhar

Electronics and Communication Engineering
NIT Jalandhar

GitHub: Vikas-Jakhar
