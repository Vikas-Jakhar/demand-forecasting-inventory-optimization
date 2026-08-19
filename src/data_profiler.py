import pandas as pd
import numpy as np


def detect_frequency(date_series):
    """
    Estimate the observation frequency from date differences.

    Returns:
        frequency label and median interval in days.
    """

    dates = (
        pd.to_datetime(
            date_series,
            errors="coerce"
        )
        .dropna()
        .drop_duplicates()
        .sort_values()
    )

    if len(dates) < 2:
        return "Unknown", None

    differences = dates.diff().dropna()

    median_days = differences.dt.total_seconds().median() / 86400

    if median_days <= 1.5:
        frequency = "Daily"

    elif median_days <= 8:
        frequency = "Weekly"

    elif median_days <= 35:
        frequency = "Monthly"

    elif median_days <= 100:
        frequency = "Quarterly"

    else:
        frequency = "Irregular"

    return frequency, median_days


def profile_dataset(
    df,
    date_col,
    product_col,
    demand_col,
    promotion_col=None
):
    """
    Analyze the structure and forecasting capability
    of an uploaded demand dataset.
    """

    # ---------------------------------------------------------
    # Basic information
    # ---------------------------------------------------------

    df = df.copy()

    df[date_col] = pd.to_datetime(
        df[date_col],
        errors="coerce"
    )

    df[demand_col] = pd.to_numeric(
        df[demand_col],
        errors="coerce"
    )

    valid_dates = df[date_col].dropna()

    # ---------------------------------------------------------
    # Date range
    # ---------------------------------------------------------

    if len(valid_dates) > 0:

        start_date = valid_dates.min()
        end_date = valid_dates.max()

        history_days = (
            end_date - start_date
        ).days + 1

    else:

        start_date = None
        end_date = None
        history_days = 0

    # ---------------------------------------------------------
    # Frequency
    # ---------------------------------------------------------

    frequency, median_interval = detect_frequency(
        valid_dates
    )

    # ---------------------------------------------------------
    # Product information
    # ---------------------------------------------------------

    number_of_skus = df[product_col].nunique()

    records = len(df)

    # ---------------------------------------------------------
    # Missing values
    # ---------------------------------------------------------

    missing_values = int(
        df[[date_col, product_col, demand_col]]
        .isna()
        .sum()
        .sum()
    )

    missing_demand = int(
        df[demand_col].isna().sum()
    )

    # ---------------------------------------------------------
    # Duplicate product-date combinations
    # ---------------------------------------------------------

    duplicate_product_dates = int(
        df.duplicated(
            subset=[date_col, product_col]
        ).sum()
    )

    # ---------------------------------------------------------
    # Date coverage
    # ---------------------------------------------------------

    unique_dates = (
        df[date_col]
        .dropna()
        .drop_duplicates()
        .sort_values()
    )

    expected_dates = 0
    missing_dates = 0

    if frequency == "Daily" and len(unique_dates) > 1:

        expected_dates = (
            unique_dates.max() -
            unique_dates.min()
        ).days + 1

        actual_dates = len(unique_dates)

        missing_dates = max(
            expected_dates - actual_dates,
            0
        )

    # ---------------------------------------------------------
    # Seasonality capability
    # ---------------------------------------------------------

    if frequency == "Daily":

        weekly_seasonality = history_days >= 21

        monthly_seasonality = history_days >= 90

        annual_seasonality = history_days >= 365

        strong_annual_seasonality = history_days >= 730

    elif frequency == "Weekly":

        weekly_seasonality = False

        monthly_seasonality = history_days >= 90

        annual_seasonality = history_days >= 365

        strong_annual_seasonality = history_days >= 730

    elif frequency == "Monthly":

        weekly_seasonality = False

        monthly_seasonality = history_days >= 180

        annual_seasonality = history_days >= 730

        strong_annual_seasonality = history_days >= 1095

    else:

        weekly_seasonality = False
        monthly_seasonality = False
        annual_seasonality = False
        strong_annual_seasonality = False

    # ---------------------------------------------------------
    # ML forecasting capability
    # ---------------------------------------------------------

    if frequency == "Daily":

        if history_days >= 90:

            forecasting_mode = "ML + Baseline"

        elif history_days >= 30:

            forecasting_mode = "Baseline + Limited ML"

        else:

            forecasting_mode = "Baseline Only"

    elif frequency == "Weekly":

        if history_days >= 180:

            forecasting_mode = "ML + Baseline"

        elif history_days >= 90:

            forecasting_mode = "Baseline + Limited ML"

        else:

            forecasting_mode = "Baseline Only"

    elif frequency == "Monthly":

        if history_days >= 730:

            forecasting_mode = "ML + Baseline"

        elif history_days >= 365:

            forecasting_mode = "Baseline + Limited ML"

        else:

            forecasting_mode = "Baseline Only"

    else:

        forecasting_mode = "Baseline Only"

    # ---------------------------------------------------------
    # Demand statistics
    # ---------------------------------------------------------

    valid_demand = df[demand_col].dropna()

    if len(valid_demand) > 0:

        avg_demand = float(
            valid_demand.mean()
        )

        median_demand = float(
            valid_demand.median()
        )

        std_demand = float(
            valid_demand.std()
        )

        min_demand = float(
            valid_demand.min()
        )

        max_demand = float(
            valid_demand.max()
        )

        if avg_demand != 0:

            coefficient_of_variation = (
                std_demand / avg_demand
            )

        else:

            coefficient_of_variation = np.nan

    else:

        avg_demand = np.nan
        median_demand = np.nan
        std_demand = np.nan
        min_demand = np.nan
        max_demand = np.nan
        coefficient_of_variation = np.nan

    # ---------------------------------------------------------
    # Promotion availability
    # ---------------------------------------------------------

    promotion_available = (
        promotion_col is not None
        and promotion_col in df.columns
    )

    # ---------------------------------------------------------
    # Per-SKU history
    # ---------------------------------------------------------

    sku_history = (
        df.groupby(product_col)[date_col]
        .agg(
            first_date="min",
            last_date="max",
            records="count"
        )
        .reset_index()
    )

    sku_history["history_days"] = (
        sku_history["last_date"] -
        sku_history["first_date"]
    ).dt.days + 1

    # ---------------------------------------------------------
    # SKU classification
    # ---------------------------------------------------------

    def classify_sku(days):

        if days >= 365:

            return "Sufficient history"

        elif days >= 90:

            return "Moderate history"

        elif days >= 30:

            return "Limited history"

        else:

            return "Insufficient history"

    sku_history["history_class"] = (
        sku_history["history_days"]
        .apply(classify_sku)
    )

    # ---------------------------------------------------------
    # Return profile
    # ---------------------------------------------------------

    profile = {

        "records": records,

        "number_of_skus": number_of_skus,

        "start_date": start_date,

        "end_date": end_date,

        "history_days": history_days,

        "frequency": frequency,

        "median_interval_days": median_interval,

        "missing_values": missing_values,

        "missing_demand": missing_demand,

        "duplicate_product_dates": duplicate_product_dates,

        "missing_dates": missing_dates,

        "average_demand": avg_demand,

        "median_demand": median_demand,

        "std_demand": std_demand,

        "minimum_demand": min_demand,

        "maximum_demand": max_demand,

        "coefficient_of_variation": coefficient_of_variation,

        "promotion_available": promotion_available,

        "weekly_seasonality": weekly_seasonality,

        "monthly_seasonality": monthly_seasonality,

        "annual_seasonality": annual_seasonality,

        "strong_annual_seasonality": strong_annual_seasonality,

        "forecasting_mode": forecasting_mode,

        "sku_history": sku_history
    }

    return profile