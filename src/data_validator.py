import pandas as pd


def validate_dataframe(
    df,
    date_col,
    product_col,
    demand_col
):
    """
    Validate a sales-demand dataframe before forecasting.

    Returns:
        dict containing errors, warnings and validation status.
    """

    errors = []
    warnings = []

    # ---------------------------------------------------------
    # Basic dataframe check
    # ---------------------------------------------------------

    if df is None or df.empty:
        errors.append("The uploaded dataset is empty.")
        return {
            "valid": False,
            "errors": errors,
            "warnings": warnings
        }

    # ---------------------------------------------------------
    # Required column check
    # ---------------------------------------------------------

    required_columns = {
        "date": date_col,
        "product": product_col,
        "demand": demand_col
    }

    for field, column in required_columns.items():

        if column is None or column not in df.columns:
            errors.append(
                f"Required {field} column is missing."
            )

    # Stop here if required columns don't exist
    if errors:
        return {
            "valid": False,
            "errors": errors,
            "warnings": warnings
        }

    # ---------------------------------------------------------
    # Date validation
    # ---------------------------------------------------------

    dates = pd.to_datetime(
        df[date_col],
        errors="coerce"
    )

    invalid_dates = dates.isna().sum()

    if invalid_dates > 0:
        errors.append(
            f"{invalid_dates} invalid date values detected."
        )

    # ---------------------------------------------------------
    # Demand validation
    # ---------------------------------------------------------

    demand = pd.to_numeric(
        df[demand_col],
        errors="coerce"
    )

    invalid_demand = demand.isna().sum()

    if invalid_demand > 0:
        errors.append(
            f"{invalid_demand} invalid or non-numeric demand values detected."
        )

    # ---------------------------------------------------------
    # Negative demand
    # ---------------------------------------------------------

    negative_demand = (demand < 0).sum()

    if negative_demand > 0:
        warnings.append(
            f"{negative_demand} negative demand values detected. "
            "Check whether these represent returns or data errors."
        )

    # ---------------------------------------------------------
    # Duplicate product-date records
    # ---------------------------------------------------------

    duplicate_mask = df.duplicated(
        subset=[date_col, product_col],
        keep=False
    )

    duplicate_rows = duplicate_mask.sum()

    if duplicate_rows > 0:
        warnings.append(
            f"{duplicate_rows} rows belong to duplicate "
            "product-date combinations."
        )

    # ---------------------------------------------------------
    # Missing values
    # ---------------------------------------------------------

    missing_demand = df[demand_col].isna().sum()

    if missing_demand > 0:
        warnings.append(
            f"{missing_demand} missing demand values detected."
        )

    # ---------------------------------------------------------
    # History length
    # ---------------------------------------------------------

    valid_dates = dates.dropna()

    if not valid_dates.empty:

        history_days = (
            valid_dates.max() -
            valid_dates.min()
        ).days + 1

        if history_days < 30:

            warnings.append(
                "Less than 30 days of history are available. "
                "Forecast reliability may be low."
            )

        elif history_days < 90:

            warnings.append(
                "Less than 90 days of history are available. "
                "Long-term seasonality cannot be estimated reliably."
            )

    # ---------------------------------------------------------
    # Final validation result
    # ---------------------------------------------------------

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings
    }