import pandas as pd
import numpy as np
from typing import List

# Columns to drop due to excessive missingness or irrelevance
COL_TO_DROP: List[str] = [
    "CustomValueEstimate", "CapitalOutstanding",
    "WrittenOff", "Rebuilt", "Converted",
    "CrossBorder", "NumberOfVehiclesInFleet"
]

REQUIRED_COLUMNS = ["TotalClaims", "TotalPremium"]

def preprocess_insurance_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the insurance dataset for EDA and modeling.
    
    Improvements added:
    - Robust error handling and validation
    - Ensures mandatory columns are present
    - Handles divide-by-zero and infinite values safely
    - More defensive filling strategy
    - Safe mode() access even when mode is empty
    - Additional sanity checks on numeric and categorical columns

    Args:
        df (pd.DataFrame): Raw insurance dataset

    Returns:
        pd.DataFrame: Cleaned + feature-enhanced dataset

    Raises:
        ValueError: If required columns are missing or df is empty
    """

    # -----------------------------
    # 0️⃣ Validation checks
    # -----------------------------
    if df is None or df.empty:
        raise ValueError("Input DataFrame is empty or None.")

    missing_required = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    df = df.copy()

    # -----------------------------
    # 1️⃣ Drop sparse/uninformative columns
    # -----------------------------
    df.drop(columns=[col for col in COL_TO_DROP if col in df.columns],
            inplace=True,
            errors='ignore')

    # -----------------------------
    # 2️⃣ Create LossRatio column safely
    # -----------------------------
    try:
        df["LossRatio"] = df["TotalClaims"] / df["TotalPremium"].replace(0, np.nan)
    except Exception as e:
        raise ValueError(f"Error computing LossRatio: {e}")

    df["LossRatio"] = df["LossRatio"].replace([np.inf, -np.inf], np.nan)

    # -----------------------------
    # 3️⃣ Clean categorical columns
    # -----------------------------
    cat_cols = df.select_dtypes(include=["object", "category"]).columns

    for col in cat_cols:
        # Strip extra whitespace safely
        df[col] = df[col].astype(str).str.strip()

        # Mode handling (avoid error when mode is empty)
        mode_value = df[col].mode(dropna=True)
        fill_value = mode_value[0] if not mode_value.empty else "Unknown"

        df[col] = df[col].fillna(fill_value).astype("category")

    # -----------------------------
    # 4️⃣ Fill numeric columns with median
    # -----------------------------
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns

    for col in num_cols:
        try:
            df[col] = df[col].fillna(df[col].median())
        except Exception:
            df[col] = df[col].fillna(0)

    # -----------------------------
    # 5️⃣ Fill boolean columns
    # -----------------------------
    bool_cols = df.select_dtypes(include=["bool"]).columns

    for col in bool_cols:
        mode_value = df[col].mode(dropna=True)
        fill_value = mode_value[0] if not mode_value.empty else False
        df[col] = df[col].fillna(fill_value)

    # -----------------------------
    # 6️⃣ Final sanity checks
    # -----------------------------
    if df["LossRatio"].isna().sum() > 0:
        df["LossRatio"] = df["LossRatio"].fillna(df["LossRatio"].median())

    return df