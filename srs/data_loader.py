import pandas as pd
import numpy as np
from typing import List

# Columns grouped by type
DATE_COLS = ["TransactionMonth", "VehicleIntroDate"]

BOOL_STR_COLS = [
    "WrittenOff", "Rebuilt", "Converted", "CrossBorder"
]

BOOL_BOOL_COLS = ["IsVATRegistered"]

NUMERIC_COLS = [
    "mmcode", "Cylinders", "cubiccapacity", "kilowatts", "NumberOfDoors",
    "CustomValueEstimate", "CapitalOutstanding", "NumberOfVehiclesInFleet",
    "SumInsured", "CalculatedPremiumPerTerm", "TotalPremium", "TotalClaims"
]


def load_insurance_data(filepath: str) -> pd.DataFrame:
    """
    Loads the insurance dataset safely, converts datatypes, and prepares it 
    for downstream EDA and preprocessing.

    Improvements:
    - Full validation on file input
    - Column existence checks for all expected columns
    - Safe type casting with fallback behavior
    - Resilient boolean conversions for dirty inputs
    - Protects against unexpected or missing columns

    Parameters
    ----------
    filepath : str
        Path to the pipe-separated .txt dataset.

    Returns
    -------
    pd.DataFrame
        Clean, typed DataFrame ready for EDA and modeling.

    Raises
    ------
    FileNotFoundError
        If the file path does not exist.
    ValueError
        If the file is empty or cannot be parsed.
    """

    # -------------------------------------------------------
    # 1️⃣ Validate File Exists
    # -------------------------------------------------------
    try:
        df = pd.read_csv(filepath, sep="|", header=0)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}")
    except Exception as e:
        raise ValueError(f"Error reading file '{filepath}': {e}")

    if df.empty:
        raise ValueError("Dataset loaded is empty.")

    # -------------------------------------------------------
    # 2️⃣ Clean column names
    # -------------------------------------------------------
    df.columns = df.columns.str.strip()

    # -------------------------------------------------------
    # 3️⃣ Clean string/object columns
    # -------------------------------------------------------
    obj_cols = df.select_dtypes(include=["object"]).columns
    for col in obj_cols:
        df[col] = df[col].astype(str).str.strip()

    # -------------------------------------------------------
    # 4️⃣ Safe date conversion
    # -------------------------------------------------------
    for col in DATE_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # -------------------------------------------------------
    # 5️⃣ Numeric conversion with fallback
    # -------------------------------------------------------
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # -------------------------------------------------------
    # 6️⃣ Convert boolean string columns (Yes/No → True/False)
    # -------------------------------------------------------
    for col in BOOL_STR_COLS:
        if col in df.columns:
            df[col] = (
                df[col]
                .str.strip()
                .str.title()   # Ensures "yes", "YES" all become "Yes"
                .map({"Yes": True, "No": False})
            )

           

    # -------------------------------------------------------
    # 7️⃣ Clean True/False columns
    # -------------------------------------------------------
    for col in BOOL_BOOL_COLS:
        if col in df.columns:
            df[col] = df[col].astype(bool)

    return df