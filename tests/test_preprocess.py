import pandas as pd 
import numpy as np
from src.preprocess import preprocess_insurance_data


def test_preprocess_drops_col(): 
    # create a sample dataframe
    df=pd.DataFrame({
        "CustomValueEstimate":[100,200],
        "TotalClaims":[500,50],
        "TotalPremium":[50,0],
        "Gender":["Male","Famale"]
    })
    df_processed=preprocess_insurance_data(df)
    # check for drop
    assert "CustomValueEstimate" not in df_processed.columns
    # check for LossRatio
    assert "LossRatio" in df_processed.columns
    # check the first value of LossRatio
    assert df_processed.loc[0,"LossRatio"]==10.0
    # check the second value of LossRatio
    assert pd.isna(df_processed.loc[1,"LossRatio"])
def test_preprocess_catagorical_and_fillna():
    # create a sample dataframe
    df=pd.DataFrame({
        "Gender":[np.nan,"Female"],
        "Bank":["BankA",np.nan],
        "TotalClaims":[500,50],
        "TotalPremium":[50,0]
    })
    df_processed=preprocess_insurance_data(df)
    # check for catagorical dtype
    assert df_processed["Bank"].dtype.name=="category"
    # check for filled with mode
    assert df_processed.loc[0,"Gender"]=="Female"
def test_preprocess_numeric_fillna():
    # create a sample dataframe
    df = pd.DataFrame({
        "TotalPremium": [1000, np.nan],
        "TotalClaims": [50, np.nan],
        "Gender": ["Male", "Female"]
    })
    df_processed = preprocess_insurance_data(df)
    # check for filled with median
    assert df_processed.loc[1, "TotalPremium"]==1000
    assert df_processed.loc[1, "TotalClaims"]==50