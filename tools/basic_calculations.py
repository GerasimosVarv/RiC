import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

# Enable import of other modules
import sys
ROOT = r"C:\Users\g.varvounis\Documents\RiC"
if ROOT not in sys.path:
    sys.path.append(ROOT)
###############################

from tools.general_tools import create_path

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create console handler and set level to info
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Create formatter and add it to the handler
formatter = logging.Formatter('%(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(console_handler)

def calculate_volatility_and_returns(df: pd.DataFrame, rf: str, calculation_window: int, min_periods: int = 5, returns_horizon_in_days: int =1,
                                     volatility_column_name: str = 'volatility', returns_column_name: str = 'log_returns'):
    """Calculates rolling volatility and returns for a certain risk factor."""
    df[returns_column_name] = np.log(df[rf].dropna() / df[rf].dropna().shift(returns_horizon_in_days))
    df[volatility_column_name] = df[returns_column_name].dropna().rolling(window=calculation_window, min_periods=min_periods, closed='both').std(ddof=1)
        
    return df

def calculate_mtd_price(df: pd.DataFrame, rf: str, frequency: str = 'ME'):
    """Calculates MtD price for a certain risk factor."""

    df_mtd = df[rf].resample(frequency).mean().copy()

    return df_mtd

def calculate_ytd_price(df: pd.DataFrame, rf: str, frequency: str = 'YE'):
    """Calculates MtD price for a certain risk factor."""
    
    df_ytd = df[rf].resample(frequency).mean().copy()

    return df_ytd

def calculate_qtd_price(df: pd.DataFrame, rf: str, frequency: str = 'QE'):
    """Calculates QtD price for a certain risk factor."""

    df_qtd = df[rf].resample(frequency).mean().copy()

    return df_qtd

def calculate_wtd_price(df: pd.DataFrame, rf: str, frequency: str = 'W-FRI'):
    """Calculates QtD price for a certain risk factor."""

    df_wtd = df[rf].resample(frequency).mean().copy()

    return df_wtd

def create_detailed_tables(df: pd.DataFrame, risk_factor: str, calculation_window: int):
    """Calculates Volatility, Returns and MtD / YtD values for DataFrames of multiple risk factors."""

    # Initialie DataFrame with Risk Factor Data
    df_rf_temp = pd.DataFrame(df[risk_factor])

    # Calculate Volaitility and Logarithmic Returns 
    df_vol_rets = calculate_volatility_and_returns(df=df_rf_temp, rf=risk_factor, calculation_window=calculation_window)

    # Calculate MtD Price
    df_mtd = calculate_mtd_price(df=df_rf_temp, rf=risk_factor)

    # Calculate YtD Price
    df_ytd = calculate_ytd_price(df=df_rf_temp, rf=risk_factor)

    # Calculate WtD Price
    df_wtd = calculate_ytd_price(df=df_rf_temp, rf=risk_factor)

    return df_vol_rets, df_mtd, df_ytd, df_wtd

def calculate_spread(df_index_1: pd.DataFrame, df_index_2: pd.DataFrame, index_1: str, index_2: str):
    """Calculates spread between two market indices."""

    logger.info(f"Calculating {index_1} - {index_2} Spread...")

    # Calculate Spread of Indices (stored in pandas.series data structure)
    df_spread_calculated = df_index_1[index_1] - df_index_2[index_2]

    # Concert pandas.series to pandas.dataframe data structure
    df_spread_calculated = pd.DataFrame(df_spread_calculated)

    # Rename column of pandas.dataframe
    df_spread_calculated.columns = [index_1 + ' - ' + index_2]

    logger.info(f"Spread {index_1} - {index_2} Calculated sucessfully!")

    return df_spread_calculated

def create_term_structure(
            data_dict:dict, tenors: list, curve: str = "GR"):
    """Calculates spread between two market indices."""

    logger.info(f"Creating Term Structure Table for {curve} curve...")

    values = []
    diffs = []

    for tenor in tenors:
         
        df_temp = data_dict[tenor].copy()

        df_temp_diffs = df_temp.copy()
        df_temp_diffs = df_temp_diffs.diff()
        try:
            temp_value = df_temp[tenor].dropna().iloc[-1]
            temp_diff = df_temp_diffs[tenor].dropna().iloc[-1]
        except:
            logger.info(f"Tenor: {tenor} unsucesfully added to Curve Constructor!")
            temp_value = 0
            temp_diff = 0

        values.append(temp_value)
        diffs.append(temp_diff)

    df_term_structure = pd.DataFrame({"Tenor": tenors, "Value": values, "Delta Daily": diffs})

    logger.info(f"Term Structure Table for {curve} created sucessfully!")

    return df_term_structure

def calculate_correlations(
            data_dict:dict, tenors: list, recent_days_mode: int = 100):
    """Calculates Correlation Table for all available risk factors."""

    logger.info(f"Risk Factors' Correlation table is being calculated...")

    for idx, tenor in enumerate(tenors):
        if idx == 0:
            df_only_values_ts = pd.DataFrame(data_dict[tenor][tenor])
        else:
             df_only_values_ts = pd.concat([df_only_values_ts, pd.Series(data_dict[tenor][tenor])], axis=1)

    df_corr_table = df_only_values_ts.corr()

    df_corr_table_recent_days = df_only_values_ts.iloc[-recent_days_mode:, :].corr()
    
    logger.info(f"Risk Factors' Correlation table calcuted sucessfully!!")

    return df_corr_table, df_corr_table_recent_days

def calculate_linear_pnl(df: pd.DataFrame, pnl_col: str, quant_col: str, price_col: str, market_price_col: str):
    """Calculates P&L of linear products (e.g., futures)."""
    df[pnl_col] = df[quant_col] * (df[market_price_col] - df[price_col])

    return df

            



