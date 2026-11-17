import os
from datetime import datetime
import pandas as pd
from datetime import datetime as dt

# Enable import of other modules
import sys
ROOT = r"C:\Users\g.varvounis\Documents\RiC"
if ROOT not in sys.path:
    sys.path.append(ROOT)
###############################

def create_path(directory: str = '', filename: str = ''):
    """Creates the full path of files."""
    full_path = os.path.join(directory, filename)

    return full_path

def create_output_dir(out_dir:str):
    """Creates timestamp-based output directory."""
    # Create a 'results' directory if it doesn't exist
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Create a new directory with a timestamped name inside the 'results' directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    new_directory = os.path.join(out_dir, timestamp)
    os.makedirs(new_directory)

    return new_directory

def turn_index_date(df: pd.DataFrame, ascending_order: bool = True):
    """Turns DataFrame index to Datetime and sort based on dates."""
    # Create Datetime index and scort in asceding order
    df.index = pd.to_datetime(df.index)
    df = df.sort_index(ascending=ascending_order)

    return df

def rename_bbg_columns(df_bbg: pd.DataFrame, new_name: str):
    """Renames columns of risk factor DataFrames extracted from Bloomberg."""
    # Cure dimsinionality issue
    new_cols = [col[0] for col in df_bbg.columns]
    df_bbg.columns = new_cols
    # Provide the actual new column name
    df_bbg.columns = [new_name]

    return df_bbg

def construct_api_key(date_start, date_end, start: str, mid: str, end: str):
    """Constructs API key from Perseo."""
    return start + date_start + mid + date_end + end

def identify_term_structure_names(
        data_rf_dict: dict, curve_identifier: str = "GR",
        mtd_identifier: str = "MtD", ytd_identifier: str = "YtD", 
        da_identifier: str = "DA", spot_identifier: str = "SPOT"):
    """Returns a list with risk factor names solely for risk factors of a certain curve."""
    curve_names_list = []

    for tenor in list(data_rf_dict.keys()):
        if (ytd_identifier not in tenor) and (mtd_identifier not in tenor) and (da_identifier not in tenor) and (spot_identifier not in tenor) and (curve_identifier in tenor):

            curve_names_list.append(tenor)
        
    return curve_names_list

def identify_risk_factor_names(
        data_rf_dict: dict, 
        mtd_identifier: str = "MtD", ytd_identifier: str = "YtD", 
        da_identifier: str = "DA", spot_identifier: str = "SPOT"):
    """Returns a list with risk factor names solely for risk factors and not additional calcualtions (e.g., MtD, YtD, volatilities)."""
    curve_names_list = []

    for tenor in list(data_rf_dict.keys()):
        if (ytd_identifier not in tenor) and (mtd_identifier not in tenor) and (da_identifier not in tenor) and (spot_identifier not in tenor):

            curve_names_list.append(tenor)
        
    return curve_names_list
