import warnings
warnings.filterwarnings('ignore')

# Enable import of other modules
import sys
ROOT = r"C:\Users\g.varvounis\Documents\RiC"
if ROOT not in sys.path:
    sys.path.append(ROOT)
###############################
from pathlib import Path
import json
import os
import pickle
import pandas as pd
import numpy as np

from tools.general_tools import identify_risk_factor_names, create_path, create_output_dir

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


class StressPeriodIdentifier():
    """Identifies a stress period for a certain curve."""

    def __init__(self, curve: str = "GR Curve", data_filename: str = "data_risk_factors.pkl"):

        # Path to the current file
        current_file = Path(__file__).resolve()

        # Go one folder up and into RicTool folder
        json_path = current_file.parent.parent

        self.data_filename = data_filename
        self.curve = curve

        # Read config file
        with open(os.path.join(json_path, "config_stress_engine.json"), "r", encoding="utf-8") as file_bbg:
            self.config = json.load(file_bbg)

            # Attributes for data 
            self.output_dir = self.config["outputs_path"]
            self.save_outputs = self.config["save_outputs"]

            self.minimum_data = self.config["minimum_required_historical_data"]
    
            self.vol_dof = self.config["vol_dof"]
            self.level = self.config["level"]
            self.window_size = self.config["stress_window_size"]

            self.year_index = self.config["year_index"]
            # self.stress_start = self.config["stress_date_start"]
            # self.stress_end = self.config["stress_date_end"]

            self.risk_factors = self.config["risk_factors"][curve]
            self.risk_factors_only = None

            self.rf_data_dict = None

    def fetch_rf_data(self):
        """Fetches risk factor data from dedicated pickle file."""
        logger.info(f"Fetching risk factor data...")
        try:
            with open(self.data_filename, "rb") as f:
                data_dict = pickle.load(f)
            logger.info(f"Risk Factor data fetched sucessfully!")
        except:
            logger.info(f"Error in fetching risk factor data!")

        self.rf_data_dict = data_dict.copy()
        
        return data_dict

    def construct_outputs_dir(self):
        """Creates the final data output directory per run."""
        final_dir = create_output_dir(out_dir=self.output_dir)
        self.output_dir = final_dir
    
        return final_dir

    def keep_only_rf_keys(self):
        """From Data dictionary keep only data related to risk factors and not additional data (e.g., MtD calculations)."""
        #rf_names_list = identify_risk_factor_names(data_rf_dict=self.rf_data_dict)
        rf_names_list = self.risk_factors.copy()
        self.risk_factors_only = rf_names_list.copy()
    
        return rf_names_list

    def process_risk_factor_values(self, df_values: pd.DataFrame, utilize_specific_stress_period: bool = True):
        """Process risk factor values for the Valu-at-Risk calculation."""
        if utilize_specific_stress_period:
            df_rf_values_smooth = df_values.copy()  
        else:
            # Keep data after a specific date
            df_rf_values_smooth = df_values[df_values.index.year >= self.year_index]
            df_rf_values_smooth = df_rf_values_smooth[df_rf_values_smooth.index.day != df_rf_values_smooth.index.days_in_month]
                
        return df_rf_values_smooth

    def normalize_returns(self, df_returns: pd.DataFrame, max_gap_days: int = 3):
        """Normalizes returns by dividing with sqrt of day difference between sequential returns to ensure / approximate 1-day returns."""
        D = df_returns.index.to_series().diff().dt.days
            
        df_returns = df_returns.div(np.sqrt(D), axis=0)
    
        # Drop rows where the gap is too large to be a real trading-day return
        df_returns = df_returns.loc[D <= max_gap_days]
            
        return df_returns

    def calculate_rf_returns(self, df_values_processed: pd.DataFrame):
        """Calculates risk factor returns."""
        logger.info(f"Calculating returns...")

        # Calculate risk factor returns
        for idx, rf in enumerate(self.risk_factors_only):
            if idx==0:
                df_rf_returns = pd.DataFrame({rf: np.log(df_values_processed[rf] / df_values_processed[rf].shift(1))})
            else:
                df_rf_returns[rf] = np.log(df_values_processed[rf]/ df_values_processed[rf].shift(1))
                
        # Normalize returns to reflect 1-day gap
        df_rf_returns = self.normalize_returns(df_returns=df_rf_returns)

        logger.info(f"Returns calculated sucessfully!")

        return df_rf_returns

    import pandas as pd

    def get_metric_extreme_ranges(self,
            df: pd.DataFrame, metric_column: str, risk_factor_column: str = "Risk Factor",
            start_date_column:str ="Start Date", end_date_column:str ="End Date"):
        """Returns min / max date range for a specific metric under each rf."""
        required_columns = {risk_factor_column, start_date_column, end_date_column, metric_column}

        missing_columns = required_columns.difference(df.columns)

        if missing_columns:
            raise ValueError(
                f"Missing columns: {sorted(missing_columns)}")

        data = df.copy()

        data[start_date_column] = pd.to_datetime(data[start_date_column], errors="coerce")

        data[end_date_column] = pd.to_datetime(data[end_date_column], errors="coerce")

        data[metric_column] = pd.to_numeric(data[metric_column], errors="coerce")

        result = {}

        for risk_factor, group in data.groupby(risk_factor_column, sort=False):
            # Keep only rows with valid metric and date values
            valid_group = group.dropna(
                subset=[metric_column, start_date_column, end_date_column])

            if valid_group.empty:
                result[str(risk_factor)] = {"maximum": None, "minimum": None}
                continue

            maximum_row = valid_group.loc[valid_group[metric_column].idxmax()]

            minimum_row = valid_group.loc[valid_group[metric_column].idxmin()]

            result[str(risk_factor)] = {
                "maximum": {"start_date": maximum_row[start_date_column].strftime("%Y-%m-%d"), 
                            "end_date": maximum_row[end_date_column].strftime("%Y-%m-%d"), 
                            "value": float(maximum_row[metric_column])},
                "minimum": {"start_date": minimum_row[start_date_column].strftime("%Y-%m-%d"),
                            "end_date": minimum_row[end_date_column].strftime("%Y-%m-%d"),
                            "value": float(minimum_row[metric_column])}}

        return result

    def calculate_window_metrics(self, window_returns):
        """Calculates utilized metrics per rolling window of the available time period."""
        window_returns = pd.Series(window_returns).dropna()

        #if len(window_returns) < self.minimum_data:
        if len(window_returns) < self.window_size /2:
            return pd.Series({
                "Cumulative Return": np.nan,"Annualized Volatility": np.nan, "Expected Shortfall": np.nan, "Maximum Drawdown": np.nan})
        else:
            # Cumulative return from log-returns
            cumulative_return = (np.exp(window_returns.sum())-1)

            # Annualized volatility
            annualized_volatility = (window_returns.std(ddof=self.vol_dof) * np.sqrt(self.window_size))

            # Expected Shortfall
            alpha = 1 - self.level
            var_threshold = window_returns.quantile(alpha)
            tail_returns = window_returns[window_returns <= var_threshold]
            expected_shortfall = (tail_returns.mean())

            # Maximum Drawdown
            cumulative_index = np.exp(window_returns.cumsum())
            running_maximum = cumulative_index.cummax()
            drawdown = (cumulative_index / running_maximum) - 1
            maximum_drawdown = drawdown.min()

            return pd.Series({
                "Cumulative Return": cumulative_return, "Annualized Volatility": annualized_volatility, 
                "Expected Shortfall": expected_shortfall, "Maximum Drawdown": maximum_drawdown})


    def orchestrate_calculation(self):
        """Orchestrates process for Parametric VaR calculation."""    
        # Gather required data & configurations
        _ = self.construct_outputs_dir()
        _ = self.fetch_rf_data()
        _ = self.keep_only_rf_keys()
        
        # Combine & Align risk factor values
        for idx, rf in enumerate(self.risk_factors_only):
            if idx==0:
                df_rf_values = self.rf_data_dict[rf][rf]
            else:
                df_rf_values_temp = self.rf_data_dict[rf][rf]
                df_rf_values = pd.concat([df_rf_values, df_rf_values_temp], axis=1)

        # Process risk factor values to perform Value-at-Risk Calcualtion
        df_processed_rf_values = self.process_risk_factor_values(df_values=df_rf_values)

        # Calculate risk factor returns
        df_returns = self.calculate_rf_returns(df_values_processed=df_processed_rf_values)

        # Option 1: Equal-weighted average return
        df_returns["Synthetic Risk Factor"] = df_returns.mean(axis=1,skipna=False)


        all_window_results = []
        for rf in pd.DataFrame(df_returns["Synthetic Risk Factor"]).columns:
            risk_factor_series = df_returns[rf]
            for start_pos in range(len(risk_factor_series) - self.window_size + 1):
                end_pos = start_pos + self.window_size
                window_returns = risk_factor_series.iloc[start_pos:end_pos]
                metrics = self.calculate_window_metrics(window_returns)
                all_window_results.append({"Risk Factor": rf,
                                           "Start Date": window_returns.index[0],
                                           "End Date": window_returns.index[-1], **metrics.to_dict()})
        window_metrics_df = pd.DataFrame(all_window_results)

        result_stress_periods = self.get_metric_extreme_ranges(df=window_metrics_df, metric_column="Annualized Volatility")
        
        if self.save_outputs:
            df_rf_values.to_excel(create_path(directory=self.output_dir, filename="Risk Factor Values.xlsx"))
            df_processed_rf_values.to_excel(create_path(directory=self.output_dir, filename="Utilized Risk Factor Values.xlsx"))
            df_returns.to_excel(create_path(directory=self.output_dir, filename="Risk Factor Returns.xlsx"))
            window_metrics_df.to_excel(create_path(directory=self.output_dir, filename="Stress Periods Result.xlsx"))
            txt_path = create_path(directory=self.output_dir, filename="risk_factor_ranges.txt")
            with open(txt_path, "w", encoding="utf-8") as file_sp:
                json.dump(result_stress_periods, file_sp,indent=4,ensure_ascii=False)
            
        return df_rf_values
        

if __name__ == "__main__":

    cl1 = StressPeriodIdentifier()

    df_values = cl1.orchestrate_calculation()