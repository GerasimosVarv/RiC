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


class CurveStressEngine():
    """Stresses a Curve."""

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

            self.year_index = self.config["year_index"]
            self.stress_start = self.config["stress_date_start"]
            self.stress_end = self.config["stress_date_end"]

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

    def process_risk_factor_values(self, df_values: pd.DataFrame, utilize_specific_stress_period: bool = False):
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

    def calculate_rf_vols(self, df_returns: pd.DataFrame):
        """Calculates risk factor volatilities."""
        logger.info(f"Calculating volatilities...")

        # Calculate risk factor volatilities
        for idx, rf in enumerate(self.risk_factors_only):
            if idx==0:
                df_rf_vols = pd.DataFrame({rf: df_returns[rf].std(ddof=self.vol_dof)}, index=[0])
            else:
                df_rf_vols[rf] = df_returns[rf].std(ddof=self.vol_dof)

        logger.info(f"Volatilities calculated sucessfully!")
        
        return df_rf_vols

    def calculate_correlation_table(self, df_returns: pd.DataFrame):
        """Calculates Correlation Matrix for given risk factors."""
        logger.info(f"Calculating correlations' matix...")
            
        df_corr = df_returns.corr()
    
        logger.info(f"Correlations' matrix calculated sucessfully!")
    
        return df_corr
        
    def calcualte_covariance_matrix(self, df_returns: pd.DataFrame):
        """"Calculates Covariance Matrix for given risk factors."""
        logger.info(f"Calculating covariance matix...")
    
        df_cov = df_returns.cov()
    
        logger.info(f"Covariance matrix calculated sucessfully!")
                
        return df_cov

    def create_curve(self, values_df: pd.DataFrame):
        """Create the curve as of today for the selected curve."""
        df_temp_curve = values_df.iloc[[-1]].T
        df_temp_curve.columns = ["Value"]

        return df_temp_curve

    def calculate_shocks_per_rf(self, returns_df: pd.DataFrame, level: float = 0.95):
        """Calculate shocks (up and down) per risk factor."""
        # Percentile Shocks
        df_schocks = pd.DataFrame({
            "shock_down": returns_df.quantile(1 - level), "shock_up": returns_df.quantile(level)
        })

        # ES shocks
        es_shock_down = returns_df.where(returns_df.le(returns_df.quantile(1 - level), axis="columns")).mean()
        es_shock_up = returns_df.where(returns_df.ge(returns_df.quantile(level), axis="columns")).mean()

        df_schocks["ES_shock_down"] = es_shock_down
        df_schocks["ES_shock_up"] = es_shock_up

        return df_schocks

    def apply_shock_per_rf(self, curve_df: pd.DataFrame, shocks_df: pd.DataFrame, use_log_rets: bool = True):
        """Applies calculated shocks per rf current curve."""
        df_result = curve_df.join(shocks_df)

        if use_log_rets:
            # Apply Percentile shocks
            df_result["shocked_up"] = (df_result["Value"] * np.exp(df_result["shock_up"]))
            df_result["shocked_down"] = (df_result["Value"] * np.exp(df_result["shock_down"]))

            # Apply ES shocks
            df_result["ES_shocked_up"] = (df_result["Value"] * np.exp(df_result["ES_shock_up"]))
            df_result["ES_shocked_down"] = (df_result["Value"] * np.exp(df_result["ES_shock_down"]))
        else:
            # Apply Percentile shocks
            df_result["shocked_up"] = (df_result["Value"] * (1 + df_result["shock_up"]))
            df_result["shocked_down"] = (df_result["Value"] * (1 + df_result["shock_down"]))
            
            # Apply ES shocks
            df_result["ES_shocked_up"] = (df_result["Value"] * (1 + df_result["ES_shock_up"]))
            df_result["ES_shocked_down"] = (df_result["Value"] * (1 + df_result["ES_shock_down"]))

        return df_result


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

        # Calculate risk factor returns
        df_volatilities = self.calculate_rf_vols(df_returns=df_returns)

        # Calculate Correlation Matrix
        df_correlation = self.calculate_correlation_table(df_returns=df_returns)

        # Calculate Covariance Matrix
        df_covariance = self.calcualte_covariance_matrix(df_returns=df_returns)

        # Extract curve as of the last closure in markets
        df_curve_shape = self.create_curve(values_df=df_rf_values)

        # Calculate Shocks as percentiles of returns' distribution per rf
        df_cs = self.calculate_shocks_per_rf(returns_df=df_returns, level=self.level)

        # Shock the latest curve
        df_shocked_curve = self.apply_shock_per_rf(curve_df=df_curve_shape, shocks_df=df_cs)
        
        if self.save_outputs:
            df_rf_values.to_excel(create_path(directory=self.output_dir, filename="Risk Factor Values.xlsx"))
            df_processed_rf_values.to_excel(create_path(directory=self.output_dir, filename="Utilized Risk Factor Values.xlsx"))
            df_returns.to_excel(create_path(directory=self.output_dir, filename="Risk Factor Returns.xlsx"))
            df_volatilities.to_excel(create_path(directory=self.output_dir, filename="Risk Factor Volatility.xlsx"))
            df_correlation.to_excel(create_path(directory=self.output_dir, filename="Correlation Matrix.xlsx"))
            df_covariance.to_excel(create_path(directory=self.output_dir, filename="Covariance Matrix.xlsx"))
            df_curve_shape.to_excel(create_path(directory=self.output_dir, filename=self.curve +".xlsx"))
            df_cs.to_excel(create_path(directory=self.output_dir, filename=self.curve +"_Shocks_" + str(self.level) + ".xlsx"))
            df_shocked_curve.to_excel(create_path(directory=self.output_dir, filename=self.curve +"_Shocked_Curve.xlsx"))
            
        return df_rf_values
        

if __name__ == "__main__":

    cl1 = CurveStressEngine()

    df_values = cl1.orchestrate_calculation()