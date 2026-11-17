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
from datetime import datetime
import pandas as pd
import numpy as np
from scipy.stats import norm

from tools.general_tools import create_path, create_output_dir

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


class ParametricVaR():
    """Parametric Value-at-Risk (VaR) Model."""

    def __init__(self, desk_name: str = "Energy Trading & Portfolio Management", 
                 drop_message: str = "#N/A Requesting Data...", drop_col: str = "Unnamed: 0"):

        # Path to the current file
        current_file = Path(__file__).resolve()

        # Go one folder up and into RicTool folder
        json_path = current_file.parent.parent

        self.desk_name = desk_name
        self.drop_msg = drop_message
        self.drop_col = drop_col

        # Read config file
        with open(os.path.join(json_path, "config_parametric_var.json"), "r", encoding="utf-8") as file_bbg:
            self.config = json.load(file_bbg)

            # Attributes for data 
            self.output_dir = self.config["outputs_path"]
            self.input_dir = self.config["inputs_path"]
            self.input_dir_var = self.config["inputs_path_var"]
            self.save_outputs = self.config["save_outputs"]
            self.curves = self.config["supported_curves"]

            self.calculation_window = self.config["calculation_window"]
            self.minimum_data = self.config["minimum_required_historical_data"]
            self.ewma_lambda = self.config["ewma"]
            self.confidence_level = self.config["confidence_level"]
            self.holding_period = self.config["holding_period"]
            self.vol_dof = self.config["vol_dof"]

            self.month_last_day_index = self.config["month_last_day_index"]
            self.year_index = self.config["year_index"]

            self.risk_factors = self.config["risk_factors"]

            self.rf_data_dict = None
            self.risk_factors_only_list = None
            self.exposures = None

    def fetch_rf_data(self):
        """Fetches risk factor data from dedicated pickle file."""
        logger.info(f"Fetching risk factor data...")

        data_dict = {}

        for curve in list(self.curves.keys()):
            for tenor in self.curves[curve]:
                logger.info(f"Fetching {tenor}....")

                # Import DataFrame of risk factor
                df_temp = pd.read_excel(
                    create_path(directory=self.input_dir, filename=curve+".xlsx"), sheet_name=tenor)
                
                df_temp = df_temp[df_temp[self.drop_col]!= self.drop_msg]
                # Convert to Datetime Index
                df_temp.set_index(self.drop_col, inplace=True)
                df_temp.index = pd.to_datetime(df_temp.index)
                df_temp.index.name = None
                # Replace BBG ticker with actuar risk factor name
                df_temp.columns = [tenor]
                # Store to a dictionary
                data_dict.update({tenor: df_temp})

                logger.info(f"{tenor} fetched sucessfully!")

        self.rf_data_dict = data_dict.copy()

        logger.info(f"Risk factor data fetched sucessfully!")

        return data_dict

    def construct_outputs_dir(self):
        """Creates the final data output directory per run."""
        final_dir = create_output_dir(out_dir=self.output_dir)
        self.output_dir = final_dir
    
        return final_dir

    def define_risk_factors_list(self, df_values: pd.DataFrame):
        """Defines a list with aliases of utilized risk factors."""
        temp_list = df_values.columns
        self.risk_factors_only_list = temp_list.copy()

        return temp_list

    def define_exposures_per_rf(self, exposure_sheet: str = "exposures", exposure_colname: str = "POS"):
        """Defines a list with aliases of utilized risk factors."""
        # Import DataFrame of risk factors' exposures
        df_temp = pd.read_excel(
            create_path(directory=self.input_dir_var, filename="Exposures.xlsx"), sheet_name=exposure_sheet, index_col=0)
        df_temp = df_temp.dropna()
        # Create dictionary
        exposures = df_temp[exposure_colname].to_dict()

        # Add to class variable
        self.exposures = exposures.copy()
    
        return exposures

    def concat_rf_vales(self):
        """Concatenates risk factor values based on Datetime Index."""
        big_df = pd.concat(self.rf_data_dict.values(), axis=1)
        big_df = big_df.sort_index(ascending=True)
        
        return big_df

    def calculate_critical_value(self):
        """Calculates critical value for provided confidence level."""
        return norm.ppf(1 - self.confidence_level)

    def process_risk_factor_values(self, df_values: pd.DataFrame):
        """Process risk factor values for the Valu-at-Risk calculation."""
        # Keep data after a specific date
        df_rf_values_smooth = df_values[df_values.index.year >= self.year_index] 
        df_rf_values_smooth = df_values[df_values.index.day != df_values.index.days_in_month] 

        # Keep the maximum data window specified in config
        df_rf_values_smooth = df_rf_values_smooth.iloc[-self.calculation_window:, :]

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
        for idx, rf in enumerate(self.risk_factors_only_list):
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
        for idx, rf in enumerate(self.risk_factors_only_list):
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
        
    def calculate_covariance_matrix(self, df_returns: pd.DataFrame):
        """"Calculates Covariance Matrix for given risk factors."""
        logger.info(f"Calculating covariance matix...")
    
        df_cov = df_returns.cov()
    
        logger.info(f"Covariance matrix calculated sucessfully!")
                
        return df_cov

    def map_last_value_per_rf(self, df_1: pd.DataFrame, df_2: pd.DataFrame, mode: str = "VALUE"):
        """"Maps the last field (Value) in each risk factor in the final file."""
        df_1[mode] = df_1['RF'].map(
                    lambda x: df_2[x].iloc[-1] if x in df_2.columns else np.nan)
                    
        return df_1, df_2

    def map_last_vol_per_rf(self, df_1: pd.DataFrame, df_2: pd.DataFrame, mode: str = "VOL"):
        """Maps the last field (Volatility) in each risk factor in the final file."""
        df_1[mode] = df_1['RF'].map(lambda x: df_2[x].iloc[-1] if x in df_2.columns else np.nan)
        
        return df_1, df_2

    def map_exposure_per_rf(self, df_1: pd.DataFrame, mode: str = "POS"):
        """Maps the last field (Exposure / Position) in each risk factor in the final file. Note that missing exposures are converted to 0 exposure!"""
        df_1[mode] = df_1["RF"].map(self.exposures)
    
        return df_1.fillna(0)

    def calculate_portfolio_variance(self, df_info: pd.DataFrame, df_cov: pd.DataFrame,
                                         rf_col: str = "RF", weight_vector_col: str = "WV"):
        """Calculates portfolio variance utilizing Covariance Matrix & Weights Vector."""
        # 1. Turn the weight table into a Series indexed by RF name
        weights = df_info.set_index(rf_col)[weight_vector_col]
    
        # 2. Align covariance matrix rows/columns and weights to a common, identical order. This is the step that actually matters.
        rf_keys = df_cov.index  # assume index == columns already (square, symmetric)
    
        w = weights.reindex(rf_keys)
    
        missing = w[w.isna()]
        if not missing.empty:
            raise ValueError(f"No weight found for risk factors: {missing.index.tolist()}")
    
        w = w.values                                  # shape (n,)
        sigma = df_cov.loc[rf_keys, rf_keys].values  # force same order on both axes
    
        # 3. w' Sigma w
        portfolio_variance = w @ sigma @ w.T
        portfolio_vol = np.sqrt(portfolio_variance)
    
        return {
            "portfolio_variance": float(portfolio_variance),
            "portfolio_volatility": float(portfolio_vol)}

    def orchestrate_calculation(self):
        """Orchestrates process for Parametric VaR calculation."""    
        # Gather required data & configurations
        _ = self.construct_outputs_dir()
        _ = self.fetch_rf_data()
        _ = self.define_exposures_per_rf()
        
        critical_value_var = round(self.calculate_critical_value(), ndigits=3)

        # Concatenate risk factor values into a single DataFrame with dt index
        df_rf_values = self.concat_rf_vales()

        # Define a list of utilzied risk factor aliases
        _ = self.define_risk_factors_list(df_values=df_rf_values)

        # Process risk factor values to perform Value-at-Risk Calcualtion
        df_processed_rf_values = self.process_risk_factor_values(df_values=df_rf_values)

        # Calculate risk factor returns
        df_returns = self.calculate_rf_returns(df_values_processed=df_processed_rf_values)

        # Calculate risk factor returns
        df_volatilities = self.calculate_rf_vols(df_returns=df_returns)

        # Calculate Correlation Matrix
        df_correlation = self.calculate_correlation_table(df_returns=df_returns)

        # Calculate Covariance Matrix
        df_covariance = self.calculate_covariance_matrix(df_returns=df_returns)

        # Create final table with final data & utilized vectors
        df_final_rf_info = pd.DataFrame({"RF": df_processed_rf_values.columns})
        df_final_rf_info, _ = self.map_last_value_per_rf(df_1=df_final_rf_info, df_2=df_processed_rf_values, mode="VALUE")
        df_final_rf_info, _ = self.map_last_vol_per_rf(df_1=df_final_rf_info, df_2=df_volatilities, mode="VOL")
        df_final_rf_info = self.map_exposure_per_rf(df_1=df_final_rf_info, mode="POS")
        df_final_rf_info["WV"] = df_final_rf_info["POS"] * df_final_rf_info["VALUE"]

        # Perform core calculations (Portfolio Variance / Volatility)
        dict_var_port = self.calculate_portfolio_variance(df_info=df_final_rf_info, df_cov=df_covariance)
        portfolio_volatility = dict_var_port["portfolio_volatility"]

        # Calculate VaR & Create final results
        final_var_1d = np.sqrt(1) * critical_value_var * portfolio_volatility
        final_var_xd = np.sqrt(self.holding_period) * critical_value_var * portfolio_volatility

        results = {
            "confidence": self.confidence_level, "holding_period": self.holding_period, "critical_value": critical_value_var, 
            "ewma_lambda": self.ewma_lambda, "portfolio_calc": portfolio_volatility, "parametric_var_1d": final_var_1d, 
            "parametric_var_xd": final_var_xd}

        if self.save_outputs:
            df_rf_values.to_excel(create_path(directory=self.output_dir, filename="Risk Factor Values.xlsx"))
            df_processed_rf_values.to_excel(create_path(directory=self.output_dir, filename="Utilized Risk Factor Values.xlsx"))
            df_returns.to_excel(create_path(directory=self.output_dir, filename="Risk Factor Returns.xlsx"))
            df_volatilities.to_excel(create_path(directory=self.output_dir, filename="Risk Factor Volatility.xlsx"))
            df_correlation.to_excel(create_path(directory=self.output_dir, filename="Correlation Matrix.xlsx"))
            df_covariance.to_excel(create_path(directory=self.output_dir, filename="Covariance Matrix.xlsx"))
            df_final_rf_info.to_excel(create_path(directory=self.output_dir, filename="Final Info Risk Factors.xlsx"))

            with open(create_path(directory=self.output_dir, filename="results.txt"), "w") as f:
                            f.write("Parametric VaR Results " + self.desk_name + "\n")
                            f.write(f"Generated: {datetime.now().isoformat()}\n")
                            f.write("-" * 40 + "\n")
                            for key, value in results.items():
                                f.write(f"{key}: {value}\n")
            

        return df_rf_values
        

if __name__ == "__main__":

    desk_name = "Energy Trading & Portfolio Management"
    desk_name = "Gas Supply & Trading"

    cl1 = ParametricVaR(desk_name=desk_name)

    df_values = cl1.orchestrate_calculation()