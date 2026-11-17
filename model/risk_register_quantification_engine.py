import warnings
warnings.filterwarnings('ignore')

# Enable import of other modules
import sys
ROOT = r"C:\Users\g.varvounis\Documents\RicTool"
if ROOT not in sys.path:
    sys.path.append(ROOT)
###############################

import pandas as pd
import json
from pathlib import Path
import os
import numpy as np
from scipy import stats as st
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


class RiskRegisterSimulation:

    def __init__(self):
        
        # Path to the current file
        current_file = Path(__file__).resolve()

        # Go one folder up and into RicTool folder
        json_path = current_file.parent.parent

        # Read config file
        with open(os.path.join(json_path, "config_risk_register_quantification.json"), "r", encoding="utf-8") as file_bbg:
            self.config = json.load(file_bbg)

        
        self.outputs_dir = self.config["outputs_path"]
        self.save_outputs = self.config["save_outputs_to_path"]

        self.risk_register_dir = self.config["risk_register_dir"]
        self.risk_register_filename = self.config["risk_register_filename"]
        self.risk_register_sheet = self.config["risk_register_sheet_name"]
        self.risk_id_col = self.config["risk_id_col_in_risk_register_lite"]
        self.risk_title_col = self.config["risk_title_col_in_risk_register"]
        self.risk_tax_i_col = self.config["risk_taxonomy_i_col_in_risk_register"]
        self.risk_tax_ii_col = self.config["risk_taxonomy_ii_col_in_risk_register"]

        self.independent_risks = self.config["independent_risks"]
        self.bernoulli_sampling = self.config["bernoulli_sampling"]
        self.horizon = self.config["horizon_years"]
        self.num_steps = self.config["num_steps"]
        self.num_scenarios = self.config["num_scenarios"]
        self.use_cap = self.config["use_cap"]
        self.cap_value = self.config["cap_value"]
        self.replication_id = self.config["replication_id"]

        self.df_rr_lite = None


    def read_risk_register(self):
        """Imports Risk Register information."""

        logger.info(f"Importing Risk Register Lite ...")

        df_rr = pd.read_excel(
            create_path(directory=self.risk_register_dir, filename=self.risk_register_filename), 
            sheet_name=self.risk_register_sheet)

        self.df_rr_lite = df_rr.copy()

        logger.info(f"Risk Register Lite imported sucessfully!")

        return df_rr
    
    def cap_impact(self, data):
        """Cap impat values to a pre-defined maximum value (optional)."""
        return np.minimum(data, self.cap_value)
    
    def define_unique_risk_factors(self):
        """Extracts & Returns the Unique IDs of risk factors."""

        return self.df_rr_lite[self.risk_id_col].values.tolist()
    
    def sample_bernoulli(self, risk_likelihood: float):
        """Samples from a Bernoulli distribution with probability of sucess equal to risk likelihood."""

        sample = np.random.binomial(n=1, p=risk_likelihood, size=self.num_scenarios)

        return sample
    
    def sample_lognormal(self, right_tail_impact:float, left_tail_impact: float):
        """Samples from a Log-normal dsitribution for a certain risk factor."""

        estimated_mean = (np.log(left_tail_impact) + np.log(right_tail_impact)) / 2
        estimated_sigma = (np.log(right_tail_impact) - np.log(left_tail_impact)) / 3.29

        data = np.random.lognormal(mean=estimated_mean, sigma=estimated_sigma, size=self.num_scenarios)

        return data
    
    def sample_uniform(self, max_impact:float, min_impact: float):
        """Samples from a Log-normal dsitribution for a certain risk factor."""

        data = np.random.uniform(low=min_impact, high=max_impact, size=self.num_scenarios)

        return data
    
    def sample_normal(self, mean:float, std: float, num_scenarios: int):
        """Samples from a Normal dsitribution for a certain risk factor."""

        data = np.random.normal(loc=mean, scale=std, size=num_scenarios)

        return data

    
    def bernoulli_sample_impact_per_risk(
            self, risk_id: int, risk_likelihood: float, distribution: str, 
            mean_rf: float, std_rf: float, max_impact: float, min_impact: float,
            seed: int):
        """Samples the impact of a risk factor from a given distribution utilizing Bernoulli sampling."""

        logger.info(f"Simulating impact of {risk_id} ... ")

        # Initalize DataFrame for each risk factor to store simulation results {num steps x num scenarios}
        df_risk_factor_sample = pd.DataFrame(columns=range(self.num_scenarios))
        df_risk_factor_sample_realization = df_risk_factor_sample.copy()

        for i in range(0, self.horizon):

            # Set a random seed to replicate identical results
            np.random.seed(seed=seed + risk_id + i)

            # Initialize an empty series for sample per step
            data = pd.Series()

            # Define whether risk materializes from a Bernoulli dist. with prob.of success equal to risk likelihood
            sample = self.sample_bernoulli(risk_likelihood=risk_likelihood)

            if distribution.lower() == "normal":
                data = self.sample_normal(mean=mean_rf, std=std_rf)
            elif distribution.lower() == "lognormal":
                data = self.sample_lognormal(right_tail_impact=max_impact, left_tail_impact=min_impact)
            elif distribution.lower() == "uniform":
                data = np.random.uniform(low=min_impact, high=max_impact)
            else:
                logger.info(f"Dstribution {distribution} not supported!")
                sys.exit(1)
            
            if self.independent_risks:
                data = data * sample

            # Cap impact value to the pre-defined maximum value (optional)
            if self.use_cap:
                data = self.cap_impact(data=data)

            # Strore Impact sampls as well as relaization (bernoulli) samples
            df_risk_factor_sample.loc[len(df_risk_factor_sample)] = data

            # In case of many steps (horizon <>1) at each step (per scenario - column) calculate cumsum of simulated impacts
            df_risk_factor_sample = df_risk_factor_sample.cumsum()

            df_risk_factor_sample_realization.loc[len(df_risk_factor_sample_realization)] = sample
        
        logger.info(f"Simulation of {risk_id} completed sucessfully!!")

        return df_risk_factor_sample, df_risk_factor_sample_realization
    
    def calculate_meta_statistics(self, df_total_results, dict_results):
        """Calculates statistics after the Risk Register simulation is conducted."""

        # 1 - Per risk factor statistics
        # Create a list containing risk factors
        risk_factors_list = list(dict_results.keys())

        # Initate a pandas.dataframe to store simulation at horizon results
        df_horizon_values_per_rf = pd.DataFrame()

        # Create a list to store risk taxonomies level I
        risk_taxonomies_i = []

        # Create a list to store risk taxonomies level II
        risk_taxonomies_ii = []

        # Create a list to store risk factor titles
        rf_titles = []

        # create a list to store per rf average impacts at horizon
        per_rf_horizon_impact_list = []

        for risk in risk_factors_list:
            # Store horizon results of impacts for the rf in a pandas.dataframe
            simulation_at_horizon_results = dict_results[str(risk)].iloc[-1, :].copy()
            simulation_at_horizon_results = pd.DataFrame([simulation_at_horizon_results])

            # Update overal pandas.dataframe with horizon results for the new risk factor
            df_horizon_values_per_rf = pd.concat(
                [df_horizon_values_per_rf, simulation_at_horizon_results], axis=0, ignore_index=True)
            
            # Store Average Impact at Horizon per rf in a list
            per_rf_horizon_impact_list.append(simulation_at_horizon_results.mean(axis=1)[0])     
            
            # Store in a list risk taxonomies information & Titles of Risk Factors
            rf_titles.append(self.df_rr_lite[self.risk_title_col][self.df_rr_lite[self.risk_id_col] == int(risk)].values[0])
            risk_taxonomies_i.append(self.df_rr_lite[self.risk_tax_i_col][self.df_rr_lite[self.risk_id_col] == int(risk)].values[0])
            risk_taxonomies_ii.append(self.df_rr_lite[self.risk_tax_ii_col][self.df_rr_lite[self.risk_id_col] == int(risk)].values[0])

        # Update pandas.dataframe of overal results per rf at horizon with risk taxonomies and risk title information
        df_horizon_values_per_rf[self.risk_tax_i_col] = risk_taxonomies_i.copy()
        df_horizon_values_per_rf[self.risk_tax_ii_col] = risk_taxonomies_ii.copy()
        df_horizon_values_per_rf.index = risk_factors_list.copy()
        df_horizon_values_per_rf[self.risk_title_col] = rf_titles.copy()
        df_horizon_values_per_rf['Average Horizon Impact'] = per_rf_horizon_impact_list.copy()
        # Restrict columns (exclude per scenario results for reporting purposes)
        df_horizon_values_per_rf = df_horizon_values_per_rf[
            [self.risk_title_col, self.risk_tax_i_col, self.risk_tax_ii_col, 'Average Horizon Impact']]
        ##################################################################

        # 2 - Total Impact Statistics
        df_total_results = df_total_results.iloc[-1,:].copy()

        per_90 = np.percentile(df_total_results.T, 90)
        per_95 = np.percentile(df_total_results.T, 95)
        per_99 = np.percentile(df_total_results.T, 99)
        per_99999 = np.percentile(df_total_results.T, 99.999)

        es_90 = np.mean(df_total_results.T[
                        df_total_results.T >= np.percentile(df_total_results.T, 90)])
        es_95 = np.mean(df_total_results.T[
                        df_total_results.T >= np.percentile(df_total_results.T, 95)])
        es_99 = np.mean(df_total_results.T[
                        df_total_results.T >= np.percentile(df_total_results.T, 99)])

        average_value = np.mean(df_total_results.T)
        median_value = np.median(df_total_results.T)
        mode_value = st.mode(df_total_results.T)[0]

        horizon_stats = {
            "90%-Percentile Impact": per_90, "95%-Percentile Impact": per_95,
            "99%-Percentile Impact": per_99, "99.999%-Percentile Impact": per_99999,
            "90% ES": es_90, "95% ES": es_95, "99% ES": es_99,
            "Expected Impact": average_value, "Median Impact": median_value, "Mode Impact": mode_value}
        
        df_impact_total_at_horizon_metrics = pd.DataFrame(horizon_stats, index=["Horizon"])

        # Re-arange columns for reporting purposes
        df_impact_total_at_horizon_metrics = df_impact_total_at_horizon_metrics[
            ["99.999%-Percentile Impact", "99%-Percentile Impact", "99% ES",
             "95%-Percentile Impact", "95% ES", "90%-Percentile Impact", "90% ES",
             "Expected Impact", "Median Impact", "Mode Impact"]]

        if self.save_outputs:
            df_horizon_values_per_rf.to_excel(create_path(directory=self.outputs_dir, filename="Horizon Values per rf.xlsx"))
            df_impact_total_at_horizon_metrics.to_excel(create_path(directory=self.outputs_dir, filename="Horizon Total impact Metrics.xlsx"))

        return df_horizon_values_per_rf, df_impact_total_at_horizon_metrics

    def perform_simulation(self):
        """Performs simulation for all risk factors."""

        logger.info(f"Simulation of Risk Register initiated ...\n")

        # Import Risk Register Lite
        _ = self.read_risk_register()

        # Create a list containing risk factors
        rf_list = self.define_unique_risk_factors()

        # Create final output dir
        time_dir = create_output_dir(out_dir=self.outputs_dir)
        self.outputs_dir = time_dir

        # Create a pandas.dataframe object to store results for all risk factors
        df_total_results = pd.DataFrame(np.zeros((self.num_steps, self.num_scenarios)))

        # Create a dictionary to store per risk factor results
        dict_per_rf_results = {}

        for risk in rf_list:

            # Define parameters of each risk factor / risk in the risk register
            likelihood_value = self.df_rr_lite['Converted Likelihood'][self.df_rr_lite[self.risk_id_col] == risk].values[0]
            min_impact_value = self.df_rr_lite['Converted Lower Impact'][self.df_rr_lite[self.risk_id_col] == risk].values[0]
            max_impact_value = self.df_rr_lite['Converted Max Impact'][self.df_rr_lite[self.risk_id_col] == risk].values[0]
            distribution_value = self.df_rr_lite['Distribution'][self.df_rr_lite[self.risk_id_col] == risk].values[0]
            mean_value = self.df_rr_lite['Mean'][self.df_rr_lite[self.risk_id_col] == risk].values[0]
            std_value = self.df_rr_lite['Std'][self.df_rr_lite[self.risk_id_col] == risk].values[0]
            # Placeholder(s): values for mean and median are not yet utilized
            mode_value = 0
            median_value = 0

            try:
                if self.bernoulli_sampling:
                    df_risk_results, df_risk_materialization = self.bernoulli_sample_impact_per_risk(
                        risk_id=risk, risk_likelihood=likelihood_value, distribution=distribution_value,
                        mean_rf=mean_value, std_rf=std_value, max_impact=max_impact_value, min_impact=min_impact_value,
                        seed=self.replication_id)
                    
                    # Update total simulation results with just simulated
                    df_total_results = df_total_results + df_risk_results

                    # Update Dictionary with results per risk factor
                    dict_per_rf_results[str(risk)] = df_risk_results.copy()

                    if self.save_outputs:
                        temp_alias_results = "simulation_of_impacts_risk_" + str(risk) + ".xlsx"
                        temp_alias_materialization = "materialization_risk_" + str(risk) + ".xlsx"

                        df_risk_results.to_excel(create_path(directory=self.outputs_dir, filename=temp_alias_results))
                        df_risk_materialization.to_excel(create_path(directory=self.outputs_dir, filename=temp_alias_materialization))
                else:
                    logger.info(f"Possion sampling mode not yet supported!")
                    sys.exit(1)
            except:
                logger.info(f"Sampling cannot be performed!")
                sys.exit(1)

        if self.save_outputs:
            temp_alias_total_results = "Total Simulated Impacts.xlsx"
            df_total_results.to_excel(create_path(directory=self.outputs_dir, filename=temp_alias_total_results))

        _, _ = self.calculate_meta_statistics(df_total_results=df_total_results, dict_results=dict_per_rf_results)
        
        logger.info(f"Simulation of Risk Register completed sucessfully! \n")
        
        return df_total_results, dict_per_rf_results


if __name__ == "__main__":

    class1 = RiskRegisterSimulation()

    class1.perform_simulation()

  