import warnings
warnings.filterwarnings('ignore')

# Enable import of other modules
import sys
ROOT = r"C:\Users\g.varvounis\Documents\RiC"
if ROOT not in sys.path:
    sys.path.append(ROOT)
###############################
from tools.general_tools import create_path, create_output_dir
from datetime import datetime

import os
import numpy as np
import pandas as pd
from scipy.stats import norm
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

class BachelierPricer():
    """CSS Option Strip Pricer - Bachelier (Nomral) Model."""

    def __init__(
            self, heat_rate, emission_factor, strike_price, premium_per_option, mw_per_option_per_hour,
            save_results: bool=True, output_dir: str = r"C:\\Users\\g.varvounis\\Documents\\RiC\\_other_outputs\\_options"):
        
        self.save_results = save_results
        self.output_dir = output_dir

        self.heat_rate = heat_rate
        self.emission_factor = emission_factor
        self.strike_price = strike_price
        self.premium_per_option = premium_per_option
        self.mw_per_option_per_hour = mw_per_option_per_hour

    def construct_outputs_dir(self):
        """Creates the final data output directory per run."""
        final_dir = create_output_dir(out_dir=self.output_dir)
        self.output_dir = final_dir
            
        return final_dir

    def calculate_css(self, power, gas, eua):
        """Calculates Clean Spark Spread (CSS)."""
        return power - self.heat_rate * gas - self.emission_factor * eua

    def price_bachelier_call(self, F, sigma, T, r):
        """
        Price a European Call Option under the Bachelier (normal) model.

        Params:
        ----------
            - F [EUR/MWh]: forward value of the underlying (e.g., fwd prices' CSS)
            - sigma [EUR/MWh]: absolute ("normal") volatility of the underlying
            - T [Years]: Time to expiry in years
            - r: continuously compounded discount rate (annual)

        Returns:
        ----------
         - Discounted option value [EUR / MWh] for a certain European Option in the strip
        """
        logger.info(f"Pricing European option(s) of strip...")

        # Convert input values to NumPy arrays
        F, sigma, T = np.asarray(F, dtype=float), np.asarray(sigma, dtype=float), np.asarray(T, dtype=float)
        K = np.asarray(self.strike_price, dtype=float)

        # Calculate d factor
        d = np.where(sigma * np.sqrt(T) > 0, (F - K) / (sigma * np.sqrt(T)), np.inf * np.sign(F - K))

        # Calculate unidiscounted European option value
        und_price = (F - K) * norm.cdf(d) + sigma * np.sqrt(T) * norm.pdf(d)

        # Ensure T=0 or sigma=0 constraints: Payoff collapses to the intrinsic value max(F-K,0)
        intrinsic_price = np.maximum(F - K, 0)
        und_price = np.where((sigma * np.sqrt(T)) > 0, und_price, intrinsic_price)
        price = np.exp(-r * T) * und_price

        logger.info(f"European Option(S) of strip priced sucesfully!\n")

        return price

    def combine_rf_volatilities(self, sigma_power, sigma_gas, sigma_eua, rho_pg, rho_pe, rho_ge):
        """
        Combines the volatility of Power, Gas, EUA to a single one based on correlations of risk factors.

        (sigma_css)^2 = (sigma_p)^2 + (hr)^2 (sigma_gas)^2 + (ef)^2 (sigma_eua)^2
                        - 2 (hr) (rho_pg) (sigma_power) (sigma_gas)
                        - 2 (ef) (rho_pe) (sigma_power) (sigma_eua)
                        + 2 (hr) (ef) (rho_ge) (sigma_gas) (sigma_eua) 

        """
        logger.info(f"Calculating aggregated volatility of CSS...")

        hr = self.heat_rate
        ef = self.emission_factor

        vol_sq = (
            sigma_power**2 + (hr * sigma_gas)**2 + (ef * sigma_eua)**2
            - 2 * hr * rho_pg * sigma_power * sigma_gas
            - 2 * ef * rho_pe * sigma_power * sigma_eua
            + 2 * ef * hr * rho_ge * sigma_gas * sigma_eua)

        vol = np.sqrt(vol_sq)

        logger.info(f"Aggregated volaitlity for CSS calculated sucesfully!!")
        
        return vol

    def price_strip_of_options(
              self, df, valuation_date, r, 
              volume_col: str = "volume_mw", dt_hours: float = 1.0, value_correction: bool = True):
        """
        Prices the entire strip of European Options on CSS.

        Note
        ------
        df must contain:
            - "delivery_hour"
            - "css_fwd"
            - "sigma"
            - "volume_mw"
        """
        logger.info(f"Pricing Strip Option....\n")

        df_out = df.copy()
        df_out["T"] = (df_out["delivery_hour"] - valuation_date).dt.total_seconds() / (365 * 24 * 3600)
        df_out["option_value"] = self.price_bachelier_call(F=df_out["css_fwd"], sigma=df_out["sigma"], T=df_out["T"], r=r)
        if value_correction:
             df_out["option_value"] = np.where(df_out["css_fwd"] >= 0, df_out["option_value"], 0)
        df_out["premium_contribution"] = df_out["option_value"] * df_out[volume_col] * dt_hours

        total_premium = df_out["premium_contribution"].sum()
        total_mwh = (df_out[volume_col] * dt_hours).sum()
        strip_premium_per_mwh = total_premium / total_mwh

        df_out["in_the_money"] = np.where(df_out["css_fwd"] > 0, 1, 0)

        summary_results = {
            "total_premium_sum": total_premium, "total_mwh_sum": total_mwh, 
            "strip_premium_eur_per_mwh": strip_premium_per_mwh, "n_hours": len(df_out)}

        if self.save_results:
            df_out.to_excel(create_path(directory=self.output_dir, filename="Future_Output_Details.xlsx"))
            with open(create_path(
                  directory=self.output_dir, filename="future_results.txt"), "w") as f:
                            f.write("Bachelier Pricer "  + "\n")
                            f.write(f"Generated: {datetime.now().isoformat()}\n")
                            f.write("-" * 40 + "\n")
                            for key, value in summary_results.items():
                                f.write(f"{key}: {value}\n")

        logger.info(f"Strip option priced sucesfully!!\n")

        return df_out, summary_results

    def calculate_realized_pnl(self,df: pd.DataFrame, dt_hours: float = 1.0, volume_col: str = "volume_mw"):
        """
        Calculates realized P&L so far.

         Note
        ------
        df must contain:
            - "delivery_hour"
            - "css_fwd" -> despite tile, it refers to actual (realized CSS per past hour)
            - "sigma"
            - "volume_mw"
        """
        logger.info(f"Calcualting P&L from expired options of the strip...")

        df_out = df.copy()
        #df_out["pnl"] = min(self.premium_per_option, self.premium_per_option - df_out["css_fwd"])
        df_out["pnl"] = self.premium_per_option - df_out["css_fwd"]
        df_out.loc[df_out["css_fwd"] <0, "pnl"] = self.premium_per_option
        df_out["total_pnl"] = df_out["pnl"] * df_out[volume_col]

        total_mwh = (df_out[volume_col] * dt_hours).sum()

        realized_pnl =  df_out["total_pnl"].sum()

        df_out["in_the_money"] = np.where(df_out["css_fwd"] > 0, 1, 0)

        summary_results = {
                "realized_pnl": realized_pnl, "total_mwh_sum": total_mwh, "n_hours": len(df_out)}

        if self.save_results:
             df_out.to_excel(create_path(directory=self.output_dir, filename="Past_Output_Details.xlsx"))
             with open(create_path(
                            directory=self.output_dir, filename="past_results.txt"), "w") as f:
                                f.write("Bachelier Pricer " + "\n")
                                f.write(f"Generated: {datetime.now().isoformat()}\n")
                                f.write("-" * 40 + "\n")
                                for key, value in summary_results.items():
                                    f.write(f"{key}: {value}\n")

        logger.info(f"P&L from expired options of the strip calculated sucesfully!!")

        return df_out, summary_results

    def calculate_css_vol(self, df_temp, periods_in_year: int, dof: float = 1.0, monthly_resampling: bool=True):
        """Calculates annualized volatility for CSS directly.""" 
        if monthly_resampling:
            df_temp = df_temp.resample('ME').mean().diff().dropna()
            vol = df_temp.std(ddof=1)
            annualized_vol = vol * np.sqrt(periods_in_year)
        else:
            days_between = df_temp.index.to_series().diff().dt.days
            df_temp["CSS_diff_daily"] = df_temp["CSS"].diff() / days_between
            vol = df_temp["CSS_diff_daily"].std(ddof=dof)
            annualized_vol = vol*np.sqrt(periods_in_year) 
        
        if self.save_results:
            df_temp.to_excel(create_path(directory=self.output_dir, 
                                         filename="CSS for Volatility Calculation.xlsx"))

        return annualized_vol[0]

    def calculate_effective_hedging(self, vol_col: str,  df_temp: pd.DataFrame, resample: str = "ME"):
         """Calculates effective hedging profile, based on in the money european options."""
         df_temp["Hedging"] = df_temp["in_the_money"] * df_temp[vol_col]

         df_month = df_temp["Hedging"].resample(resample).sum()

         return df_month

    def orchestrate_calculation(
              self, df_future, df_past,df_cal, val_date, r, num_periods, monthly_resampling: bool=True,
              dt_hours: float = 1.0, volumes_col:str = "volume_mw", price_correction_switch: bool = True):
        """Orchestrates calcualtion."""
        _ = self.construct_outputs_dir()

        df_past[volumes_col] = self.mw_per_option_per_hour
        df_future[volumes_col] = self.mw_per_option_per_hour

        df_future["delivery_hour"] = df_future.index.copy()
        df_past["delivery_hour"] = df_past.index.copy()

        df_future["css_fwd"] = df_future["FWD"] - self.heat_rate * df_future["TTF MA"] - self.emission_factor * df_future["EUA Y0"]
        df_past["css_fwd"] = df_past["MCP"] - self.heat_rate * df_past["TTF MA"] - self.emission_factor * df_past["EUA Y0"]

        sigma = self.calculate_css_vol(df_temp=df_cal, periods_in_year=num_periods, monthly_resampling=monthly_resampling)
        df_future["sigma"] = sigma

        df_1, dict1 = self.price_strip_of_options(
             df=df_future, valuation_date=val_date, r=r, 
             volume_col=volumes_col, dt_hours=dt_hours, value_correction=price_correction_switch)

        df_2, dict2 = self.calculate_realized_pnl(df=df_past, dt_hours=dt_hours, volume_col=volumes_col)
        
        df_eff_hedging = self.calculate_effective_hedging(vol_col=volumes_col, df_temp=df_1)
        df_pst_hedging = self.calculate_effective_hedging(vol_col=volumes_col, df_temp=df_2)

        if self.save_results:
            df_past.to_excel(create_path(directory=self.output_dir, filename="Past Data.xlsx"))
            df_future.to_excel(create_path(directory=self.output_dir, filename="Future Data.xlsx"))
            df_eff_hedging.to_excel(create_path(directory=self.output_dir, filename="Effective Hedging.xlsx"))
            df_pst_hedging.to_excel(create_path(directory=self.output_dir, filename="Past Hedging.xlsx"))

        return df_1, dict1, df_2, dict2

    
if __name__ == "__main__":

    print("Under Construction!!")