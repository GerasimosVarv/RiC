import warnings
warnings.filterwarnings('ignore')

# Enable import of other modules
import sys
ROOT = r"C:\Users\g.varvounis\Documents\RiC"
if ROOT not in sys.path:
    sys.path.append(ROOT)
###############################
import os
from pathlib import Path
import logging
import datetime as dt
from tools.general_tools import create_path, create_output_dir
from typing import Optional
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
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

@dataclass
class FuturesProduct:
    level: str            # "Y", "Q", or "M"
    start: dt.datetime
    end: dt.datetime      # exclusive
    value: float          # EUR/MWh, flat (baseload) settlement price
    label: str = "" 

class CuevrBuilder():
    """Builds Hourly Forward Curve based on futures' teors and DAM history."""

    def __init__(self, curve: str="GR", area_hour_identifier: str = "Europe/Athens", dam_price_col: str = "value",
                 inputs_path: str = r"C:\\Users\\g.varvounis\\OneDrive - HELLENiQ ENERGY\\Risk Section Activity\\RiC Monitoring\\_data\\BBG\\builder",
                 save_results: bool=True, 
                 output_dir: str = r"C:\\Users\\g.varvounis\\Documents\\RiC\\_other_outputs\\_curve"):

        self.curve_alias = curve
        self.area_hour_identifier = area_hour_identifier
        self.dam_price_col = dam_price_col
        self.inputs_path = inputs_path
        self.save_results = save_results
        self.output_dir = output_dir

    def month_bounds(self, year: int, month: int) -> tuple[dt.datetime, dt.datetime]:
        """Returns [start, end) datetimes (hour resolution, end exclusive) for a month."""
        start = dt.datetime(year, month, 1)
        if month == 12:
            end = dt.datetime(year+1, 1, 1)
        else:
            end = dt.datetime(year, month+1, 1)

        return start, end

    def quarter_of_month(self, month: int) -> int:
        """Defines the quarter of a given month."""
        return (month - 1) // 3 + 1

    def quarter_bounds(self, year: int, quarter: int) -> tuple[dt.datetime, dt.datetime]:
        """Returns bounds of a given quarter."""
        first_month = (quarter - 1) * 3 + 1
        start, _ = self.month_bounds(year=year, month=first_month)
        _, end = self.month_bounds(year=year, month= first_month + 2)

        return start, end

    def year_bounds(self, year: int) -> tuple[dt.datetime, dt.datetime]:
        """Returns bounds of a given year."""
        return dt.datetime(year, 1, 1), dt.datetime(year + 1, 1, 1)

    def construct_outputs_dir(self):
        """Creates the final data output directory per run."""
        final_dir = create_output_dir(out_dir=self.output_dir)
        self.output_dir = final_dir
        
        return final_dir

    def hours_between(self, start: dt.datetime, end: dt.datetime, dst_count: bool = False) -> int:
        """
        Returns number of hourly slots between start (incl.) and end (excl.).

        Note: Naive cont assuming no DST-drive 23/ 25h days.
        """
        if dst_count:
            start = pd.Timestamp(start).tz_localize(self.area_hour_identifier)
            end = pd.Timestamp(end).tz_localize(self.area_hour_identifier)
            return len(pd.date_range(start, end, freq="h", inclusive="left"))
        else:
            return int((end - start).total_seconds() // 3600)

    def day_type(self, date: dt.date, holidays: list = []) -> str:
        """Classify a calendar day for shaping purposes."""
        holidays = holidays or list()

        if date in holidays:
            return "Sunday" # treat public holidays as sundays (common convension)
        week_day = date.weekday() # Monday=0,..., SUnday=6
        if week_day == 5:
            return "Saturday"
        if week_day == 6:
            return "Sunday"
        return "Weekday"

    def load_dam_prices(
            self, start: str = "01-01-2020", end: str = "26-09-2026", 
            api_start: str = "http://10.7.32.34/cdm/Quotations/GetRData/91535c0a-31eb-4906-b895-2cd0fdcf36eb?startDate=",
            api_mid: str = "&endDate=", 
            api_end: str ="&dates=true&simulation=0") -> pd.DataFrame:
        """Returns historical DAM price time-series (hourly)."""
        logger.info(f"Importing DAM prices for {self.curve_alias}...")

        api = api_start + start + api_mid + end + api_end

        df_dam = pd.read_csv(api)
        
        df_dam["date"] = pd.to_datetime(df_dam["date"])
        df_dam = df_dam.set_index("date")
        df_dam.index.name=None
        df_dam.index = pd.to_datetime(df_dam.index)
        
        logger.info(f"DAM prices for {self.curve_alias} imported sucesfully!!")

        return df_dam

    def build_shape_profile(
            self, df_dam: pd.DataFrame, holidays: list = [],
            trim_quantiles: tuple[float, float] = (0.01, 0.99)) -> pd.DataFrame:
        """
        Builds normalized shape factors bucketed by month, day-type, hour.
        
        Steps:
            1. Optionally trim extreme spikes. Then calculate month, day-type, hour bucket averages.
            2. Average historical price within each bucket (month, day-type, hour).
            3. Normalize so that, for each *month*, the hour-weighted average of the shape factors
            across all day-type, hour combinations occuring in that calendar month equals 1.0.

            The last step guarantees that price_h = shape_h * FLAT FUTURE MONTHLY PRICE
            reconciles exactly back to the actual FLAT FUTURE MONTHLY PRICE when hour-weighted averaged over month.
        """
        df = df_dam.copy()

        # Optional: trim in spikes of historical DAM prices
        if trim_quantiles:
            low, high = df[self.dam_price_col].quantile(trim_quantiles[0]), df[self.dam_price_col].quantile(trim_quantiles[1])
            df[self.dam_price_col] = df[self.dam_price_col].clip(low, high)

        # Step 1: Raw average price per bucket (month, day-type, hour)
        df["month"] = df.index.month
        df["hour"] = df.index.hour
        df["day_type"] = [self.day_type(d.date(), holidays=holidays) for d in df.index]

        bucket_avg = (
            df.groupby(
                ["month", "day_type", "hour"])[self.dam_price_col]
                .mean().reset_index()
                .rename(columns={self.dam_price_col: "raw_avg"}))

        # Step 2: Normalization
        # To normalize per month we need, for each historical month, how many actual hours
        # of each (day-type, hour) occured -> use historical hour counts as weights
        # the above captures e.g., "this month had 4 Sundays that one 5..."
        counts = (
            df.groupby(
                ["month", "day_type", "hour"])[self.dam_price_col]
                .count().reset_index()
                .rename(columns={self.dam_price_col: "n_obs"}))
        bucket_avg = bucket_avg.merge(counts, on=["month", "day_type", "hour"])

        # Calculate wighted mean per month of the raw average (weight->n_obs)
        bucket_avg["_" + self.dam_price_col] = bucket_avg["raw_avg"] * bucket_avg["n_obs"]

        month_weighted_mean = (
            bucket_avg.groupby("month")
            .apply(lambda g: g["_" + self.dam_price_col].sum() / g["n_obs"].sum(), include_groups=False))

        bucket_avg["_month_mean"] = bucket_avg["month"].map(month_weighted_mean)
        bucket_avg["shape"] = bucket_avg["raw_avg"] / bucket_avg["_month_mean"]

        return bucket_avg[["month", "day_type", "hour", "shape"]].reset_index(drop=True)

    def shape_lookup_table(self, df_shape: pd.DataFrame) -> dict:
        """
        Converts DataFrame with shape to a fast dictionary lookup:
            (month, day-type, hour) bucket -> shape coefficient
        """
        return {
            (int(r.month), r.day_type, int(r.hour)): float(r.shape)
            for r in df_shape.itertuples()}

    def load_future_prices(self, products_sheet_name: str = "curve_builder"):
        """Load the prices of futures for the desired curve."""
        products = []
        path = os.path.join(self.inputs_path, self.curve_alias + "_CURVE_builder.xlsx")

        df = pd.read_excel(path, sheet_name=products_sheet_name)
        
        for _, row in df.iterrows():
            ptype = row["product_type"].strip().lower()
            period = str(row["period"]).strip()
            price = float(row[self.dam_price_col])

            if ptype == "year":
                y = int(period)
                start, end = self.year_bounds(y)
                products.append(FuturesProduct("Y", start, end, price, f"Cal-{y % 100:02d}"))
            elif ptype == "quarter":
                y_str, q_str = period.split("-Q")
                y, q = int(y_str), int(q_str)
                start, end = self.quarter_bounds(y, q)
                products.append(FuturesProduct("Q", start, end, price, f"Q{q}-{y % 100:02d}"))
            elif ptype == "month":
                y, m = period.split("-")
                y, m = int(y), int(m)
                start, end = self.month_bounds(y, m)
                products.append(FuturesProduct("M", start, end, price, f"{start:%b}-{y % 100:02d}"))
            else:
                raise ValueError(f"Unknown product_type: {row["product_type"]}")
                                
        return products

    def strip_to_monthly(
            self, products: list, 
            horizon_start: dt.datetime, horizon_end: dt.datetime) -> pd.DataFrame:
        """
        Bootstrap of overlapping Year/Quarter/Month products into a single 
        flat price per calendar month covering [horizon_start, horizon_end).

        Method for standard EEX-style products of forward curve:
            - For each Year product: value_year = price_year * hour_year
            Allocate value_year down to its 4 quarters, weighted by hours as a 
            provisional quarter price -- unless a quarter is directly quoted, in 
            which case that quarter keeps its quoted price and the *residual* value
            (value_year - sum of quoted quarter values) is spread across the remaining
            unqoted quarters by hour.
            Then, within each quorter (quoted or derived), repeat the same residual logic
            to split into months, using directly quoted Month products where available
            - Months with neither a Month, Quarter or Year product covering them are left 
            as NaN.
        Returns a DataFrame indexed by month start date with column "value".

        """
        calendar_start = horizon_start.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        months = pd.date_range(calendar_start, horizon_end, freq="MS", inclusive="left")
        month_hours = {m: self.hours_between(*self.month_bounds(m.year, m.month)) for m in months}

        month_price: dict[pd.Timestamp, float] = {m: np.nan for m in months}

        years_present = sorted(
            {p.start.year for p in products if p.level == "Y"} | 
            {p.start.year for p in products if p.level in ("Q", "M")})

        quoted_by_level = {"Y": {}, "Q": {}, "M": {}}

        for p in products:
            if p.level == "Y":
                quoted_by_level["Y"][p.start.year] = p.value
            elif p.level == "Q":
                quoted_by_level["Q"][(p.start.year, self.quarter_of_month(p.start.month))] = p.value
            elif p.level == "M":
                quoted_by_level["M"][(p.start.year, p.start.month)] = p.value

        for y in years_present:
            quarter_price: dict[int, Optional[float]] = {
                q: quoted_by_level["Q"].get((y, q)) for q in (1,2,3,4)}
            quarter_hours = {
                q: self.hours_between(*self.quarter_bounds(y, q)) 
                for q in (1, 2, 3, 4)}

            if y in quoted_by_level["Y"]:
                year_price = quoted_by_level["Y"][y]
                year_value = year_price * sum(quarter_hours.values())
                quoted_value = sum(
                    quarter_price[q] * quarter_hours[q] 
                    for q in (1, 2, 3, 4) if quarter_price[q] is not None)
                unquoted_qs = [q for q in (1, 2, 3, 4) if quarter_price[q] is None]

                if unquoted_qs:
                    residual_value = year_value - quoted_value
                    residual_hours = sum(quarter_hours[q] for q in unquoted_qs)
                    residual_price = residual_value / residual_hours if residual_hours else 0.0
                    for q in unquoted_qs:
                        quarter_price[q] = residual_price

            for q in (1, 2, 3, 4):
                q_start, q_end = self.quarter_bounds(y, q)
                q_months = [m for m in months if q_start <=m < q_end]
                if not q_months:
                    continue # this quarter outside requested horizon

                m_price = {m.month: quoted_by_level["M"].get((y, m.month)) for m in q_months}
                m_hours = {m.month: month_hours[m] for m in q_months}

                if quarter_price[q] is not None:
                    q_value = quarter_price[q] * sum(m_hours.values())
                    quoted_m_value = sum(m_price[mm] * m_hours[mm] 
                                         for mm in m_price 
                                         if m_price[mm] is not None)
                    unquoted_ms = [mm for mm in m_price if m_price[mm] is None]

                    if unquoted_ms:
                        residual_value = q_value - quoted_m_value
                        residual_hours = sum(m_hours[mm] for mm in unquoted_ms)
                        residual_price = residual_value / residual_hours if residual_hours else 0.0

                        for mm in unquoted_ms:
                            m_price[mm] = residual_price

                for m in q_months:
                    if m_price[m.month] is not None:
                        month_price[m] = m_price[m.month]

        df_out = pd.DataFrame({"month_start": list(month_price.keys()),
                            "value": list(month_price.values())})
        df_out = df_out.set_index("month_start").sort_index()
        n_missing = df_out["value"].isna().sum()

        if n_missing:
            print(f"[warning] {n_missing} month(s) in horizon have no covering futures product "
              f"(Year/Quarter/Month) and are NaN. Extend the futures data or apply a fallback.")

        return df_out   

    def build_hourly_fw_curve(
            self, df_monthly_prices: pd.DataFrame, shape_lookup: dict,
            horizon_start: dt.datetime, horizon_end: dt.datetime,
            holidays: list) -> pd.DataFrame:
        """
        Produces the final hourly time-series: price_h = shape_h * FlatMonthlyPrice(month of h).

        Returns a DataFrame indexed by hourly datetime with columns:
            - value
            - month
            - quarter
            - year
        """
        idx = pd.date_range(horizon_start, horizon_end, freq="h", inclusive="left")
        df = pd.DataFrame(index=idx)
        df["month_start"] = df.index.to_period("M").to_timestamp()
        df["month"] = df.index.month
        df["quarter"] = df.index.month.map(self.quarter_of_month)
        df["year"] = df.index.year
        df["day_type"] = [self.day_type(d.date(), holidays) for d in df.index]
        df["hour"] = df.index.hour

        df = df.merge(df_monthly_prices, left_on="month_start", right_index=True, how="left")
        df = df.rename(columns={self.dam_price_col: "flat_month_price"})

        df["raw_shape"] = [
            shape_lookup.get((int(r.month), r.day_type, int(r.hour)), 1.0)
            for r in df.itertuples()]

        month_mean_shape = df.groupby("month_start")["raw_shape"].transform("mean")
        df["shape"] = df["raw_shape"] / month_mean_shape
        df[self.dam_price_col] = df["shape"] * df["flat_month_price"]

        return df[[self.dam_price_col, "flat_month_price", "shape", "month", "quarter", "year"]]


    def orchestrate_process(
            self, api_start: str, api_mid: str, api_end: str,
            valuation_date: dt.date,
            dam_history_start: dt.date, dam__history_end: dt.date,
            horizon_years: int, holidays: list) -> pd.DataFrame:
        """Orchestrates the whole process."""
        logger.info(f"Creating Output Dir...")
        _ = self.construct_outputs_dir()
        logger.info(f"Output Dir created sucesfully!")

        dam_df = self.load_dam_prices(
            start=dam_history_start, end=dam__history_end,
            api_start=api_start, api_mid=api_mid, api_end=api_end)

        logger.info(f"Defining shape characteristics...")
        shape_df = self.build_shape_profile(df_dam=dam_df, holidays=holidays)
        shape_lu = self.shape_lookup_table(df_shape=shape_df)
        logger.info(f"Shape characteristics defined sucessfully!!")

        logger.info(f"Loading Forward Curve...")
        futures = self.load_future_prices()
        logger.info(f"Forward Curve loaded sucesfully!!")

        horizon_start = dt.datetime.combine(valuation_date + dt.timedelta(days=1), dt.time.min)
        horizon_end = dt.datetime(valuation_date.year + horizon_years, 1, 1)

        monthly =self.strip_to_monthly(
            products=futures, horizon_start=horizon_start, 
            horizon_end=horizon_end)

        logger.info(f"Determining Forward Curve...")
        hourly = self.build_hourly_fw_curve(
            df_monthly_prices=monthly, shape_lookup=shape_lu,
            horizon_start=horizon_start, horizon_end=horizon_end,
            holidays=holidays)
        logger.info(f"Forward Curve determined sucesfully!!")

        if self.save_results:
            dam_df.to_excel(create_path(directory=self.output_dir, filename="DAM History.xlsx"))
            shape_df.to_excel(create_path(directory=self.output_dir, filename="Shape.xlsx"))
            hourly.to_excel(create_path(directory=self.output_dir, filename=self.curve_alias + " Hourly FWD Curve.xlsx"))

        return hourly
     

if __name__ == "__main__":

    print("Pending Holiday update!!")

    # cl1 = CuevrBuilder()

    # df_result = cl1.orchestrate_process(
    #     api_start="http://10.7.32.34/cdm/Quotations/GetRData/91535c0a-31eb-4906-b895-2cd0fdcf36eb?startDate=",
    #     api_mid="&endDate=", api_end="&dates=true&simulation=0",
    #     valuation_date=dt.datetime(2026,9,25),
    #     dam_history_start="01-01-2023", dam__history_end="25-09-2026",
    #     horizon_years=2, holidays=[]
    # )
    # print(df_result)
    # df_result.to_excel("test1.xlsx")

    # s1, e1 = cl1.month_bounds(year=2026, month=1)
    # print(s1, e1)
    # for k in range(1, 13):
    #     q1 = cl1.quarter_of_month(month=k)
    #     print(q1, k)

    # for k in range(1, 5):
    #     print(cl1.quarter_bounds(year=2026, quarter=k))

    # print(cl1.year_bounds(year=2026))
    # print(cl1.year_bounds(year=2027))

    # print(cl1.hours_between(start=dt.datetime(2026, 1, 1), end=dt.datetime(2026,7,1), dst_count=False))
    # print(cl1.hours_between(start=dt.datetime(2026, 1, 1), end=dt.datetime(2026,7,1), dst_count=True))

    # print(cl1.hours_between(start=dt.datetime(2020, 1, 1), end=dt.datetime(2020,7,1), dst_count=False))
    # print(cl1.hours_between(start=dt.datetime(2020, 1, 1), end=dt.datetime(2020,7,1), dst_count=True))

    # print(cl1.day_type(date=dt.datetime(2026, 9, 26))) # #TODo: creaet a lsit of dt.datetime with all public hos

    # df1 = cl1.load_dam_prices()
    # print(df1.info())
    # print(df1[df1.index.hour == 23])

    # sh_df = cl1.build_shape_profile(df_dam=df1.dropna()) # #TODo: startegy with NaN in hourly DAM prices time-series (now=drop)

    # print(cl1.shape_lookup_table(df_shape=sh_df))

    # prods = cl1.load_future_prices()

    # out_df = cl1.strip_to_monthly(
    #     products=prods, horizon_start=dt.datetime(2026, 9, 26), 
    #     horizon_end=dt.datetime(2028, 1, 1)) # #TODo: study this



