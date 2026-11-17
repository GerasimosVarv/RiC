import pandas as pd
import numpy as np

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

def fetch_api_data(api: str, risk_factor: str, mode: str = 'DA', date_format: str = '%Y-%m-%d %H:%M:%S', 
                   resample_frequency: str = 'D', peak_hour_1: str = "08:00", peak_hour_2: str = "19:00", output_dir: str = ''):
        """Fetches data from Perseo."""
        # Fetch Data from API
        df = pd.read_csv(api)
        df_hour = None
        # Convert Date to Datetime data type & set as index of DataFrame
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', drop=True, inplace=True)

        # Keep aside values for the PK MtD / YtD Average Calculation
        df_pk = df.dropna().copy()

            # Keep hours of inerest
        df_pk = df_pk.between_time(peak_hour_1, peak_hour_2)
            # keep weekdays only
        df_pk = df_pk[df_pk.index.dayofweek < 5]
        df_pk.index = pd.to_datetime(df_pk.index)
        
        df_pk_ytd = df_pk.resample("YE").mean()
        df_pk_mtd = df_pk.resample("ME").mean()
        #############################################################

        # Rename index
        df.index.name = None
        df.index = pd.to_datetime(df.index)
        df_hour=df.copy()
        if 'DA' in mode:
            df_pk_ytd.to_excel(create_path(output_dir, risk_factor + " - Peak YtD.xlsx"))
            df_pk_mtd.to_excel(create_path(output_dir, risk_factor + " - Peak MtD.xlsx"))
            
            df.to_excel(create_path(output_dir, risk_factor + " - Initial Frequency.xlsx"))
            df = df.resample(resample_frequency).mean()

        # Rename column of DataFrame
        df.columns = [risk_factor]
        df_mtd = df.resample("ME").mean()
        df_ytd = df.resample("YE").mean()

        return df, df_mtd, df_ytd, df_hour


if __name__=="__main__":

    import pyodbc
    import pandas as pd
    import sqlite3
    from sqlalchemy import create_engine
    import urllib

    # conn = pyodbc.connect(
    #     "DRIVER={ODBC Driver 17 for SQL Server};"
    #     "SERVER=ELED-BI-DWH\ENT2019;"
    #     "DATABASE=DWH;"
    #     "Trusted_Connection=yes;"       
    # )

    # query = """
    # SELECT TOP 10 *
    # FROM dw.ENTSOE_DayAheadPrices
    # """

    # df = pd.read_sql(query, conn)

    # print(df.head())

    import pandas as pd
    from sqlalchemy import create_engine
    import urllib

    
    params = urllib.parse.quote_plus(
        f"DRIVER={{ODBC Driver 17 for SQL Server}};"
        f"SERVER=ELED-BI-DWH\ENT2019;"
        f"DATABASE=DWH;"
        "Trusted_Connection=yes;"
    )

    engine = create_engine(f"mssql+pyodbc:///?odbc_connect={params}")
    query = """
    SELECT TOP 10 *
    FROM [DWH].[dw].[ENTSOE_DayAheadPrices]
    """
    df = pd.read_sql(
    query,
    engine
    )

    print(df.head())
    



    


     