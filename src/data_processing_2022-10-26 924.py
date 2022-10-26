
# CLICK TO EXPAND SECTIONS
# DATA RETRIEVAL

import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

## SQL
def execute_query(query_string, return_pandas=True, limit=50):
    """
    Create a function to execute queries.
    limit  (int): Maximum number of rows to return. Default is 50.
    """
    con = psycopg2.connect(database='mid_term_project', user='lhl_student', password='lhl_student',
        host='lhl-data-bootcamp.crzjul5qln0e.ca-central-1.rds.amazonaws.com', port='5432')
    cur = con.cursor()

    if limit:
        query_string+=' LIMIT '+str(limit)
        print(query_string)
    if return_pandas:
        response = pd.read_sql_query(query_string, con)
    else:
        cur.execute(query_string)
        response = cur.fetchall()
    con.close()
    return response

# Retrieve select columns from flights table
query = """
    SELECT mkt_carrier,
    mkt_carrier_fl_num,
    fl_date,
    branded_code_share,
    op_unique_carrier,
    op_carrier_fl_num,
    origin_airport_id,
    origin_city_name,
    dest_airport_id,
    dest_city_name,
    crs_dep_time,
    dep_time,
    dep_delay,
    taxi_out,
    taxi_in,
    crs_arr_time,
    arr_time,
    arr_delay,
    cancelled,
    cancellation_code,
    diverted,
    crs_elapsed_time,
    actual_elapsed_time,
    air_time,
    distance,
    carrier_delay,
    weather_delay,
    nas_delay,
    security_delay,
    late_aircraft_delay,
    first_dep_time,
    total_add_gtime,
    longest_add_gtime
        FROM flights
"""
flights = execute_query(query,limit=600000)

# Retrieve select columns from passengers table
query = """
SELECT airline_id,
unique_carrier,
departures_performed,
 payload,
 seats,
 passengers,
 freight,
 mail,
 distance,
 air_time,
region,
carrier_group_new,
 origin_airport_id,
 origin_city_market_id,
 origin_city_name,
 origin_country_name,
 dest_airport_id,
 dest_city_market_id,
 dest_city_name,
 dest_country_name,
 aircraft_group,
 aircraft_type,
 aircraft_config,
 year,
 month,
class 
 FROM passengers
"""
passengers = execute_query(query,limit=None)

# EXPORT DATAFRAME TO CSV
flights.to_csv('../data/raw/flights_train.csv')

# EDA
    # TEST NORMALITY
print(st.shapiro(flights['arr_delay']))


# REMOVE NULLS
    # SEE PERCENTAGE OF VALUES THAT ARE NULL
explore(flights,printValues=False,id=0)

