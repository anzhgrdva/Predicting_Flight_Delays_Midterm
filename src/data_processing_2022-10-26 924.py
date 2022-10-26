"""
For ease, name the flights dataframe as 'flights'.
"""


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
from midterm_functions import *

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

# EXPORT CSV
flights.to_csv('../data/raw/flights_train.csv')



# EDA
    # TEST NORMALITY
print(st.shapiro(flights['arr_delay']))

    # SEE PERCENTAGE OF VALUES THAT ARE NULL
explore(flights,printValues=False,id=0)

    # Look at distribution of time data
plot_int_hist(flights.filter(regex=r'time|cancelled'),color='cancelled')

    # plotting a correlation matrix and getting correlation values
correlation(fuel.filter(regex='gallon')) # fuel table



# HANDLING MISSING VALUES
    # FUEL TABLE
    # Fill missing total_gallons values with sum of tdom_gallons and tint_gallons
fuel.fillna(fuel['tdomt_gallons'] + fuel['tint_gallons'], inplace=True)
print((fuel.filter(regex='gallons') == 0).sum(axis=0)) # Check
fuel.head()
    # Drop rows with total_gallons = 0 because the data is incorrect
print('Number of rows: ',len(fuel))
fuel_rows_to_drop = fuel['total_gallons'] == 0
fuel.drop(fuel[fuel_rows_to_drop].index,inplace=True)
print('Number of rows: ',len(fuel))


# FLIGHTS: FEATURE SELECTION AND ENGINEERING
from datetime import timedelta
import pandas as pd
import numpy as np

    # To flights table, add columns with the  features for date and forecasting date
date_forecast_columns(flights,date_column='fl_date',format='%Y-%m-%d')

    # Calculate mean dep_delay and arr_delay for a given carrier 
groupby_columns_1 = ['mkt_carrier', 'origin_airport_id']
columns = ['dep_delay', 'arr_delay']
flights = flights.groupby(
    groupby_columns_1,group_keys=False).apply( # for a given month
        lambda x: aggregate(x,columns,'carrier_origin_month').groupby(
    groupby_columns_1,group_keys=False).apply( # for a given WEEK OF THE YEAR
        lambda x: aggregate(x,columns,'carrier_origin_week')).groupby(
    groupby_columns_1,group_keys=False).apply( # for a given DAY OF YEAR
        lambda x: aggregate(x,columns,'carrier_origin_date')))

    # FORECASTING DATA COLUMNS: ['dep_delay', 'arr_delay']
# columns for self-join using forecasting columns
columns_list = [
    'mean_dep_delay_carrier_origin_date',
    'mean_arr_delay_carrier_origin_date'
]
# Mean delay for the 7 days ago
groupby_tm1_week_date = groupby_columns_1 + ['fl_date_t-1_week_date'] 
# Mean delay for the 1 week before
groupby_tm1_week_week_number = groupby_columns_1 + ['fl_date_t-1_week_week_number']
# Mean delay for the same week 1 year ago
groupby_tm1_year_week = groupby_columns_1 + ['fl_date_year','fl_date_week_number']
# Mean delay for the same MONTH 1 year ago; 
groupby_tm1_year_month = groupby_columns_1 + ['fl_date_year_month']

        # ADD FORECASTING COLUMNS WITH SELF-JOIN
flights = flights.merge( 
                flights.filter(items=columns_list + groupby_tm1_week_date).groupby(groupby_tm1_week_date).mean(), # Mean delay for the 7 days ago
                how='left',
                # indicator='merge_tm1_week_date',
                left_on=groupby_columns_1+['fl_date_dt'],
                right_index=True,
                suffixes=[None,'_t-1_week'],
                copy=False
        ).merge( 
                flights.filter(items=columns_list + groupby_tm1_week_week_number).groupby(groupby_tm1_week_week_number).mean(), # Mean delay for the 1 week before # YES
                how='left',
                # indicator='merge_tm1_week_week_number',
                left_on=groupby_columns_1+['fl_date_week_number'],
                right_index=True,
                suffixes=[None,'_t-1_week_week_number'],
                copy=False
        ).merge(
                flights.filter(items=columns_list+ groupby_tm1_year_week).groupby(groupby_tm1_year_week).mean(), # Mean delay for the same week 1 year ago
                how='left',
                # indicator='merge_tm1_year_week',
                right_index=True, 
                left_on=groupby_columns_1+['fl_date_t-1_year_year','fl_date_week_number'],
                suffixes=[None,'t-1_year_week'],
                copy=False
        ).merge( 
                flights.filter(items=columns_list+groupby_tm1_year_month).groupby(groupby_tm1_year_month).mean(),# Mean delay for the same MONTH 1 year ago; 
                how='left',
                # indicator='merge_tm1_year_month',
                left_on=groupby_columns_1+['fl_date_t-1_year_month'],
                right_index=True,
                suffixes=[None,'_t-1_year_month'],
                copy=False

)
flights.columns

    # CONVERT crs_dep_time TO DATETIME OBJECT
column = 'crs_dep_time'
flights = date_columns(flights,column,format='%H%M',dropna=True)

# REMOVE FEATURES WITH NULL VALUES ABOVE THRESHOLD
threshold = 100
drop_features(flights,threshold=threshold,show_update=False)

# REMOVE NULLS