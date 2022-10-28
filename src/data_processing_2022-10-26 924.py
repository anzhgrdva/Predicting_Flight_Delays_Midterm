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


# FLIGHTS: FEATURE SELECTION AND ENGINEERING
from datetime import timedelta
import pandas as pd
import numpy as np


    # CONVERT crs_dep_time TO DATETIME OBJECT
column = 'crs_dep_time'
flights = time_columns(flights,column,format='%H%M',dropna=True)


    # To flights table, add columns with the  features for date and forecasting date
flights = date_forecast_columns(flights,date_column='fl_date',format='%Y-%m-%d')

    # Calculate mean dep_delay and arr_delay for a given carrier 
groupby_columns_1 = ['mkt_carrier', 'origin_airport_id']
columns = ['dep_delay', 'arr_delay'] # columns on which to get aggregate data
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
groupby_tm1_year_week = groupby_columns_1 + ['year','fl_date_week_number'] # SH 2022-10-27 19:11 updated
# Mean delay for the same MONTH 1 year ago; 
groupby_tm1_year_month = groupby_columns_1 + ['month'] # SH 2022-10-27 19:11 updated

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



# PASSENGERS TABLE
# Load the CSV file with the variable name `passengers` for convenience.

    # Calculate average per departure
col_for_averaging = [
    'payload',
    'seats',
    'passengers',
    'freight',
    'mail',
]
    # Calculate average per departure
for column in col_for_averaging:
    passengers[str('mean_'+column+'_per_departure')] = passengers[column] / passengers['departures_performed']

    # calculate mean empty seats
passengers['mean_empty_seats_per_departure'] = passengers['mean_seats_per_departure'] - passengers['mean_passengers_per_departure']

    # Drop columns whose data were put into the new columns as 'mean_.*'
passengers.drop(columns=col_for_averaging, inplace=True)

    # Prepare for the merge!
left_join_columns = ['mkt_carrier','origin_airport_id', 'dest_airport_id', 
    'fl_date_t-1_year_year', 'month']
right_join_columns = ['carrier','origin_airport_id', 'dest_airport_id', 
    'year', 'month'] # SH 2022-10-27 20:42 changed 'unique_carrier' to 'carrier'

    # Join flights and passengers table
flights = flights.merge(
    passengers,how='left',left_on=left_join_columns, 
    right_on=right_join_columns)

    # check that the new columns have values and not all NaNs.
flights.values_sort('mean_seats_per_departure')

    # Drop duplicates after the merge
flights.drop_duplicates(subset=['mkt_unique_carrier','fl_date','mkt_carrier_fl_num'], inplace=True)

# # HANDLING MISSING VALUES
#     # FUEL TABLE
#     # Fill missing total_gallons values with sum of tdom_gallons and tint_gallons
# fuel.fillna(fuel['tdomt_gallons'] + fuel['tint_gallons'], inplace=True)
# print((fuel.filter(regex='gallons') == 0).sum(axis=0)) # Check
# fuel.head()
#     # Drop rows with total_gallons = 0 because the data is incorrect
# print('Number of rows: ',len(fuel))
# fuel_rows_to_drop = fuel['total_gallons'] == 0
# fuel.drop(fuel[fuel_rows_to_drop].index,inplace=True)
# print('Number of rows: ',len(fuel))

# FLIGHTS TABLE TRAINING DATA: Fill mean flight historical/forecasting delay data. 
dict = {
    'mean_dep_delay_carrier_origin_date_t-1_week': 'mean_dep_delay_carrier_origin_week', 
    'mean_arr_delay_carrier_origin_date_t-1_week': 'mean_arr_delay_carrier_origin_week',
    'mean_dep_delay_carrier_origin_date_t-1_week_week_number': 'mean_dep_delay_carrier_origin_week',
    'mean_arr_delay_carrier_origin_date_t-1_week_week_number': 'mean_arr_delay_carrier_origin_week',
    # 'mean_dep_delay_carrier_origin_datet-1_year_week': 'mean_dep_delay_carrier_origin_week', # DON'T USE THIS FOR NOW
    # 'mean_arr_delay_carrier_origin_datet-1_year_week': 'mean_arr_delay_carrier_origin_week', # DON'T USE THIS FOR NOW
    # 'mean_dep_delay_carrier_origin_date_t-1_year_month': 'mean_dep_delay_carrier_origin_month', # DON'T USE THIS FOR NOW
    # 'mean_arr_delay_carrier_origin_date_t-1_year_month': 'mean_arr_delay_carrier_origin_month' # DON'T USE THIS FOR NOW
}
fill_missing(flights,dict,fill_w_mean=False) # Call the function
explore(flights.filter(regex='mean')) # Recheck missing values

# REMOVE NULLS
    # REMOVE FEATURES WITH NULL VALUES ABOVE THRESHOLD
threshold = 0.5 
drop_features(flights,threshold=threshold,show_update=False)

    # Drop rows with any missing values

df = df_with_passangers.dropna(subset=['departures_performed', 'distance', 'unique_carrier', 'airline_id',
       'origin_airport_id', 'dest_airport_id', 'year', 'month',
       'mean_payload_per_departure', 'mean_seats_per_departure',
       'mean_passengers_per_departure', 'mean_freight_per_departure',
       'mean_mail_per_departure', 'mean_empty_seats_per_departure'])

# Define a function to fill missing values with the mean value in that column
def fill_with_mean(df,columns,agg='mean',inplace=True):
    """
    Get the average value in the column.

    Parmaters:
    - Data: `Dataframe groupby().apply()` argument.
    - Columns: Column names on which to perform calculations. Use a list for multiple.
    - agg (string, optional): Aggregate function to apply. Default is mean.
    """

    for column in columns:
        df.fillna(df.loc[:,column].agg(agg), inplace=True)

    return df

groupby_columns = ['mkt_carrier']
columns_to_fill = # columns to fill
flights = flights.groupby(groupby_columns,group_keys=False).apply( 
        lambda x: fill_with_mean(x,columns_to_fill))



# PCA
    # Data should be scaled before this step
    # Remove columns that cannot be used for prediction, as well as identification columns
    # Double check what is listed. 
columns_for_ID = ['op_carrier_fl_num', 'origin_airport_id', 'dest_airport_id',
       'month', 'fl_date_year',
       'fl_date_week_number', 
    'crs_dep_time', 'crs_arr_time', 
       'actual_elapsed_time', 
       'arr_delay', 'dep_delay',
       ]

    # Drop columns that shouldn't be included in PCA
ID_columns = flights[columns_for_ID].copy() # Save these columns for identification later
ID_columns.pd.to_csv('pca_ID_columns.csv')

flights.drop(labels=columns_for_ID, inplace=True, axis=1)

       # Run the PCA
pca = run_pca(flights, n_components=0.95, cluster_col=None)
# Save the pca dataframe. This dataframe is the data that will be entered into the model.


# Extract Jan and Dec 2019 data 
filter = (df_with_passangers['year'] == 2019) & ((df_with_passangers['month'] == 1) | (df_with_passangers['month'] == 12))
df_with_passangers[filter].to_csv('historical_features_for_test_data.csv')