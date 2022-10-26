data_columns = [
    
    'mean_dep_delay_carrier_origin_month',
    'mean_arr_delay_carrier_origin_month',
    'mean_dep_delay_carrier_origin_week',
    'mean_arr_delay_carrier_origin_week',
    'mean_dep_delay_carrier_origin_date',
    'mean_arr_delay_carrier_origin_date',
    'mean_dep_delay_carrier_origin_date_t-1_week',
    'mean_arr_delay_carrier_origin_date_t-1_week',
    'mean_dep_delay_carrier_origin_date_t-1_week_week_number',
    'mean_arr_delay_carrier_origin_date_t-1_week_week_number',
    'mean_dep_delay_carrier_origin_datet-1_year_week',
    'mean_arr_delay_carrier_origin_datet-1_year_week',
    'mean_dep_delay_carrier_origin_date_t-1_year_month',
    'mean_arr_delay_carrier_origin_date_t-1_year_month'


]



def explore(df,id=0,print_n_unique=True, printValues=False):
    """
    Explore dataframe data and print missing values.
    Parameters:
    - df: Dataframe.
    - id: Column number or name with the primary IDs. Default is zero.
    - print_n_unique (bool): If the number of unique values in the first column doesn't match 
        the number of rows in the df, print the number of unique values in each column to see if 
        there's another column that might serve as a unique id.
    """
    if (id==False) & (id !=0):
        pass
    elif isinstance(id,int):
    # if type(id)==int:
        print(f'Unique IDs: {len(set(df.iloc[:,0]))}. # of rows: {df.shape[0]}. Match: {len(set(df.iloc[:,0]))==df.shape[0]}')
    else:
        print(f'Unique IDs: {len(set(df[id]))}. # of rows: {df.shape[0]}. Match: {len(set(df[id]))==df.shape[0]}')
    
    # if the number of unique values in the first column doesn't match the number of rows in the df,
    # print the number of unique values in each column to see if there's another column that migh
    # serve as a unique id.
    if (print_n_unique==True):
        if len(set(df.iloc[:,0])) !=df.shape[0]: 
            for column in df.columns:
                print(len(df[column].value_counts()),'\t', column)
    
    # count amount of missing values in each column
    total = df.isnull().sum().sort_values(ascending=False) 
    # % of rows with missing data from each column
    percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False) 

    # create a table that lists total and % of missing values starting with the highest
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent']) 

    if (printValues == True):
        # extract the names of columns with missing values
        cols_with_missing = missing_data[missing_data.Percent > 0].index.tolist()
        print(df.dtypes[cols_with_missing])

    print(f'')
    return missing_data

explore(flights_df,print_n_unique=False)



explore(flights_df,print_n_unique=False).head(35)







## Silvia 2022-10-25 20:48 Example of how to call the custom function for supervised learning
param_lr = {
    # 'penalty': ['l1','l2', 'elasticnet'],
    'C': C_list,
    'max_iter' : max_iter_list,
    'class_weight': [None, 'balanced']
}

lr = LogisticRegression(random_state=0)
lr_attributes = supervised(df, lr, param_lr, model_name='logistical regression')
best_lr = lr_attributes.get_best_model()

# Save the model
model = best_lr

filename = 'model_best_lr.sav'
pickle.dump(model, open(filename, 'wb'))







def correlation(df):
    """
    Plot the correlation matrix.
    Returns the dataframe with the correlation values.
    """

    # Create a mask to exclude the redundant cells that make up half of the graph.
    mask = np.triu(np.ones_like(df.corr(), dtype=bool))

    # Create the heatmap with the mask and with annotation
    sns.heatmap(data=df.corr(numeric_only=True),mask=mask,annot=True)
    return df.corr()



# Silvia @ 2022-10-25 18:48 Function to drop columns with missing values above a given threshold:
def drop_features(df,threshold=100, show_update=True):
    """
    Drop columns in a dataframe with null values above the specified threshold.
    Parameters:
    - df: Dataframe.
    - threshold (float): Float between 0 and 100. 
        Threshold of % null values over which columns will be dropped.
    - show_update: If true, show missing values for the updated dataframe
        (calls the custom function explore)
    """ 
    
    # count amount of missing values in each column
    total = df.isnull().sum().sort_values(ascending=False) 
    # % of rows with missing data from each column
    percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False) 

    # create a table that lists total and % of missing values starting with the highest
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent']) 

    filter = missing_data['Percent'] > threshold
    to_drop = missing_data[filter].index.tolist()
    df.drop(to_drop, axis=1, inplace=True)
    print(f'Threshold of percentage values for dropping columns: {threshold}')
    print(f'Columns dropped: {to_drop}')
    if show_update == True:
        return explore(df,id=0,print_n_unique=False, printValues=False)

# # Apply the function by uncommenting

# threshold = 100 # Set the threshold
# drop_features(flights_df,threshold=threshold)




# 2022-10-26 12:12: For filling in missing data in mean delay data

def fill_missing(df, dict, inplace=True):
    """
    Parameters:
    - df: dataframe to fill using `.fillna()` method.
    - dict: Dictionary with column name to fill as the key and name of column with the data to 
    fill missing values with.
    - Inplace: bool, default True.
        If True, fill in-place. Note: this will modify any other views on this object (e.g., a no-copy slice for a column in a DataFrame).
    
    """
    for column_to_fill, column_filler in dict.items():
        df[column_to_fill].fillna(df[column_filler], inplace=True)
    return df


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

# Call the function
fill_missing(flights_df,dict)
explore(flights_df) # Recheck missing values







# Save the file to CSV
from datetime import datetime
datetime_now = datetime.now().strftime('%Y-%m-%d_%H%M')
filename = f'flights_{datetime_now}.csv'
print(filename)














