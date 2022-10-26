import pandas as pd
 
import matplotlib.pyplot as plt
import seaborn as sns

def load_csv(filepath,filename,column1_as_index=False):
    """
    Load a csv file as a dataframe using specified file path copied from windows file explorer.
    Back slashes in file path will be converted to forward slashes.
    Arguments:
    - filepath (raw string): Use the format r'<path>'.
    - filename (string).
    - colum1_as_index (bool): If true, take the first column as the index. 
        Useful when importing CSV files from previously exported dataframes.

    Returns: dataframe object.

    Required import: pandas
    """

    filename = f'{filepath}/'.replace('\\','/')+filename
    df = pd.read_csv(filename)
    if column1_as_index==True:
        df.set_index(df.columns[0], inplace=True)
        df.index.name = None
    return df

def save_csv(df,filename,path=None,append_version=False):
    """
    Export dataframe to CSV.
    Parameters:
    - df: Dataframe variable name.
    - filename: Root of the filename.
    - filepath (raw string): Use the format r'<path>'. If None, file is saved in same director.
    - append_version (bool): If true, append date and time to end of filename.
    """

    from datetime import datetime
    
    if path:
        path = f'{path}/'.replace('\\','/')
    if append_version == True:
        filename+=datetime.now().strftime('%Y-%m-%d_%H%M')
    df.to_csv(path+filename+'.csv')
    print('File saved: ',path+filename)

# convert dates from string to datetime objects
def date_columns(df,date_column='fl_date',format='%Y-%m-%d'):
    """ 
    Take the dates in a dateframes to create new columns:
        _date_standard: Datetime data 
        _year
        _month
    Parmaters:
    - df: Dataframe.
    - date_column: Name of the column containing the date strings.
    - Format: Original date format in the dateframe. Default: '%d.%m.%Y'
    
    Make sure to do the following import: 
    from datetime import datetime
    """

    date_column=str(date_column)
    
    # df[str(date_column+'_year')] = pd.to_datetime(df[date_column],
    #     format='%d.%m.%Y')
    date = pd.to_datetime(df[date_column],
        format=format)
    # df.get(str(date_column+'_standard'),date)
    # df.get(str(date_column+'_year'),date.dt.year)
    # df.get(str(date_column+'_month'),date.dt.month)
    df[str(date_column+'_standard')] = date
    df[str(date_column+'_year')] = date.dt.year
    df[str(date_column+'_month')] = date.dt.month
    return df

def compare_id(df1, df1_column, df2, df2_column,print_common=False,print_difference=True):
    """
    Print the number of common values and unique values between two dataframe columns.
    
    """
    df1_values = df1[df1_column].unique()
    df2_values = df2[df2_column].unique()
    common_values = set(df1_values) & set(df2_values)
    if len(df1_values) > len(df2_values):
        different_values = set(df1_values) - set(df2_values)
        print(f'Proper subset = {set(df2_values) < set(df1_values)}')
    else:
        different_values = set(df2_values) - set(df1_values)
        print(f'Proper subset = {set(df1_values) < set(df2_values)}')
    print('Unique values in df1:',len(df1_values))
    print('Unique values in df2:',len(df2_values))
    print('Number of common values between df1 and df2:',len(common_values))
    print('Number of different values between df1 and df2:',len(different_values))
    if print_common == True:
        print('Values in common:',common_values)
    if print_difference == True:
        print('Different values:',different_values)
    
# function that prints null values

def explore(df,id=0,print_n_unique=False, printValues=False):
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

# Function to plot multiple histograms using Plotly. Show different colours based on classification.
def plot_int_hist(df, columns=None, color=None):
    """
    Use Plotly to plot multiple histograms using the specified columns of a dataframe.
    Arguments:
    - df: Dataframe.
    - columns (optional): Columns of dataframe on which to create the histogram. If blank, all numeric data will be plotted.
    - color (optional): Provide name of colum containing binary classification values 0 and 1. 
        Data points classified as 1 will be in red.
    
    Make sure to do the following imports:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    """
    # import plotly.express as px
    import plotly.graph_objects as go
    import plotly 
    from plotly.subplots import make_subplots

    if columns == None:
        columns = df.dtypes[df.dtypes != 'object'].index.tolist()
    fig = make_subplots(rows=round((len(columns)+.5)/2), cols=2,subplot_titles=columns)
    for i, feature in enumerate(columns):
        if color:
            bins = dict(
                start = min(df[feature]),
                end =  max(df[feature]),
                # size=
            )
            zero = df[df[color]==0]
            one = df[df[color] != 0]
            fig.add_trace(go.Histogram(x=zero[feature],
                marker_color='#330C73',
                opacity=0.5,
                xbins=bins), 
                row=i//2+1, col=i % 2 + 1
                )
            fig.add_trace(go.Histogram(x=one[feature],
                marker_color='red',
                opacity=0.5,
                xbins=bins),
                row=i//2+1, col=i % 2 + 1)
        else:
            fig.add_trace(go.Histogram(x=df[feature]), 
            row=i//2+1, col=i % 2 + 1)
    fig.update_layout(height=300*round((len(columns)+.5)/2), 
        showlegend=False,barmode='overlay')
    fig.show()
   
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

# Function to plot multiple histograms
def plot_hist(df, columns=None):
    """
    Plot multiple histograms using the specified columns of a dataframe.
    Arguments:
    df: Dataframe.
    columns (optional): Columns of dataframe on which to create the histogram. If blank, all numeric data will be plotted.
    
    Make sure to `import seaborn as sns`.
    """
    if columns == None:
        columns = df.dtypes[df.dtypes != 'object'].index.tolist()
    fig, ax = plt.subplots(nrows=round((len(columns)+.5)/2), ncols=2, figsize=(10,18))
    for i, feature in enumerate(columns):
        sns.histplot(data=df,x=feature,ax=ax[i//2, i % 2])
    plt.tight_layout()

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

# create function to convert dates from string to datetime objects
def date_forecast_columns(df,date_column='fl_date',format='%Y-%m-%d'):
    """ 
    Take the dates in a dateframes to create new columns:
        _date_standard: Datetime data 
        _year
        _month
        _1_week_ago
        _1_year_ago

    Parmaters:
    - df: Dataframe.
    - date_column: Name of the column containing the date strings.
    - Format: Original date format in the dateframe. Default: '%d.%m.%Y'
    
    Make sure to do the following import: 
    from datetime import datetime
    """
    import pandas as pd
    date_column=str(date_column)
    
    date = pd.to_datetime(df[date_column],
        format=format)
    df[str(date_column+'_dt')] = date # date
    df[str(date_column+'_year')] = date.dt.to_period('Y') # year
    df[str(date_column+'_year_month')] = date.dt.to_period('M') # month
    df[str(date_column+'_Monday_of_week')] = date.dt.to_period('W').dt.start_time # Monday of the week
    df[str(date_column+'_week_number')] = date.dt.isocalendar().week # week of the year


    df[str(date_column+'_t-1_week_week_number')] = (date - pd.Timedelta(days=7)).dt.isocalendar().week # previous week's week number of the year 
    df[str(date_column+'_t-1_week_date')] = date - pd.Timedelta(days=7) # 7 days before

    df[str(date_column+'_t-1_year_year')] = (date - pd.Timedelta(days=365)).dt.to_period('Y') # Previous year
    df[str(date_column+'_t-1_year_month')] = (date - pd.Timedelta(days=365)).dt.to_period('M') # Same month 1 year ago
    df[str(date_column+'_t-1_year_day')] = date - pd.Timedelta(days=365) # 365 days before
    
    return df

# function to convert time to datetime objects
def date_columns(df,time_column,format='%H%M'):
    """ 
    Take the time in a dateframes to create new columns:
        _sin
        _cos

    Parmaters:
    - df: Dataframe.
    - time_column (string or list of strings): Name of the column containing the time.
    - Format: Original time format in the dateframe. Default is in flight format: '%H%M'
    
    Make sure to do the following import: 
    from datetime import datetime
    """
    if type(time_column) == str:
        time_column = [time_column]
    for column in time_column:
        df[str(column+'_time')] = pd.to_datetime(df[column],format=format)
   
    return df

time_column = ['crs_dep_time','crs_arr_time']
date_columns(flights,time_column,format='%H%M').filter(regex='time')

def date_columns(df,column,format='%H%M',dropna=False, fillna=1):
    """ 
    Take the time in a dateframe to create new column with date time object.
    Any null values will be replaced with 1. 

    ** Note 2022-10-26 8:54**: For flights data, only works on 'crs_arr_time' column.
    Raises error for 'dep_time', 'crs_arr_time', 'arr_time' columns.

    Parmaters:
    - df: Dataframe.
    - column (string): Name of the column containing the time.
    - Format: Original time format in the dateframe. Default is in flight format: '%H%M'
    - dropna (bool): If true, drop rows where the column value is null.
    - fillna (int): If true, fill missing values with the given number.
    
    Make sure to do the following import: 
    from datetime import datetime
    """
    if dropna == True:
        df.dropna(subset=column, inplace=True)
    else:
        df[column].fillna(value=fill_na, inplace=True)
    df[column].fillna(value=1, inplace=True)
    df[str(column+'_dt')] = df[column].astype(int).astype(str).apply(lambda x: datetime.strptime(x.zfill(3), "%H%M"))

    return df

def aggregate(data,columns,groupby,agg='mean'):
    """
    Get the average value.

    Parmaters:
    - Data: `Dataframe groupby().apply()` argument.
    - Columns: Column names on which to perform calculations. Use a list for multiple.
    - groupby (string): String to append to the end of the new columns to indicate
        how data were grouped.
    - agg (string, optional): Aggregate function to apply. Default is mean.
    """

    for column in columns:
        data.loc[:,str('mean_'+column+'_'+groupby)] = data.loc[:,column].agg(agg)

    return data

# create function to convert dates from string to datetime objects
def date_forecast_columns(df,date_column='fl_date',format='%Y-%m-%d'):
    """ 
    Take the dates in a dateframes to create new columns:
        _date_standard: Datetime data 
        _year
        _month
        _1_week_ago
        _1_year_ago

    Parmaters:
    - df: Dataframe.
    - date_column: Name of the column containing the date strings.
    - Format: Original date format in the dateframe. Default: '%d.%m.%Y'
    
    Make sure to do the following import: 
    from datetime import datetime
    """
    import pandas as pd
    date_column=str(date_column)
    
    date = pd.to_datetime(df[date_column],
        format=format)
    df[str(date_column+'_dt')] = date # date
    df[str(date_column+'_year')] = date.dt.to_period('Y') # year
    df[str(date_column+'_year_month')] = date.dt.to_period('M') # month
    df[str(date_column+'_Monday_of_week')] = date.dt.to_period('W').dt.start_time # Monday of the week
    df[str(date_column+'_week_number')] = date.dt.isocalendar().week # week of the year


    df[str(date_column+'_t-1_week_week_number')] = (date - pd.Timedelta(days=7)).dt.isocalendar().week # previous week's week number of the year 
    df[str(date_column+'_t-1_week_date')] = date - pd.Timedelta(days=7) # 7 days before

    df[str(date_column+'_t-1_year_year')] = (date - pd.Timedelta(days=365)).dt.to_period('Y') # Previous year
    df[str(date_column+'_t-1_year_month')] = (date - pd.Timedelta(days=365)).dt.to_period('M') # Same month 1 year ago
    df[str(date_column+'_t-1_year_day')] = date - pd.Timedelta(days=365) # 365 days before
    
    return df