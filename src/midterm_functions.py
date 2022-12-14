import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime
from sklearn.preprocessing import StandardScaler
import psycopg2

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

def save_csv(df,filename,path=None,append_version=True):
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

def drop_features(df,threshold=.99):
    """
    Drop columns in a dataframe with null values above the specified threshold.
    Parameters:
    - df: Dataframe.
    - threshold (float): Float between 0 and 1. 
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
    return df

# function to convert time to datetime objects. 
# 2022-10-26 8:54: For flights data, only works on 'crs_arr_time' column.capitalize
    # Raises error for 'dep_time', 'crs_arr_time', 'arr_time' columns.

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

def time_columns(df,column,format='%H%M', dropna=False, fillna=1):
    """ 
    Take the time in a dateframe to create new columns with:
    - date time object
    - Hour in 24 hour format as an integer.
    
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
    time = df[column].astype(int).astype(str).apply(lambda x: datetime.strptime(x.zfill(3), "%H%M"))
    df[str(column+'_dt')] = time
    df[str(column+'_hour')] = time.dt.hour
    return df

# function to get aggregate data by group.
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

    # SH 2022-10-28 7:37 Added
    df['year'] = date.dt.year # SH 2022-10-28 7:37 Added
    df['month'] = date.dt.month
    # df['day'] = date.dt.day # not used
    # df[str(date_column+'_year')] = date.dt.to_period('Y') # year # SH 2022-10-28 7:37 remove
    
    # Convert datetime column/series to day of the week
    df['day_of_week'] = date.dt.weekday
    # Replace day numbers with day names
    df['day_of_week'].replace({ 
        0: 'Monday', 
        1: 'Tuesday', 
        2: 'Wednesday', 
        3: 'Thursday', 
        4: 'Friday', 
        5: 'Saturday', 
        6: 'Sunday'}, inplace=True)
    # SH 2022-10-28 7:37 end of add

    # Convert datetime column/series to year
    df['day_of_year'] = date.dt.day_of_year

    df[str(date_column+'_year_month')] = date.dt.to_period('M') # month
    df[str(date_column+'_Monday_of_week')] = date.dt.to_period('W').dt.start_time # Monday of the week
    df[str(date_column+'_week_number')] = date.dt.isocalendar().week # week of the year


    df[str(date_column+'_t-1_week_week_number')] = (date - pd.Timedelta(days=7)).dt.isocalendar().week # previous week's week number of the year 
    df[str(date_column+'_t-1_week_date')] = date - pd.Timedelta(days=7) # 7 days before

    df[str(date_column+'_t-1_year_year')] = (date - pd.Timedelta(days=365)).dt.year # Previous year
    df[str(date_column+'_t-1_year_month')] = (date - pd.Timedelta(days=365)).dt.month # Same month 1 year ago # updated 2022-10-27 11:51
    df[str(date_column+'_t-1_year_day')] = date - pd.Timedelta(days=365) # 365 days before
    
    return df

def fill_missing(df, dict, inplace=True, fill_w_mean=False):
    """
    Fill  missing values. First step is to fill with the data from the mapped column. 
    If fill_w_mean is True (default False), second step will fill with the remaining 
    missing values with mean for the entire column.

    Parameters:
    - df: dataframe to fill using `.fillna()` method.
    - dict: Dictionary with column name to fill as the key and name of column with the data to 
    fill missing values with.
    - Inplace: bool, default True.
        If True, fill in-place. Note: this will modify any other views on this object (e.g., a no-copy slice for a column in a DataFrame).
    - fill_w_mean (bool): Default is False. If true, second step will fill with the remaining missing values with mean for the entire column.
    
    """
    for column_to_fill, column_filler in dict.items():
        df[column_to_fill].fillna(df[column_filler], inplace=True)
        if fill_w_mean==True:
            df[column_to_fill].fillna(df[column].agg('mean'), inplace=True)
    return df

def scale_data(df, numeric_cols, cat_cols, scaler=None):
    """
    - Perform standardization (StandardScaler) on the numeric_cols of the dataframe. 
    - combines both numeric and categorical back to the entire feature dataframe.

    Params:
    - df: Dataframe object with both feature and target data. 
    - numeric_cols: Name of the numeric columns to be scaled.
    - cat_cols: Name of the categorical columns (dummy variables) NOT to be scaled.
    - scaler (optional): Provide fitted scaler option to fit data on. Default is None.

    Returns: 
    - Dataframe with numeric data scaled and categorical data as-is. Original index retained
    - Scaler for subsequent use
    """
    # Create the scaler based on the training dataset
    if scaler:
        scaler = scaler
        print('Supplied scaler applied.')
    else:
        scaler = StandardScaler()
        print('New scaler created.')
    scaler.fit(df.filter(numeric_cols))
    print('Scaler fit complete.')
    X_numeric_scaled = scaler.transform(df.filter(numeric_cols))
    print('Scaling complete.')
    X_categorical = df.filter(cat_cols).to_numpy()
    # X = pd.DataFrame(np.hstack((X_categorical, X_numeric_scaled)))
    X = pd.DataFrame(np.hstack((X_categorical, X_numeric_scaled)), 
    index=df.index,
    columns=cat_cols + numeric_cols)
    return X, scaler

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

# Perform PCA: SH 2022-10-27 
def run_pca(df, n_components=2, column_range=None, cluster_col=None,plot=False):
    """
    Run a PCA, then plot data along the first 2 PC dimensions and the projections.

    Parameters: 
    - col_range: Start and end index for column numbers to include in the model.
        Default is None to include all columns.
    - cluster_col (tuple): Column(s) with cluster ids.

    Return PCA result as a dataframe.

    """
    from sklearn.decomposition import PCA
    import seaborn as sns

    pca = PCA(n_components=n_components)
    if column_range == None:
        columns = df.columns
    else:
        columns = df.columns[column_range[0]:column_range[1]]
    data_scaled = df[columns]
    
    # Apply PCA
    pca.fit(data_scaled)
    data_pca = pca.transform(data_scaled)
    data_pca = pd.DataFrame(data_pca)

    # Get the projections ('loadings') of each dimension along each principal component:
    loadings = pd.DataFrame(pca.components_)

    # rename the columns from the PCA dataframe result
    loadings.columns = columns
    
    # plot PCA showing both KMeans clusters and AC clusters
    if plot:
        # To plot the raw data along with the loading plot, scale the raw data down:
        xscale = 1/(data_pca[0].max()-data_pca[0].min())
        yscale = 1/(data_pca[1].max()-data_pca[1].min())
        # Make the plots
        fig, ax = plt.subplots(ncols=2,nrows=1,figsize=(12,5))

        # Plot showing KMeans clusters
        if cluster_col:
            clustering_col1 = df.columns[cluster_col[0]]

            sns.scatterplot(
                x=data_pca[0]*xscale,y=data_pca[1]*yscale,
                hue=df[clustering_col1].values,
                ax=ax[0]
                )
            for feature, vector in loadings.items():
                # Plot each feature using the two principal components as axes
                ax[0].arrow(0,0,vector[0],vector[1]) 
                # Label each arrow at the tip of the line
                if (vector[0] > loadings.loc[0,:].mean()) | (vector[1] > loadings.loc[1,:].mean()):
                    ax[0].text(vector[0],vector[1],feature)
                    print('Feature vector component above average: ',feature)
            ax[0].set_xlabel('PC1')
            ax[0].set_ylabel('PC2')
            ax[0].set_title(clustering_col1)
            
            # Plot showing AC clusters
            if len(cluster_col) == 2:
                clustering_col2 = df.columns[cluster_col[-1]]
                sns.scatterplot(
                    x=data_pca[0]*xscale,y=data_pca[1]*yscale,
                    hue=df[clustering_col2].values,
                    ax=ax[1]
                    )
                for feature, vector in loadings.items():
                    # Plot each feature using the two principal components as axes
                    ax[1].arrow(0,0,vector[0],vector[1]) 
                    # Label each arrow at the tip of the line
                    if (vector[0] > loadings.loc[0,:].mean()) | (vector[1] > loadings.loc[1,:].mean()):
                        ax[1].text(vector[0],vector[1],feature)
                ax[1].set_xlabel('PC1')
                ax[1].set_ylabel('PC2')
                ax[1].set_title(clustering_col2)

    return data_pca