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
    import pandas as pd
    filename = f'{filepath}/'.replace('\\','/')+filename
    df = pd.read_csv(filename)
    if column1_as_index==True:
        df.set_index(df.columns[0], inplace=True)
        df.index.name = None
    return df

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