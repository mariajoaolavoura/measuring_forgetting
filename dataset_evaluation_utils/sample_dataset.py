from datetime import datetime, timezone

import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('whitegrid')
# import plotly.offline as py
# pd.options.plotting.backend = "plotly"
# py.init_notebook_mode() # graphs charts inline (IPython).

a4_dims = (11.7, 8.27)


def filter_out_users_with_less_than_k_rates(df, user_col='user_id', k=5):
    '''
        df: pandas DataFrame,
        user_col: str = 'user_id',
        k: int = 5

        returns a new dataframe
    '''
    data = df.copy()

    k_core_users = data[user_col].value_counts()[ data[user_col].value_counts() >= k ].index
    return data.set_index(user_col).loc[k_core_users].reset_index()


def filter_out_users_with_less_than_k_rates_per_period(df, user_col='user_id', k=5, period='month'):
    '''
        df: pandas DataFrame,
        user_col: str = 'user_id',
        k: int = 5

        returns a new dataframe
    '''

    user_inter_col = 'user_interactions_per_'+period
    user_interactions_per_period = df[[user_col, period]].value_counts()\
                                        .to_frame(name = user_inter_col)
    user_interactions_per_period.reset_index(inplace = True)

    k_core_users = user_interactions_per_period[ user_interactions_per_period[user_inter_col] >= k ].index
    
    return user_interactions_per_period.set_index(user_col).loc[k_core_users].reset_index()



def split_timestamp(df, time_col='timestamp'):
    '''
        df: pandas DataFrame, requires a timestamp (dtype:int) column

        returns a new dataframe with year-month, year, month and day columns
    '''
    data = df.copy()

    # data['date'] = data[time_col].apply(lambda x: datetime.utcfromtimestamp(x)) # convert unix timestamp to date
    data['date'] = data[time_col].apply(lambda x: datetime.fromtimestamp(x, tz=timezone.utc))
    data = data.sort_values(by='date') # sort by date
    data['year-month'] = data['date'].apply(lambda x: datetime.strptime( str(x.year)+'-'+str(x.month), '%Y-%m' ))
    data['year'] = data['date'].dt.year
    data['month'] = data['date'].dt.month
    data['day'] = data['date'].dt.day

    return data


def sample_time_period(time_period, df, col='user_id', time_col='year-month', period='month'):
    '''
        time_period: list of tupples, start and end times with the respective time format compatible with the time_col refered
                     for example [('2013-01', '%Y-%m'), ('2017-01', '%Y-%m')]
                     or ['2013-01', '2017-01']

        df: pandas DataFrame, requires a time column in the '%Y-%m' format      

        col: str, 'user_id' or 'item_id'
        
        plots the Count of users per period

        returns the dataframe filtered by the given time period, 
                the str of start period,
                the str of end period
    '''


    if len(time_period)!=2:
        print('Wrong time_period shape!')
        return None

    # time_period shape of ['2013-01', '2017-01']
    if len(time_period[0]) != 2:
        
        time_period_start = time_period[0]
        time_period_end = time_period[1]

        time_period, dt_start, dt_end = get_time_period_and_datetime(time_period_start,
                                                                     time_period_end,
                                                                     period)
        y_filter = (df[time_col] >= dt_start) & \
                    (df[time_col] < dt_end)
    
    # time_period shape of [('2013-01', '%Y-%m'), ('2017-01', '%Y-%m')]
    else:
        time_period_start = time_period[0][0]
        time_period_end = time_period[1][0]
        
        y_filter = (df[time_col] >= datetime.strptime(*time_period[0])) & \
                (df[time_col] < datetime.strptime(*time_period[1]))
        
    
    if col == 'user_id':
        df[y_filter][[col, time_col]]\
            .drop_duplicates()\
            .groupby([time_col]).count()\
                .plot(title='Count of different users per '+period+' '+time_period_start+' to '+time_period_end);
    else:
        df[y_filter][[col, time_col]]\
            .groupby([time_col]).count()\
                .plot(title='Count of interactions per '+period+' '+time_period_start+' to '+time_period_end);

    return df[y_filter], time_period, time_period_start, time_period_end



def get_time_period_and_datetime(start, end, period='month'):
    '''
        start: str, 
        end: str
        period: str [day, month, trimestre, semestre, year]

        returns list of tuples start and end time with the respective time format ex: [('2013-01', '%Y-%m'), ('2017-01', '%Y-%m')], 
                datetime of start,
                datetime of end
    '''

    if period == 'month':
        time_period = [(start, '%Y-%m'), (end, '%Y-%m')]
        return time_period, datetime.strptime(*time_period[0]), datetime.strptime(*time_period[1])
    else:
        print('not implemented yet!')
        return None


    
# def plot_users_per_month(data, user_col='user_id', time_col='year-month'):
#     '''
#         Plot users per month
#         Data: interactions dataset. must have a user_col and time_col column.
#     '''
#     data[[user_col, time_col]]\
#     .drop_duplicates()\
#         .groupby(time_col)\
#             .count().plot(title='how many different users per month');

# def plot_interactions_per_month(data, item_col='item_id', time_col='year-month'):
#     data[[item_col, time_col]]\
#         .groupby(time_col)\
#             .count().plot(title='how many interactions per month');