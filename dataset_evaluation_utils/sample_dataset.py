from datetime import datetime

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

def filter_out_users_with_less_than_k_rates_per_period(df, user_col='user_id', k=5, period='year-month'):
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
        df: pandas DataFrame

        returns a new dataframe
    '''
    data = df.copy()

    data['date'] = data[time_col].apply(lambda x: datetime.utcfromtimestamp(x)) # convert unix timestamp to date
    data = data.sort_values(by='date') # sort by date
    data['year-month'] = data['date'].apply(lambda x: datetime.strptime( str(x.year)+'-'+str(x.month), '%Y-%m' ))
    data['year'] = data['date'].dt.year
    data['month'] = data['date'].dt.month
    data['day'] = data['date'].dt.day

    return data


def sample_time_period(time_period, df, user_col='user_id', time_col='year-month'):
    time_period_start = time_period[0][0]
    time_period_end = time_period[1][0]

    y_filter = (df[time_col] >= datetime.strptime(*time_period[0])) & \
            (df[time_col] < datetime.strptime(*time_period[1]))
    
    df[y_filter][[user_col, time_col]]\
        .groupby([time_col]).count()\
            .plot(title='Count of users per month '+time_period_start+' to '+time_period_end);

    return df[y_filter], time_period_start, time_period_end