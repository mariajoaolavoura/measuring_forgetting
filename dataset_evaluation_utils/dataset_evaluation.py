from datetime import datetime, timedelta

from eval_implicit import EvaluateAndStore

import joblib
import pandas as pd
from pandas import Timestamp, date_range
from pandas.tseries.offsets import MonthBegin, MonthEnd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('whitegrid')

# -----------------------------------------
# Dataset evaluation

# timewise evaluation
def get_interactions_info(data, user_col, quarter_info=False, semester_info=False):
    '''
    Input is data dataframe, which contains a 'date' column.
    user_col is the name of the column that contains users IDs.
    Returns:
    user_presence_df, user_month_interactions, trimestres, user_trimestre_interactions, semestres, user_semestre_interactions
    '''
    #counting user interactions per month
    user_month_interactions = data.groupby(by=[user_col, 'date']).count().iloc[:, 0]
    user_month_interactions.name = 'count'
    user_month_interactions = user_month_interactions.reset_index()
    user_month_interactions.sort_values(by=['date'], ascending=[True], inplace=True)

    # defining quarter and semester intervals in dataset

    # https://pandas.pydata.org/docs/user_guide/timeseries.html#timeseries-offset-aliases
    # https://stackoverflow.com/questions/69714505/how-can-generate-trimester-dates-in-python
    start_ts, end_ts = Timestamp(user_month_interactions['date'].unique()[0]), Timestamp(user_month_interactions['date'].unique()[-1])
    
    if quarter_info:
        starts = date_range(start_ts, end_ts - MonthEnd(3), freq='3M') + MonthBegin(-1)
        # starts = date_range(start_ts, end_ts, freq='3M') + MonthBegin(-1)
        ends = date_range(start_ts + MonthEnd(3), end_ts, freq='3M')
        trimestres = list(zip(starts, ends))
        user_trimestre_interactions = dict()
    
    if semester_info:
        starts = date_range(start_ts, end_ts - MonthEnd(6), freq='6M') + MonthBegin(-1)
        ends = date_range(start_ts + MonthEnd(6), end_ts, freq='6M')
        semestres = list(zip(starts, ends))
        user_semestre_interactions = dict()

    # verifying user presence in months, quarter, and semester
    user_presence_percentage = []
    # user_presence_map = dict()
    c = 0
    for u in user_month_interactions[user_col].unique():
        progress = round( 100*( c/user_month_interactions[user_col].nunique() ), 4 )
        if progress%5==0:
            print( progress, '%' )
        c+=1
        
        uidx = user_month_interactions[user_col] == u
        month_presence = user_month_interactions.loc[uidx, 'date'].nunique() / user_month_interactions['date'].nunique()
        
        if quarter_info:
            trimestre_presence = np.repeat(False, len(trimestres)+1 )
            trimestre_count = np.zeros( len( trimestres )+1 )
        
        if semester_info:
            semestre_presence = np.repeat(False, len(semestres)+1 )
            semestre_count = np.zeros( len( semestres )+1 )
        
        if quarter_info or semester_info:
            for udt in user_month_interactions.loc[uidx, 'date']:
                if quarter_info:
                    idx = np.where( [ t[0]<=Timestamp( udt )<=t[1] for t in trimestres ] + [Timestamp( udt ) > trimestres[-1][1]] )[0]
                    trimestre_presence[idx] = True
                    trimestre_count[idx] += 1
                if semester_info:
                    idx = np.where( [ t[0]<=Timestamp( udt )<=t[1] for t in semestres ] + [Timestamp( udt ) > semestres[-1][1]] )[0]
                    semestre_presence[idx] = True
                    semestre_count[ idx ] += 1
                    
            if quarter_info: user_trimestre_interactions[u] = trimestre_count
            if semester_info: user_semestre_interactions[u] = semestre_count
        
        # storing trues and falses for each user ( here, we know exactly where user appears)
        # user_presence_map[u] = [user_month_interactions.loc[uidx, 'date'].unique(), trimestre_presence, semestre_presence]
        # storing user occurence in % of intervals they occur
        if quarter_info and semester_info:
            user_presence_percentage.append(
                [u, month_presence, sum(trimestre_presence)/len(trimestre_presence), sum(semestre_presence)/len(semestre_presence)] )
            cols = ['UserID', 'month_%', 'trimestre_%', 'semestre_%']
        elif quarter_info:
            user_presence_percentage.append(
                [u, month_presence, sum(trimestre_presence)/len(trimestre_presence)])
            cols = ['UserID', 'month_%', 'trimestre_%']
        elif semester_info:
            user_presence_percentage.append(
                [u, month_presence, sum(semestre_presence)/len(semestre_presence)] )
            cols = ['UserID', 'month_%', 'semestre_%']
        else:
            user_presence_percentage.append(
                [u, month_presence])
            cols = ['UserID', 'month_%']

    # building DF from presence percentage
    user_presence_df = pd.DataFrame(
        user_presence_percentage,
        columns=cols
        ).sort_values(by='month_%', ascending=False)
    user_presence_df.reset_index(drop=True, inplace=True)
    # building DF with counts of quarter and semester interactions
    if quarter_info and semester_info:
        user_trimestre_interactions = pd.DataFrame( user_trimestre_interactions ).T#.reset_index()
        user_semestre_interactions = pd.DataFrame( user_semestre_interactions ).T#.reset_index()

        #user_trimestre_interactions.columns = user_month_interactions.columns
        #user_semestre_interactions.columns = user_month_interactions.columns

        return user_presence_df, user_month_interactions, trimestres, user_trimestre_interactions, semestres, user_semestre_interactions
    
    elif quarter_info:
        user_trimestre_interactions = pd.DataFrame( user_trimestre_interactions ).T
        return user_presence_df, user_month_interactions, trimestres, user_trimestre_interactions
    
    elif semester_info:
        user_semestre_interactions = pd.DataFrame( user_semestre_interactions ).T
        return user_presence_df, user_month_interactions, semestres, user_semestre_interactions
    
    else:
        return user_presence_df, user_month_interactions

def plot_interactions_per_month(data, dataset_name):
    '''
    Plot interactions per month
    Data: interactions dataset. must have a 'date' column.
    dataset_name: used to store the image in the folder images. path: images/user_bucket_analysis/{dataset_name}_interactions_year_month.png
    '''
    interactions_per_month = data.groupby(by=['date']).count().iloc[:, 0]
    interactions_per_month.name = 'count'
    interactions_per_month=interactions_per_month.reset_index()
    fig, ax = plt.subplots(figsize=(6,10))
    fig = sns.barplot(x='count', y='date', data=interactions_per_month, color='blue', ax=ax )
    ax.set_yticklabels(labels=list( interactions_per_month['date'].dt.year.astype(str) +'-'+ interactions_per_month['date'].dt.month.astype(str) ))
    plt.title(f'{dataset_name}: Interactions per year-month')    
    plt.savefig(f'images/user_bucket_analysis/{dataset_name}_interactions_year_month.png')

def plot_user_presence_distribution(user_presence_df, dataset_name):
    '''
    Plots presence distribution for month, quarter, and semester from user_presence_df.
    user_presence_df is the output of the function 'get_interactions_info'
    dataset_name: used to store the image in the folder images. path: 'images/user_bucket_analysis/{dataset_name}_user_presence_distribution.png'
    '''
    # fig, ax = plt.subplots(3,1, figsize=(17,4))
    # user_presence_df['month_%'].plot(kind='hist', ax=ax[0], title='user month presence')
    # if 'trimestre_%' in user_presence_df.columns:
    #     user_presence_df['trimestre_%'].plot(kind='hist', ax=ax[1], title='user quarter presence')    
    # if 'semestre_%' in user_presence_df.columns:
    #     user_presence_df['semestre_%'].plot(kind='hist', ax=ax[2], title='user semester presence')   
    
    user_presence_df.plot(kind='hist', subplots=True, title='user presence') 
    plt.suptitle(f'{dataset_name}: User presence distribution')
    plt.savefig(f'images/user_bucket_analysis/{dataset_name}_user_presence_distribution.png');

def plot_interactions_per_qns(user_interactions, date_range, dataset_name=None, type_of_range='quarter'):
    '''
    Plots number of interactions per quarter or semester from user interactions.
    user_interactions is the output of the function 'get_interactions_info'
    date_range is either trimestres or semestres, outputs of the function 'get_interactions_info'
    dataset_name: used to store the image in the folder images. path: 'images/user_bucket_analysis/{dataset_name}_interactions_quarter.png'
    type_of_range is used to create the plot title.
    '''
    dates = [datetime.strftime( d[0], '%Y-%m' ) for d in date_range] + [datetime.strftime( date_range[-1][1]+timedelta(days=1), '%Y-%m' )]
    values = user_interactions.sum()
    values = values.reset_index()
    values.columns = ['date', 'count']
    values['date'] = dates
    fig, ax = plt.subplots(figsize=(6,10))
    fig = sns.barplot(x='count', y='date', data=values, color='blue', ax=ax )
    if dataset_name:
        plt.title(f'{dataset_name}: Interactions per {type_of_range}')
        plt.savefig(f'images/user_bucket_analysis/{dataset_name}_interactions_quarter.png')

def get_frequent_users(user_presence_df, frequency_threshold = 0.8):
    '''
    Get frequent users per month, quarter, and semester from user_presence_df. 
    user_presence_df is the output of the function 'get_interactions_info'.
    Frequent users = users with that occur in at least 'frequency_threshold' of intervals. 
    Return lists with frequent users (in months, quarters, and semesters).
    '''
    # getting frequent users per month
    frequent_users_month = user_presence_df[user_presence_df['month_%']>=frequency_threshold ]['UserID'].values
    # percentage of users that are *frequent in months
    print(
f'''{len(frequent_users_month)} users \
of {user_presence_df['UserID'].nunique()} \
({round( 100*len(frequent_users_month) / user_presence_df.shape[0], 3)}%) occurr in {frequency_threshold*100}% or more months.''' #  (of {user_month_interactions["date"].nunique()})
    )

    if 'trimestre_%' in user_presence_df.columns:
        # getting frequent users per quarter
        frequent_users_trimestre = user_presence_df[user_presence_df['trimestre_%']>=frequency_threshold ]['UserID'].values
        # percentage of users that are *frequent in quarters
        print(
f'''{len(frequent_users_trimestre)} users \
of {user_presence_df['UserID'].nunique()} \
({round( 100*len( frequent_users_trimestre ) / user_presence_df.shape[0], 3)}%) occurr in {frequency_threshold*100}% or more quarters.''' #  (of {len( trimestres) +1})
        )

    if 'semestre_%' in user_presence_df.columns:
        # getting frequent users per semester
        frequent_users_semestre = user_presence_df[user_presence_df['semestre_%']>=frequency_threshold ]['UserID'].values
        # percentage of users that are *frequent in semesters
        print(
f'''{len(frequent_users_semestre)} users \
of {user_presence_df['UserID'].nunique()} \
({round( 100*len( frequent_users_semestre ) / user_presence_df.shape[0], 3)}%) occurr in {frequency_threshold*100}% or more semesters.''' #  (of {len( semestres) +1})
        )
    
    if 'trimestre_%' in user_presence_df.columns and 'semestre_%' in user_presence_df.columns:
        return frequent_users_month, frequent_users_trimestre, frequent_users_semestre
    elif 'trimestre_%' in user_presence_df.columns:
        return frequent_users_month, frequent_users_trimestre
    elif 'semestre_%' in user_presence_df.columns:
        return frequent_users_month, frequent_users_semestre
    else:
        return frequent_users_month

def get_frequent_user_statistics(interactions_df, frequent_users_list):
    '''
    returns series with statistics from dataframe with user interactions per quarter or semester, or buckets.
    interactions_df is either user_trimestre_interactions, user_semestre_interactions, or user_bucket_interactions - outputs of get_interactions_info and get_fixed_buckets_info
    returns:
        holdout_users_per_interval - number of frequent users with at least 1 interactions, in each interval
        median_interactions_per_interval - median number of user interactions, in each interval
    '''
    # this is the number of users that MAY be used for testing (holdouts) in each interval - they got to be seen previously too.
    # ** they have at least 1 interactions in these intervals - to be used for testing, if user has been seen.
    holdout_users_per_interval = (interactions_df.loc[frequent_users_list] >= 1).sum()
    holdout_users_per_interval.name = 'freq_users_at_least_1_interaction'
    # median of user interactions per interval
    median_interactions_per_interval = interactions_df.loc[frequent_users_list].median(axis=0)
    median_interactions_per_interval.name = 'median_freq_user_interactions'
    return pd.concat( [holdout_users_per_interval, median_interactions_per_interval], axis=1 )

# fixed bucket size evaluation

def get_bucket_intervals(data, bucket_size):
    '''
    get bucket interval indexes from data and bucket_size.
    '''
    interval_start = np.arange(0, data.shape[0], int(bucket_size))
    interval_end = np.arange(int(bucket_size), data.shape[0]+int(bucket_size), int(bucket_size))
    return interval_start, interval_end

def plot_users_per_fixed_bucket(data, user_col, interval_start, interval_end):
    '''
    barplot of unique users per bucket
    '''
    unique_users_per_bucket = [data.iloc[i:j][user_col].nunique() for i, j in zip(interval_start, interval_end) ]
    sns.barplot(x=np.arange(1, len( interval_start )+1), y=unique_users_per_bucket, color='blue')
    plt.hlines(y=data[user_col].nunique(), xmin=0, xmax=len( interval_start )-1, label='Number of unique users', color='red');
    plt.title('Users per bucket')
    plt.legend(loc='center right')

def get_fixed_buckets_info(data, user_col, interval_start, interval_end):
    '''
    Get information from fixed buckets.
    data is a dataframe with user item interactions. it should present the column 'date'
    user_col is the name of the column that has user IDs.
    interval_start, interval_end are obtained from the function get_bucket_intervals

    returns:
        user_presence_df - dataframe with number of user interactions per bucket
        dates_fixed_buckets_df - dataframe with number of dates per bucket
    '''
    user_bucket_interactions = { u:np.zeros(len(interval_start)) for u in data[user_col].unique() }
    for b, (i, j) in enumerate( zip(interval_start, interval_end) ):
        for u in data[user_col].unique():
            user_bucket_interactions[u][b] += ( data.iloc[i:j][user_col] == u ).sum()

    user_bucket_interactions = pd.DataFrame(user_bucket_interactions).T

    try:
        dates_fixed_buckets = { d:np.zeros(len(interval_start)) for d in data['date'].unique() }
        for b, (i, j) in enumerate( zip(interval_start, interval_end) ):
            for d in data['date'].unique():
                dates_fixed_buckets[d][b] += ( data.iloc[i:j]['date'] == d ).sum()
        dates_fixed_buckets_df = pd.DataFrame(dates_fixed_buckets).T
        columns = ['date'] + [f'bucket_{i}' for i in range(dates_fixed_buckets_df.shape[1])]
        dates_fixed_buckets_df = dates_fixed_buckets_df.reset_index()
        dates_fixed_buckets_df.columns = columns
    except:
        print("No 'date' columns")
        return user_bucket_interactions

    return user_bucket_interactions, dates_fixed_buckets_df

def get_frequent_users_fixed_buckets(user_presence_df, frequency_threshold):
    '''
    Get frequent users per bucket from user_presence_df. 
    user_presence_df is the output of the function 'get_fixed_buckets_info'.
    Frequent users = users with that occur in at least 'frequency_threshold' of buckets. 
    Return lists with frequent users
    '''
    # getting frequent users per bucket
    frequent_users_bucket = user_presence_df[ ( user_presence_df != 0 ).sum(axis=1) / user_presence_df.shape[1] >= frequency_threshold ].index
    # percentage of users that are *frequent in buckets
    print(
f'''{len(frequent_users_bucket)} users \
of {user_presence_df.shape[0]} \
({round( 100*len( frequent_users_bucket ) / user_presence_df.shape[0], 3)}%) occur in {frequency_threshold*100}% or more buckets.''' #  (of {len( interval_start)})
    )
    return frequent_users_bucket

def plot_user_interactions_per_bucket(user_bucket_interactions_df, frequent_users_bucket):
    '''
    barplot of frequent user interactions per bucket
    '''
    user_bucket_interactions_df.loc[ frequent_users_bucket ].sum().plot(kind='bar')

def plot_timestamps_per_bucket(dates_fixed_buckets_df):
    '''
    barplots of timestamps presence per bucket
    '''
    fig, ax = plt.subplots(1, dates_fixed_buckets_df.shape[1]-1, figsize=(25, 10), sharey=True, sharex=True)
    for i, bckt in enumerate(dates_fixed_buckets_df.columns[1:]):
        sns.barplot(x=bckt, y='date', data=dates_fixed_buckets_df.reset_index(), ax=ax[i])

#------------------------------------------
# Holdout evaluation - for when holdouts are available in EvaluateAndStore objects

def get_bucket_map(eval_object:EvaluateAndStore):
    '''
    Create a dict that maps users (external id) to the buckets they are in (from 1 to last).
    Used with EvaluateAndStore objects
    '''
    user_bucket_map = {user:[] for user in eval_object.data.userset}
    for user in eval_object.data.userset:
        for i, bucket in enumerate( eval_object.holdouts ):
            if user in bucket.userset:
                user_bucket_map[user].append(i+1)
    return user_bucket_map

def median_user_presence_per_bucket(eval_object:EvaluateAndStore, user_bucket_map:dict):
    '''
    On median terms, on how many buckets a user is.
    Used with EvaluateAndStore objects
    '''
    presence_list = [len(user_bucket_map[user]) for user in eval_object.data.userset]
    return np.median( presence_list )

def plot_bucket_size(eval_object:EvaluateAndStore, dataset_name, filename=None):
    ''' 
    Used with EvaluateAndStore objects
    '''
    bucket_size = pd.Series( [bucket.size for bucket in eval_object.holdouts] )
    bucket_size = bucket_size.reset_index()
    bucket_size.columns = ['Bucket', 'Size']
    bucket_size['Bucket'] = bucket_size['Bucket']+1
    plt.figure(figsize=(10,5))
    sns.barplot(x='Bucket', y='Size', data=bucket_size, color='b')
    # sns.lineplot(data=np.repeat(n_users, n_users_bucket.shape[0]), label='total users', color='orange')
    plt.title(f'Bucket size - {dataset_name}')
    if filename:
        plt.savefig(f'images/user_bucket_analysis/{filename}')

def plot_n_users_per_bucket(eval_object:EvaluateAndStore, dataset_name:str, filename:str=None):
    '''
    Used with EvaluateAndStore objects
    '''
    n_users = len( eval_object.data.userset )
    n_users_bucket = pd.Series( [len( bucket.userset ) for bucket in eval_object.holdouts] )
    n_users_bucket = n_users_bucket.reset_index()
    n_users_bucket.columns = ['Bucket', 'N_users']
    n_users_bucket['Bucket'] = n_users_bucket['Bucket']+1
    plt.figure(figsize=(10,5))
    sns.barplot(x='Bucket', y='N_users', data=n_users_bucket, color='b', label='users per bucket')
    sns.lineplot(data=np.repeat(n_users, n_users_bucket.shape[0]), label='total users', color='orange')
    plt.title(f'Users per bucket - {dataset_name}');
    if filename:
        plt.savefig(f'images/user_bucket_analysis/{filename}')

def store_user_presence(dataset_name, median_user_presence):
    '''
    also used on the computation of median user presence per bucket - in the case of existing holdouts from the EvaluateAndStore outputs.
    '''
    with open('output/bucket_info_dump/median_user_presence.txt', 'a') as file:
        file.write(f'{dataset_name}: in median terms, users are present in {median_user_presence} bucket\n')