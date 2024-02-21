
import pandas as pd 
from datetime import datetime
import matplotlib.pyplot as plt


def load_data_print_info(data_path):
    data = pd.read_csv(data_path)

    print('data.shape', data.shape)
    print('number of users', data.user_id.nunique())
    print('number of items', data.item_id.nunique())
    print('number of duplicated user-item interactions', data[['user_id', 'item_id']].duplicated().sum())


    print('\n',data.head())


    # convert timestamp
    data['date'] = data['date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))


    interactions_per_month = data.groupby(by=['date']).count().iloc[:, 0]
    interactions_per_month.name = 'count'
    interactions_per_month=interactions_per_month.reset_index()
    _ = interactions_per_month.copy()
    _['date'] = _['date'].dt.date
    _.groupby('date').sum().plot(kind='bar')
    plt.title('interactions per month');

    return data