from sys import path
path.insert(0, '/home/kpfra/streamRec-forgetting/elliot_experiments')
from datetime import datetime
from source import data_processing

path_to_datasets = '/home/kpfra/streamRec-forgetting/elliot_experiments/elliot_movielens/datasets'
path_to_original_df = '/home/kpfra/streamRec-forgetting/notebooks/output/movielens_dump/sampled_movielens.csv'

# get sampled data
data_processing.process_sample(
    data_path=path_to_original_df,
    path_to_datasets=path_to_datasets,
    user_col='UserID',
    item_col='ItemID',
    date_conversion_function=lambda x: datetime.strptime(x, '%Y-%m-%d %X'),
    test_run=True
    )