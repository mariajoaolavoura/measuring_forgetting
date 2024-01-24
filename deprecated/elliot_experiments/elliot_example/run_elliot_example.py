from sys import path
path.insert(0, '/home/kpfra/streamRec-forgetting/elliot_experiments')
from datetime import datetime
from source import experiment, data_processing

'''
To run an experiment, provide:
    configuration file for the first experiment (training with bucket 0 and holdout 0)
    original dataset, to be sampled - then, provide samples
'''

path_to_config_file = "/home/kpfra/streamRec-forgetting/elliot_experiments/elliot_example/elliot_example_configuration_b0_h0.yml"
path_to_datasets = '/home/kpfra/streamRec-forgetting/elliot_experiments/elliot_example/datasets/'
path_to_original_df = '/home/kpfra/streamRec-forgetting/notebooks/output/movielens_dump/sampled_movielens.csv'

# get sampled data
data_processing.process_sample(
    data_path=path_to_original_df,
    path_to_datasets=path_to_datasets,
    user_col='UserID',
    item_col='ItemID',
    date_conversion_function=lambda x: datetime.strptime(x, '%Y-%m-%d %X')
    )

# Training/Evaluation
experiment.run(path_to_config_file, path_to_datasets)