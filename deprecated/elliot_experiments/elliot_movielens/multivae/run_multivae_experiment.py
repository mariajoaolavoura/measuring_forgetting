from sys import path
path.insert(0, '/home/kpfra/streamRec-forgetting/elliot_experiments')
from source import experiment

'''
To run an experiment, provide:
    configuration file for the first experiment (training with bucket 0 and holdout 0)
    original dataset, to be sampled - then, provide samples
'''

path_to_config_file = "/home/kpfra/streamRec-forgetting/elliot_experiments/elliot_movielens/multivae/ml_multivae_b0_h0_config.yml"
path_to_datasets = '/home/kpfra/streamRec-forgetting/elliot_experiments/elliot_movielens/datasets'

# Training/Evaluation
experiment.run(path_to_config_file, path_to_datasets)