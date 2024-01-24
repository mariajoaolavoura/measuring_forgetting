from sys import path
path.insert(0, '/home/kpfra/streamRec-forgetting/elliot_experiments/')

from source import _elliot_utils
from elliot.run import run_experiment

import tensorflow as tf
from tensorflow.python.client import device_lib
tf.autograph.set_verbosity(5)

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
device_lib.list_local_devices()

def run(path_init_config_file:str, path_to_datasets:str) -> None:
    '''
    path_init_config_file: path to .yml config file for train test of bucket 0 holdout 0.
    \tThe datasets' name strings must contain the bucket and holdout indexes in the format: _b0 and _h0.
    \tThe datasets' name strings must be equal to the ones passed to the .yml config file
    '''
    path_to_config_file = path_init_config_file
    n_buckets = _elliot_utils.getBucketsNumber(path_to_datasets)
    results_list = []
    for nb in range(n_buckets):
        print('\n', f'Running Training for Bucket {nb} - Holdout {nb}'.center(100,'*'), end='\n') 
        model_tup = None
        for nh in range(n_buckets):
            print('\n', f'Running Experiment - {nh}'.center(100,'*'), end='\n') 
            # Run Experiment based on config file
            run_experiment(path_to_config_file)
            if nh < n_buckets-1:
                # create config file for next holdout to be evaluated
                path_to_config_file, path_to_results, model_tup = _elliot_utils.setNewConfig(
                    path_to_config_file = path_to_config_file,
                    model_tup=model_tup 
                    )
        # create config file for next buckets-holdout pair to train on
        if nb < n_buckets-1:
            path_to_config_file = _elliot_utils.setNewBucketConfig(
                path_to_config_file=path_init_config_file,
                bucket_idx=nb+1
                )
        # Refactor results as a pandas table and store them
        results_list = _elliot_utils.buildResults(path_to_results, results_list) # type: ignore
    
    _elliot_utils.storeResults(results_list, path_to_config_file)
    print(100*'*',end='\n')    
    print('\n', 'End of Experiments'.center(100,'*'), end='\n')


if __name__ == '__main__':
    path_to_config_file = "/home/kpfra/streamRec-forgetting/notebooks/elliot_experiments/elliot_example/elliot_example_configuration_b0_h0.yml"
    path_to_datasets = '/home/kpfra/streamRec-forgetting/elliot_experiments/elliot_example/datasets/'
    run(path_to_config_file, path_to_datasets)