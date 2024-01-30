import os
import json
import yaml
import sys
import shutil

import pandas as pd
import numpy as np

'''
Utilities to run elliot experiments over buckets and holdouts.
'''

def getBestModelParams(path_to_results):
    for item in os.listdir(path_to_results + 'performance/'):
        if 'best' in item:
            with open(path_to_results + 'performance/' + item) as file:
                best_model_info = json.load(file)
            break
    best_model_params = best_model_info[1]['configuration'] # type: ignore
    model = best_model_params['name'].split('_')[0]
    # UNDO THIS LATER
    try:
        del best_model_params['name'], best_model_params['best_iteration'] 
    except KeyError as e:
        sys.stderr.write(f'Not able to remove {e} parameter. Possible cause: experiment being rerun over existing performance results.')
    return model, best_model_params

def getBucketsNumber(path_to_datasets):
    files = os.listdir(path_to_datasets)
    n_buckets = max( [int(f[f.find('_')+2]) for f in files if f.endswith('.csv')] ) + 1
    return n_buckets

def setNewConfig( path_to_config_file, model_tup=None):
    # assert model_tup or path_to_results, 'At least one of model_tup or path_to_results must be passed'
    with open(path_to_config_file, "r") as stream:
        try:
            yaml_file = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            sys.exit()
     
    bucket_idx = int(path_to_config_file[path_to_config_file.find('_b')+2]) # index of the bucket
    holdout_idx = int(path_to_config_file[path_to_config_file.find('_h')+2]) # index of the last holdout tested
    
    slice_pos = yaml_file['experiment']['path_output_rec_result'].rfind('/') + 1
    path_to_results = yaml_file['experiment']['path_output_rec_result'][:slice_pos]
    print('\n', f'Results will be stored in {path_to_results}'.center(100,'*'), end='\n')    

    try:
        model, best_model_params = model_tup # type: ignore
    except:
        model, best_model_params = getBestModelParams(path_to_results)

    if bucket_idx == 0:
        holdout_idx = str(holdout_idx + 1)
    elif holdout_idx == bucket_idx:
        holdout_idx = str(0)
    elif holdout_idx < bucket_idx:
        if (bucket_idx - holdout_idx) == 1:
            holdout_idx = str(holdout_idx + 2)
        else:
            holdout_idx = str(holdout_idx + 1)
    else:
        holdout_idx = str(holdout_idx + 1 )    
        
    # Update variables and files according to new holdout idx
    path_to_config_file = path_to_config_file[:path_to_config_file.find('_h')+2] + holdout_idx + path_to_config_file[path_to_config_file.find('_h')+3:]
    # update test path
    test_path = yaml_file['experiment']['data_config']['test_path']
    yaml_file['experiment']['data_config']['test_path'] = test_path[:test_path.find('_h')+2] + holdout_idx + test_path[test_path.find('_h')+3:]
    # update model params using previous best model
    yaml_file['experiment']['models'][model].update(best_model_params)
    yaml_file['experiment']['models'][model]['meta'].update({'restore': True})       
    yaml_file['experiment']['models'][model]['meta'].update({'save_weights': False})   
    # update dataset name
    dataset_name = yaml_file['experiment']['dataset']
    yaml_file['experiment']['dataset'] = dataset_name[:dataset_name.find('_h')+2] + holdout_idx
    # yaml_file['experiment']['models'][model]['meta'].update({'save_recs': True}) # TEMP

    with open(path_to_config_file, "w") as stream:
        yaml.safe_dump(yaml_file, stream)

    return path_to_config_file, path_to_results, (model, best_model_params)

def setNewBucketConfig(path_to_config_file, bucket_idx):
    with open(path_to_config_file, "r") as stream:
        try:
            yaml_file = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            sys.exit()
    # Update variables and files according to new training bucket and holdout
    bucket_idx = str(bucket_idx)
    paths_list = ['path_output_rec_result', 'path_output_rec_weight', 'path_output_rec_performance', 'path_log_folder']
    for tag in ['_h', '_b']:
        #update config file path
        path_to_config_file = path_to_config_file[:path_to_config_file.find(tag)+2] + bucket_idx + path_to_config_file[path_to_config_file.find(tag)+3:]
        # update output config:
        for p in paths_list:
            config_path = yaml_file['experiment'][p]
            yaml_file['experiment'][p] = config_path[:config_path.find(tag)+2] + bucket_idx + config_path[config_path.find(tag)+3:]
        # update dataset name
        dataset_name = yaml_file['experiment']['dataset']
        yaml_file['experiment']['dataset'] = dataset_name[:dataset_name.find(tag)+2] + bucket_idx + dataset_name[dataset_name.find(tag)+3:]
    # update train and test paths
    train_path = yaml_file['experiment']['data_config']['train_path']
    test_path = yaml_file['experiment']['data_config']['test_path']
    yaml_file['experiment']['data_config']['train_path'] = train_path[:train_path.find('_b')+2] + bucket_idx + train_path[train_path.find('_b')+3:]
    yaml_file['experiment']['data_config']['test_path'] = test_path[:test_path.find('_h')+2] + bucket_idx + test_path[test_path.find('_h')+3:]

    with open(path_to_config_file, "w") as stream: # type: ignore
        yaml.safe_dump(yaml_file, stream)

    return path_to_config_file # type: ignore

def buildResults(path_to_results, results_list):
    result_files = []

    for item in os.listdir(path_to_results + 'performance/'):
        if f'_b' in item: # if tag for bucket and holdout index are present, read results file
            result_files.append(item)
    if not results_list:
        empty_results = pd.DataFrame( np.zeros(shape = (len(result_files), len(result_files))) )
        results_list = [empty_results.copy(), empty_results.copy(), empty_results.copy()]

    for item in result_files:
        bucket_idx = int(item[item.find('_b')+2]) # index of the bucket
        holdout_idx = int(item[item.find('_h')+2]) # index of the holdout tested
        df = pd.read_csv(path_to_results + 'performance/' + item, sep='\t', index_col=False).fillna(0)        
        results_list[0].iloc[bucket_idx, holdout_idx] = df.loc[0, 'nDCG']
        results_list[1].iloc[bucket_idx, holdout_idx] = df.loc[0, 'Precision']
        results_list[2].iloc[bucket_idx, holdout_idx] = df.loc[0, 'Recall']

    return results_list

def storeResults(results_list, path_to_config_file):
    store_path = path_to_config_file[:path_to_config_file.rfind('/')]
    try:
        os.mkdir(f'{store_path}/results/')
    except FileExistsError as fee:
        sys.stderr.write(f'{os.strerror(fee.errno)}: {fee}\n')
        sys.stderr.write('Possible cause: experiment being rerun over existing results.\n')
        while True:
            ch = input('Substitute results? y/n').lower()
            if ch=='y':
                shutil.rmtree(f'{store_path}/results/')
                os.mkdir(f'{store_path}/results/')
                break
            if ch=='n':
                print('End of routine.')
                return   
        
    for result_df, metric in zip(results_list, ('nDCG', 'Precision', 'Recall') ): 
        result_df.to_csv(f'{store_path}/results/{metric}_results_matrix.csv')
    

if __name__ == '__main__':
    path_to_config_file = "/home/kpfra/streamRec-forgetting/notebooks/elliot_experiments/elliot_example/elliot_example_configuration_b0_h0.yml"
    # print(setNewBucketConfig(path_to_config_file, 1))
    # print(setNewBucketConfig(path_to_config_file, 2))
    # print(setNewBucketConfig(path_to_config_file, 10))
    # path_to_config_file, path_to_results, (model, best_model_params) = setNewConfig( path_to_config_file, model_tup=None)
    # buildResults(path_to_results, results_list=None)
    path_to_results = '/home/kpfra/streamRec-forgetting/MultiVae_Movielens_results_b0_h0/'
    results_list = buildResults(path_to_results, results_list=None)
    path_to_results = '/home/kpfra/streamRec-forgetting/MultiVae_Movielens_results_b1_h1/'
    results_list = buildResults(path_to_results, results_list=results_list)
    path_to_results = '/home/kpfra/streamRec-forgetting/MultiVae_Movielens_results_b2_h2/'
    results_list = buildResults(path_to_results, results_list=results_list)
    storeResults(results_list, path_to_config_file)