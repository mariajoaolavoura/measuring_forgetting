from eval_implicit import EvalHoldout
from recommenders_implicit import *
import pandas as pd
import numpy as np
import time

import joblib
import os

class EvaluateHoldouts():
    '''
    Instanciation:\n
        \tIncremental training of recommendation model.\n
        \tStore model checkpoints at the end of each bucket.\n
    Methods:\n
        \tEvaluateHoldouts to evaluate models over holdouts - recall@N_recommendations. Known items are excluded by default.
    '''
    def __init__(self, model: Model, buckets, holdouts, ):
        self.model = model
        self.buckets = buckets
        self.holdouts = holdouts
        self.metrics = ["Recall@N"]
#         self.model_checkpoints = []
        self.IncrementalTraining_time_record = {}
        self.EvaluateHoldouts_time_record = {}
#         self._IncrementalTraining()
        self.save_files = False

    def Train_Evaluate(self, N_recommendations=20, exclude_known_items:bool=True, default_user:str='none', verbose=True):
        '''
        Incremental training of recommendation model.
        '''
        cold_start_buckets = len( self.buckets ) - len( self.holdouts )
        self.results_matrix = np.zeros( shape=( len( self.holdouts ), len( self.holdouts ) ) )
        self.verbose=verbose
        for b, bucket in enumerate(self.buckets):
            if self.verbose:
                print(100*'-')
                print(f'Train bucket {b}')
            incrtrain_time = []            
            for i in range(bucket.size):
                uid, iid = bucket.GetTuple(i) # get external IDs
                s = time.time()
                self.model.IncrTrain(uid, iid) # perform incremental training
                f = time.time()
                incrtrain_time.append(f-s)    
            if b >= cold_start_buckets:
                self._EvaluateHoldouts(
                    bucket_number=b-cold_start_buckets,
                    N_recommendations=N_recommendations,
                    exclude_known_items=exclude_known_items,
                    default_user=default_user)
            self.IncrementalTraining_time_record[f'bucket_{b}'] = {
                'size':bucket.size,
                'train time vector':incrtrain_time,
                'avg train time':np.mean(incrtrain_time),
                'total train time':np.sum(incrtrain_time),
            }

    def _EvaluateHoldouts(self, bucket_number, N_recommendations=20, exclude_known_items:bool=True, default_user:str='none'):
        '''
        exclude_known_items -- boolean, exclude known items from recommendation\n
        default_user -- str. One of: none, random, average, or median.\n\tIf user is not present in model (future user) user factors are generated. If none, then no recommendations are made (user wont count for recall)
        '''        
        metric = self.metrics[0]
        for j, hd in enumerate( self.holdouts ):
            if self.verbose:
                print(f'Test Holdout {j}')
            evaluate_time = []
            eh_instance_time = []
            
            eh_instance = EvalHoldout(model=self.model, holdout=hd, metrics=[metric], N_recommendations=N_recommendations, default_user=default_user)
            
            s = time.time()
            results = eh_instance.Evaluate(exclude_known_items=exclude_known_items)
            f = time.time()
            evaluate_time.append(f-s)
            
            result = results[metric]
            del results[metric]
            eh_instance_time.append(results)            
            n_not_seen = hd.size - len(result) # if user was not seen, its not added to recall. May be needed to store difference.
            if n_not_seen and self.verbose:
                print(f'recommendations not made for users in holdout {j} x checkpoint {bucket_number}: {n_not_seen}')
                
            result = sum( result ) / len(result)                
            self.results_matrix[bucket_number, j] = result
            
            self.EvaluateHoldouts_time_record[f'holdout_{j}'] = {
                'size': hd.size,
                'train time vector':evaluate_time,
                'avg model eval time':np.mean(evaluate_time),
                'total train time':np.sum(evaluate_time),
                'EvalHoldout time': eh_instance_time
            }

    def Train_Evaluate_Save(self, 
                            filepath:str,                           
                            N_recommendations=20, 
                            exclude_known_items:bool=True, 
                            default_user:str='none', 
                            verbose=True):
        '''
        Incremental training of recommendation model. Saves model.data
        '''

        self.save_files = True
        self.filepath = filepath
        if not os.path.exists(self.filepath):
            os.makedirs(self.filepath)


        cold_start_buckets = len( self.buckets ) - len( self.holdouts )
        self.results_matrix = np.zeros( shape=( len( self.holdouts ), len( self.holdouts ) ) )
        self.verbose=verbose        
        for b, bucket in enumerate(self.buckets):
            if self.verbose:
                print(100*'-')
                print(f'Train bucket {b}')
            incrtrain_time = []            
            for i in range(bucket.size):
                uid, iid = bucket.GetTuple(i) # get external IDs
                s = time.time()
                self.model.IncrTrain(uid, iid) # perform incremental training
                f = time.time()
                incrtrain_time.append(f-s)    
            
            if b >= cold_start_buckets:      
                # this is the only difference from Train_Evaluate, repetition of code to avoid computing an if-case every time that would not even change 
                bucket_number=b-cold_start_buckets
                joblib.dump(self.model.data, 
                            self.filepath+'model_data_b'+str(bucket_number)+'.joblib')              
                
                self._EvaluateHoldoutsSave(
                    bucket_number=bucket_number,
                    N_recommendations=N_recommendations,
                    exclude_known_items=exclude_known_items,
                    default_user=default_user)
            
            self.IncrementalTraining_time_record[f'bucket_{b}'] = {
                'size':bucket.size,
                'train time vector':incrtrain_time,
                'avg train time':np.mean(incrtrain_time),
                'total train time':np.sum(incrtrain_time),
            }

    
    def _EvaluateHoldoutsSave(self, 
                              bucket_number,
                              N_recommendations=20, 
                              exclude_known_items:bool=True, 
                              default_user:str='none'):
        '''
        exclude_known_items -- boolean, exclude known items from recommendation\n
        default_user -- str. One of: none, random, average, or median.\n\tIf user is not present in model (future user) user factors are generated. If none, then no recommendations are made (user wont count for recall)
        '''        
        metric = self.metrics[0]
        for j, hd in enumerate( self.holdouts ):
            if self.verbose:
                print(f'Test Holdout {j}')
            evaluate_time = []
            eh_instance_time = []
            
            eh_instance = EvalHoldout(model=self.model, holdout=hd, metrics=[metric], N_recommendations=N_recommendations, default_user=default_user)
            
            s = time.time()
            results = eh_instance.Evaluate_Save(bucket_number,
                                                j,
                                                self.filepath,
                                                exclude_known_items=exclude_known_items)
            f = time.time()
            evaluate_time.append(f-s)
            
            result = results[metric]
            del results[metric]
            eh_instance_time.append(results)            
            n_not_seen = hd.size - len(result) # if user was not seen, its not added to recall. May be needed to store difference.
            if n_not_seen and self.verbose:
                print(f'recommendations not made for users in holdout {j} x checkpoint {bucket_number}: {n_not_seen}')
                
            result = sum( result ) / len(result)                
            self.results_matrix[bucket_number, j] = result
            
            self.EvaluateHoldouts_time_record[f'holdout_{j}'] = {
                'size': hd.size,
                'train time vector':evaluate_time,
                'avg model eval time':np.mean(evaluate_time),
                'total train time':np.sum(evaluate_time),
                'EvalHoldout time': eh_instance_time
            }