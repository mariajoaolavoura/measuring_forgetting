from data import ImplicitData
from recommenders_implicit import *
# import numpy as np
# import pandas as pd
import time
# import random

import joblib

class EvalHoldout:
    # TODO: Documentation

    def __init__(self, model: Model, holdout: ImplicitData, metrics: list = ["Recall@N"], N_recommendations: int = 20, default_user: str = 'none'):
        '''
        Class used to evaluate tabular data constructed from sampled stream data.

        Keyword arguments:
        model -- model to be evaluated
        holdout -- tabular data constructed from sampled stream data. Instance of ImplicitData.
        metrics -- only Recall@N for now.
        N_recommendations -- number of recommendations to be used in computing the metric.
        default_user -- str. One of: random, average, or median. If user is not present in model (new user) user factors are generated.
        '''
        # TODO: Input checks
        self.model = model
        self.holdout = holdout
        self.metrics = metrics
        self.N_recommendations = N_recommendations
        self.default_user = default_user


#     def EvaluateTime(self):
#         results = dict()
#         time_get_tuple = []
#         time_recommend = []
#         time_eval_point = []

#         for metric in self.metrics:
#             results[metric] = []

#         for i in range(self.holdout.size):
#             start_get_tuple = time.time()
#             uid, iid = self.holdout.GetTuple(i)
#             end_get_tuple = time.time()
#             time_get_tuple.append(end_get_tuple - start_get_tuple)

#             if iid not in self.model.data.GetUserItems(uid, False):
#                 start_recommend = time.time()
#                 reclist = self.model.Recommend(user = uid, n = self.N_recommendations, default_user=self.default_user) # Experimentar com outro default_user???
#                 end_recommend = time.time()
#                 time_recommend.append(end_recommend - start_recommend)

#                 start_eval_point = time.time()
#                 results[metric].append(self.__EvalPoint(iid, reclist))
#                 end_eval_point = time.time()
#                 time_eval_point.append(end_eval_point - start_eval_point)


#         results['time_get_tuple'] = time_get_tuple
#         results['time_recommend'] = time_recommend
#         results['time_eval_point'] = time_eval_point

#         return results   

    def Evaluate(self, exclude_known_items: bool = True ):
        results = {'time_get_tuple':[], 'time_recommend': [], 'time_eval_point': []}

        for metric in self.metrics:
            results[metric] = []

        for i in range(self.holdout.size):
            # GetTuple
            start_get_tuple = time.time()
            uid, iid = self.holdout.GetTuple(i) # get external IDs
            end_get_tuple = time.time()
            results['time_get_tuple'].append(end_get_tuple - start_get_tuple)
            # Recommend
            start_recommend = time.time()
            reclist = self.model.Recommend(user = uid, n = self.N_recommendations, exclude_known_items = exclude_known_items, default_user=self.default_user)
            # print('@EvalHoldout.Evaluate(), reclist:\n',reclist)
            end_recommend = time.time()
            results['time_recommend'].append(end_recommend - start_recommend)
            # EvalPoint
            if len(reclist): # if user has been seen by model, add result
                start_eval_point = time.time()
                results[metric].append(self.__EvalPoint(iid, reclist))
                end_eval_point = time.time()
                results['time_eval_point'].append(end_eval_point - start_eval_point)
#             else:
#                 print(uid, 'user not seen')
        return results
    
    
    def Evaluate_Save(self, bucket_number:int, holdout_number:int, filepath:str, exclude_known_items:bool=True):
        results = {'time_get_tuple':[], 'time_recommend': [], 'time_eval_point': []}

        rec_lists = []

        for metric in self.metrics:
            results[metric] = []

        for i in range(self.holdout.size):
            # GetTuple
            start_get_tuple = time.time()
            uid, iid = self.holdout.GetTuple(i) # get external IDs
            end_get_tuple = time.time()
            results['time_get_tuple'].append(end_get_tuple - start_get_tuple)
            # Recommend
            start_recommend = time.time()
            reclist = self.model.Recommend(user = uid, n = self.N_recommendations, exclude_known_items = exclude_known_items, default_user=self.default_user)
            
            end_recommend = time.time()
            results['time_recommend'].append(end_recommend - start_recommend)
            
            # this is the difference
            rec_lists += [reclist]

            # EvalPoint
            if len(reclist): # if user has been seen by model, add result
                start_eval_point = time.time()
                results[metric].append(self.__EvalPoint(iid, reclist))
                end_eval_point = time.time()
                results['time_eval_point'].append(end_eval_point - start_eval_point)
#             else:
#                 print(uid, 'user not seen')
                
        # this is the difference
        joblib.dump(rec_lists, 
                    filepath+'rec_lists_b'+str(bucket_number)+'_h'+str(holdout_number)+'.joblib')  
        
        return results

    def __EvalPoint(self, item_id, reclist):
        result = 0
        if len(reclist) == 0:
            return 0
        for metric in self.metrics:
            if metric == "Recall@N":
                result = int(item_id in reclist[:self.N_recommendations,0])
        return result