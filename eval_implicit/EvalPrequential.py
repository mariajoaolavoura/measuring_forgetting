from data import ImplicitData
from recommenders_implicit import *
import numpy as np
import pandas as pd
import time
import random

class EvalPrequential:

    def __init__(self, model: Model, data: ImplicitData, metrics = ["Recall@N"], N_recommendations=20, seed: int = None):
        # TODO: Input checks
        self.model = model # model object
        self.data = data # stream
        self.metrics = metrics # metrics names
        self.seed = seed # random seed
        self.N_recommendations = N_recommendations


    def EvaluateTime(self, start_eval = 0, count = 0, interleaved = 1): 
        '''
        Evaluates running time for internal processes:
        - get external uid and iid
        - recommend 20 items
        - eval point ...
        - update ...
        
        start_eval - from which ui interaction should the evaluation start (interactions before the 100th are used only for incremental training)*
        count - max number of ui interactions to evaluate on (stream.size at max)**
        interleaved - each interaction has a small chance of being ignored (1 - all are ignored; prob. diminishes when number increases)***
        '''
        results = dict()
        # lists to retain running time
        time_get_tuple = []
        time_recommend = []
        time_eval_point = []
        time_update = []
        random.seed(self.seed)

        if not count:
            count = self.data.size

        count = min(count, self.data.size) # bound 'count' to the min between 'count' and the number of interactions in the stream

        for metric in self.metrics: # creates a key and empty list in the results dictionary for each metric
            results[metric] = []

        for i in range(count): # **
            start_get_tuple = time.time()
            uid, iid = self.data.GetTuple(i) # get external IDs
            end_get_tuple = time.time()
            time_get_tuple.append(end_get_tuple - start_get_tuple) 

            if i >= start_eval and random.random() <= 1/interleaved and i>100: # *, ***
                if iid not in self.model.data.GetUserItems(uid, False): # if iid is not in the users list of interacted items
                    start_recommend = time.time()
                    reclist = self.model.Recommend(uid, 20) # recommend 20 items to uid
                    end_recommend = time.time()
                    time_recommend.append(end_recommend - start_recommend)

                    start_eval_point = time.time()
                    results[metric].append(self.__EvalPoint(iid, reclist)) # evaluate recommendations before updating - Recall@20
                    end_eval_point = time.time()
                    time_eval_point.append(end_eval_point - start_eval_point)
            start_update = time.time()
            self.model.IncrTrain(uid, iid) # perform incremental training
            end_update = time.time()
            time_update.append(end_update - start_update)

        results['time_get_tuple'] = time_get_tuple
        results['time_recommend'] = time_recommend
        results['time_eval_point'] = time_eval_point
        results['time_update'] = time_update

        return results

    def Evaluate(self, start_eval = 0, count = 0, interleaved = 1):
        results = dict()

        if not count:
            count = self.data.size

        count = min(count, self.data.size)

        for metric in self.metrics:
            results[metric] = []

        for i in range(count):
            uid, iid = self.data.GetTuple(i)
            if i >= start_eval and random.random() <= 1/interleaved and i>100:
                reclist = self.model.Recommend(user=uid, n=self.N_recommendations)
                results[metric].append(self.__EvalPoint(iid, reclist))
            self.model.IncrTrain(uid, iid)

        return results

    def __EvalPoint(self, item_id, reclist):
        '''
        Receives current item_id and list of 20 recommendations.
        Returns Recall@20 - 1 if item in recommendations, else 0
        '''
        result = 0
        if len(reclist) == 0:
            return 0
        for metric in self.metrics:
            if metric == "Recall@N":
                #print('reclist', reclist)
                #print('len(reclist)', len(reclist))
                #reclist = [x[0] for x in reclist[:20]]
                result = int(item_id in reclist[:self.N_recommendations,0])
                # print(result)
        return result
