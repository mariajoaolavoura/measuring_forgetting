from eval_implicit import EvalPrequential, EvalHoldout
from data import ImplicitData
from recommenders_implicit import *
import numpy as np
# import pandas as pd
# import time
import random
import copy

class EvaluateAndStore(EvalPrequential):

    def __init__(self, model: Model, data: ImplicitData, n_holdouts=20, metrics = ["Recall@N"], N_recommendations=20, seed: int = None):
        super().__init__(model, data, metrics, N_recommendations, seed)        
        self.n_holdouts = n_holdouts
        # self.holdouts = [ImplicitData(user_list=[], item_list=[]) for _ in range(self.n_holdouts)]
        self.holdouts = [[] for _ in range(n_holdouts)]
        self.model_checkpoints = []

    def EvaluateAndStore(self, start_eval = 0, count = 0, store_only=True, default_user='none'): # , interleaved = 1): 
        '''
        Prequential evaluation of recommendation model.
        A number of 'n_holdouts' model states and holdout test sets are stored
        
        start_eval - from which ui interaction should the prequential evaluation start (interactions before the 100th are used only for incremental training)*
        count - max number of ui interactions to evaluate on (stream.size at max)**
        store_only - False if prequential evaluation is to be made (recommendation + comparison steps), True if only storage is to be carried out.
        default_user - str. One of: random, average, or median. If user is not present in model (new user) user factors are generated.
        '''
        # interleaved - each interaction has a small chance of being ignored (1 - all are ignored; prob. diminishes when number increases)*** this is used to make the process faster
        results = dict()
        random.seed(self.seed)

        for metric in self.metrics: # creates a key and empty list in the results dictionary for each metric
            results[metric] = []

        if not count:
            count = self.data.size

        count = min(count, self.data.size) # bound 'count' to the min between 'count' and the number of interactions in the stream
        min_start = 100 # minimum number of interactions before performing recommendations, other than start_eval
        checkpoint_size = (self.data.size - max(min_start, start_eval))//self.n_holdouts
        checkpoint_count = [0, 0] # pos 0 - used to track the checkpoint. pos 1 - used to count how many examples were seen.
        for i in range(count): # **
            checkpoint_count[1] += 1
            uid, iid = self.data.GetTuple(i) # get external IDs
            if i >= start_eval and i >= min_start and not store_only: # *, ***
                reclist = self.model.Recommend(user = uid, n = self.N_recommendations, default_user=default_user) # recommend N_recommendations items to uid
                results[metric].append(self._EvalPrequential__EvalPoint(iid, reclist)) # evaluate recommendations before updating - Recall@20 | careful with name mangling

            # store interaction in the holdout for this checkpoint with prob 0.1 if interaction is not yet in the holdout for this checkpoint
            if np.random.uniform(0, 1) >= 0.9 and ( [uid, iid] not in self.holdouts[checkpoint_count[0]] ):
                self._StoreInteraction(uid, iid, n_checkpoint=checkpoint_count[0]) 
            # else, use interaction for training
            else:
                self.model.IncrTrain(uid, iid) # perform incremental training

            if (checkpoint_count[1] >= checkpoint_size): # if the number of interactions exceedes the size of each checkpoint interval
                if (checkpoint_count[0] == self.n_holdouts-1): # if its the last checkpoint
                    if (i == count-1): # and if its the last item in the entire stream (to avoid leaving items unseen)
                        checkpoint_count[1] = 0
                        self._MakeCheckpoint() # store model
                        checkpoint_count[0] += 1
                    else:
                        pass
                else:
                    checkpoint_count[1] = 0
                    self._MakeCheckpoint() # store model
                    checkpoint_count[0] += 1

        self._ConvertHoldouts()

        return results
    
    def EvaluateHoldouts(self, exclude_known_items:bool=True, default_user:str='none'):
        '''
        exclude_known_items -- boolean, exclude known items from recommendation
        default_user -- str. One of: random, average, or median. If user is not present in model (new user) user factors are generated.
        '''
        self.results_matrix = np.zeros(shape=(self.n_holdouts, self.n_holdouts))
        metric = self.metrics[0]
        # results_matrix = np.zeros(shape=(eval.n_holdouts, eval.n_holdouts))
        for i, hd in enumerate( self.holdouts ):
            for j, model in enumerate( self.model_checkpoints ):
                eh_instance = EvalHoldout(model=model, holdout=hd, metrics=[metric], N_recommendations=self.N_recommendations, default_user=default_user)
                result = sum( eh_instance.Evaluate(exclude_known_items=exclude_known_items)[metric]) / hd.size
                self.results_matrix[i, j] = result
    
    def _MakeCheckpoint(self):
        model_cp = copy.deepcopy(self.model)
        self.model_checkpoints.append(model_cp) # [n_checkpoint]

    def _StoreInteraction(self, uid, iid, n_checkpoint):
        self.holdouts[n_checkpoint].append([uid, iid])
        # self.holdouts[n_checkpoint].AddFeedback(user=uid, item=iid)

    def _ConvertHoldouts(self):
        for h in range(self.n_holdouts):
            users = [ i[0] for i in self.holdouts[h] ]
            items = [ i[1] for i in self.holdouts[h] ]
            holdout = ImplicitData(user_list=users, item_list=items)
            self.holdouts[h] = holdout
            