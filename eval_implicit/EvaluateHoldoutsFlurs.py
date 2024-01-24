# https://flurs.readthedocs.io/en/latest/reference.html#evaluation-utilities
from flurs.data.entity import User, Item, Event
import pandas as pd
import numpy as np
import time

class EvaluateHoldoutsFlurs():
    def __init__(self, model, buckets, holdouts, data, user_col, item_col):
        self.model = model
        self.buckets = buckets
        self.holdouts = holdouts
        self.usermap = pd.Series(pd.unique( data[user_col] )).reset_index().set_index(0).to_dict()['index']
        self.itemmap = pd.Series(pd.unique( data[item_col] )).reset_index().set_index(0).to_dict()['index']
        self.max_users = data[user_col].nunique()
        self.max_items = data[item_col].nunique()
        self.metrics = ["Recall@N"]
        self.IncrementalTraining_time_record = {}
        self.EvaluateHoldouts_time_record = {}

    def Train_Evaluate(self, N_recommendations=20, exclude_known_items:bool=True):
        '''
        Incremental training of recommendation model.
        '''
        # model_checkpoints = []
        self.results_matrix = np.zeros( shape=( len( self.holdouts ), len( self.holdouts ) ) )
        # metric = self.metrics[0]
        self.N_recommendations = N_recommendations
        self.exclude_known_items = exclude_known_items
        # Initialize model
        # model = UserKNNRecommender(k=5) # model
        self.model.initialize()
        print('register users')
        s = time.time()
        for u in range(0, self.max_users):
            if not ((u*100)/self.max_users)%5:
                print (((u*100)/self.max_users),'%')
            self.model.register(User(u))
        f = time.time()
        print( f-s, 'register users time' )
        print('register items')
        s = time.time()
        for i in range(0, self.max_items):
            if not ((i*100)/self.max_items)%5:
                print (((i*100)/self.max_items),'%')
            self.model.register(Item(i))
        f = time.time()
        print( f-s, 'register items time' )
        self.max_item_ID = 0
        self.cold_start_buckets = len( self.buckets ) - len( self.holdouts ) 
        # for each bucket, perform incremental training, evaluate model state against all holdouts
        for b, bucket in enumerate(self.buckets):
            print(100*'-')
            print(f'Training: bucket {b}')
            incrtrain_time = []
            for i in range(bucket.size):
                uid,iid = bucket.GetTuple(i)
                s = time.time()
                u_flurs, i_flurs = self.usermap[uid], self.itemmap[iid]
                self.max_item_ID = max(self.max_item_ID, i_flurs)
                # For each interaction - register user, register item, register event
                user = User(u_flurs)
                item = Item(i_flurs)
                event = Event(user, item)
                self.model.update(event)
                f = time.time()
                incrtrain_time.append(f-s)        

            self.IncrementalTraining_time_record[f'bucket_{b}'] = {
                'size':bucket.size,
                'train time vector':incrtrain_time,
                'avg train time':np.mean(incrtrain_time),
                'total train time':np.sum(incrtrain_time)
                }            

            if b >= self.cold_start_buckets:
                self._evaluate(n_bucket=b)

    def _evaluate(self, n_bucket):
        # after learning a bucket, evaluate model over every holdout     
        evaluate_time = [] 
        time_get_tuple = []
        time_recommend = []
        time_del_user_items = []
        time_eval_point = []  
        for i, hd in enumerate(self.holdouts):
            print(f'Evaluating: model {n_bucket-self.cold_start_buckets} x holdout {i}')
            start_evaluate_time = time.time()
            results = {}
            # for metric in self.metrics:
            metric = self.metrics[0]
            results[metric] = []
            for hd_i in range(hd.size):
                # GetTuple
                s = time.time()
                uid, iid = hd.GetTuple(hd_i) # get external IDs
                u_flurs, i_flurs = self.usermap[uid], self.itemmap[iid]
                user = User(u_flurs)
                f = time.time()
                time_get_tuple.append(f - s)
                # recommend
                s = time.time()
                reclist = self.model.recommend(user, np.arange(self.max_item_ID+1) )[0]
                f = time.time()
                time_recommend.append(f - s)
                # get items seen by user up until bucket 'n_bucket'
                # delete seen items 
                s = time.time()
                user_items = self.buckets[n_bucket].GetUserItems(uid, internal=False) 
                for previous_b in range(n_bucket):
                    user_items = np.concatenate((user_items, self.buckets[previous_b].GetUserItems(uid, internal=False)))
                user_items = [self.itemmap[i] for i in user_items]
                if self.exclude_known_items:
                    reclist = np.delete(reclist, user_items)
                f = time.time()
                time_del_user_items.append(f - s)
                # get n recommendations
                n = self.N_recommendations
                if n == -1:
                    n = len(reclist)
                reclist = reclist[:n]
                if len(reclist):
                    s = time.time()
                    results[metric].append(self.__EvalPoint(i_flurs, reclist))
                    f = time.time()
                    time_eval_point.append(f - s)
                else:
                    print(uid, 'user not seen')
            result = results[metric]
            n_not_seen = hd.size - len(result) # if user was not seen, its not added to recall. May be needed to store difference.
            if n_not_seen:
                print(f'recommendations not made for users in holdout {i} x bucket {n_bucket}: {n_not_seen}')
            result = sum( result ) / len(result)
            self.results_matrix[n_bucket-self.cold_start_buckets, i] = result
            end_evaluate_time = time.time()
            evaluate_time.append(end_evaluate_time - start_evaluate_time)
            self.EvaluateHoldouts_time_record[f'model {n_bucket-self.cold_start_buckets} x holdout {i}'] = {
                'size': hd.size,
                'train time vector':evaluate_time,
                'avg model eval time':np.mean(evaluate_time),
                'total train time':np.sum(evaluate_time),
                'time_get_tuple': time_get_tuple,
                'time_recommend': time_recommend,
                'time_del_user_items': time_del_user_items,
                'time_eval_point': time_eval_point,
            }

    def __EvalPoint(self, item_id, reclist):
        result = 0
        if len(reclist) == 0:
            return 0
        for metric in self.metrics:
            if metric == "Recall@N":
                result = int(item_id in reclist)
        return result