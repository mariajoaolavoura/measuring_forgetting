from data import ImplicitData
import numpy as np
from .Model import Model
import time

class ISGD(Model):
    """
    Incremental SGD-based matrix factorization algorithm for implicit feedback:
    Vinagre, J., Jorge, A. M., & Gama, J. (2014, July). Fast incremental matrix factorization for recommendation with positive-only feedback. In International Conference on User Modeling, Adaptation, and Personalization (pp. 459-470). Springer, Cham.
    https://link.springer.com/chapter/10.1007/978-3-319-08786-3_41
    """

    def __init__(self, data: ImplicitData, num_factors: int = 10, num_iterations: int = 10, learn_rate: float = 0.01, u_regularization: float = 0.1, i_regularization: float = 0.1, random_seed: int = 1):
        """
        Constructor.

        Keyword arguments:
        data -- ImplicitData object
        num_factors -- Number of latent features (int, default 10)
        num_iterations -- Maximum number of iterations (int, default 10)
        learn_rate -- Learn rate, aka step size (float, default 0.01)
        regularization -- Regularization factor (float, default 0.01)
        random_seed -- Random seed (int, default 1)
        """
        self.data = data
        self.num_factors = num_factors
        self.num_iterations = num_iterations
        self.learn_rate = learn_rate
        self.user_regularization = u_regularization
        self.item_regularization = i_regularization
        self.random_seed = random_seed
        np.random.seed(random_seed)
        self.train_time_record = {
            'IncrTrain_0':[],
            'IncrTrain_1':[],
            'IncrTrain_2':[],
            '_UpdateFactors_inner':[],
            '_UpdateFactors_users':[], 
            '_UpdateFactors_items':[]            
        }
        self.recommend_time_record = {
            'Recommend_0':[],
            'Recommend_1':[],
            'Recommend_2':[],
            'Recommend_3':[],
            'Recommend_4':[],
            'Recommend_5':[]
            
        }
        self._InitModel()
        

    def _InitModel(self):
        self.ResetModel()

    def ResetModel(self):
        self.user_factors = [np.random.normal(0.0, 0.1, self.num_factors) for _ in range(self.data.maxuserid + 1)]
        self.item_factors = [np.random.normal(0.0, 0.1, self.num_factors) for _ in range(self.data.maxitemid + 1)]

    def BatchTrain(self):
        """
        Trains a new model with the available data.
        """
        idx = list(range(self.data.size))
        for iter in range(self.num_iterations):
            np.random.shuffle(idx)
            for i in idx:
                user_id, item_id = self.data.GetTuple(i, True)
                self._UpdateFactors(user_id, item_id)

    def IncrTrain(self, user, item, update_users: bool = True, update_items: bool = True, n_times: int = 1):
        """
        Incrementally updates the model.

        Keyword arguments:
        user_id -- The ID of the user
        item_id -- The ID of the item
        """
        # IncrTrain_0
        s = time.time()
        user_id, item_id = self.data.AddFeedback(user, item)
        f = time.time()
        self.train_time_record['IncrTrain_0'].append(np.round(f-s,3))
        # IncrTrain_1
        s = time.time()
        if len(self.user_factors) == self.data.maxuserid:
            self.user_factors.append(np.random.normal(0.0, 0.1, self.num_factors))
        if len(self.item_factors) == self.data.maxitemid:
            self.item_factors.append(np.random.normal(0.0, 0.1, self.num_factors))
        f = time.time()
        self.train_time_record['IncrTrain_1'].append(np.round(f-s,3))
        # IncrTrain_2
        s = time.time()
        if update_users or update_items:
            for _ in range(n_times):
                self._UpdateFactors(user_id, item_id, update_users, update_items)
        f = time.time()
        self.train_time_record['IncrTrain_2'].append(np.round(f-s,3))        
        
    def _UpdateFactors(self, user_id, item_id, update_users: bool = True, update_items: bool = True, target: int = 1):
        p_u = self.user_factors[user_id]
        q_i = self.item_factors[item_id]
        for _ in range(int(self.num_iterations)):
            #_UpdateFactors_inner
            s = time.time()
            err = target - np.inner(p_u, q_i)
            f = time.time()
            self.train_time_record['_UpdateFactors_inner'].append(np.round(f-s,3))
            # _UpdateFactors_users
            s = time.time()
            if update_users:
                delta = self.learn_rate * (err * q_i - self.user_regularization * p_u)
                p_u += delta
            f = time.time()
            self.train_time_record['_UpdateFactors_users'].append(np.round(f-s,3))
            # _UpdateFactors_items
            s = time.time()
            if update_items:
                delta = self.learn_rate * (err * p_u - self.item_regularization * q_i)
                q_i += delta
            f = time.time()
            self.train_time_record['_UpdateFactors_items'].append(np.round(f-s,3))
            
        self.user_factors[user_id] = p_u
        self.item_factors[item_id] = q_i

    def Predict(self, user_id, item_id):
        """
        Return the prediction (float) of the user-item interaction score.

        Keyword arguments:
        user_id -- The ID of the user
        item_id -- The ID of the item
        """
        return np.inner(self.user_factors[user_id], self.item_factors[item_id])


    def Recommend(self, user, n: int = -1, exclude_known_items: bool = True, candidates: set = {}, default_user: str = 'none'):
        """
        Returns an list of tuples in the form (item_id, score), ordered by score.

        Keyword arguments:
        user_id -- The ID of the user
        item_id -- The ID of the item
        n -- number of recommendations. default returns all items sorted by score.
        exclude_known_items -- boolean, exclude known items from recommendation
        candidates -- dictionary, ?
        default_user -- str. One of: random, average, or median. If user is not present in model (new user) user factors are generated.
        """
        recs = []
        # Recommend_0
        s = time.time()
        user_id = self.data.GetUserInternalId(user)
        f = time.time()
        self.recommend_time_record['Recommend_0'].append(np.round(f-s,3))

        if user_id == -1:
            if default_user == 'random':
                p_u = np.random.normal(0.0, 0.1, self.num_factors)
            if default_user == 'average':
                p_u = np.mean(self.user_factors, axis=0)
            if default_user == 'median':
                p_u = np.median(self.user_factors, axis=0)
            else: # none
                return []
        else:
            p_u = self.user_factors[user_id]
        # Recommend_1
        s = time.time()
        scores = np.abs(1 - np.inner(p_u, self.item_factors))
        f = time.time()
        self.recommend_time_record['Recommend_1'].append(np.round(f-s,3))        
        # Recommend_2
        s = time.time()
        recs = np.column_stack((self.data.itemset, scores))
        f = time.time()
        self.recommend_time_record['Recommend_2'].append(np.round(f-s,3))
        # Recommend_3
        s = time.time()
        if exclude_known_items and user_id != -1:
            user_items = self.data.GetUserItems(user_id)
            recs = np.delete(recs, user_items, 0)        
        f = time.time()
        self.recommend_time_record['Recommend_3'].append(np.round(f-s,3))
        
        if len(candidates):
            # TODO: testar código
            candidates_internal = self.data.GetItemInternalIds(candidates)
            condition = np.isin(recs[:,0].astype(int), candidates_internal)
            recs = recs[condition]
        
        # Recommend_4
        s = time.time()
        if n == -1 or n > len(recs) :
            n = len(recs)
        else:
            recs = recs[np.argpartition(recs[:,1], n-1)[:n]]
        f = time.time()
        self.recommend_time_record['Recommend_4'].append(np.round(f-s,3))
        
        # Recommend_5
        s = time.time()
        recs = recs[np.argsort(recs[:,1])]
        f = time.time()
        self.recommend_time_record['Recommend_5'].append(np.round(f-s,3))

        return recs
