import random
from data import ImplicitData
from .ISGD import ISGD
import numpy as np
import time

class RSISGD(ISGD):
    """
    ISGD with random negative sampling
    """
    def __init__(self, data: ImplicitData, num_factors: int = 10, num_iterations: int = 10, learn_rate: float = 0.01, u_regularization: float = 0.1, i_regularization: float = 0.1, random_seed: int = 1, ra_length: int = 1):
        super().__init__(data, num_factors, num_iterations, learn_rate, u_regularization, i_regularization, random_seed)
        self.ra_length = ra_length

    def IncrTrain(self, user, item, update_users: bool = True, update_items: bool = True):
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
        user_items = self.data.GetUserItems(user_id)
        if len(user_items) < len(self.data.itemset) - self.ra_length:
            for _ in range(self.ra_length):
                negative_item_id = random.choice(range(len(self.data.itemset)))
                while negative_item_id in user_items:
                    negative_item_id = random.choice(range(len(self.data.itemset)))

                self._UpdateFactors(user_id, negative_item_id, True, False, 0)
        
        self._UpdateFactors(user_id, item_id)
        f = time.time()
        self.train_time_record['IncrTrain_2'].append(np.round(f-s,3))     