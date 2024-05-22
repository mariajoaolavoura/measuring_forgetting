from data import ImplicitData
import numpy as np
from .Model import Model

class BPRMF(Model):
    """
    BPR: Bayesian Personalized Ranking from Implicit Feedback
    Rendle, S., Freudenthaler, C., Gantner, Z., & Schmidt-Thieme, L. (2009, June). BPR: Bayesian personalized ranking from implicit feedback. In Proceedings of the Twenty-Fifth Conference on Uncertainty in Artificial Intelligence (pp. 452-461).
    https://dl.acm.org/doi/abs/10.5555/1795114.1795167
    """

    def __init__(self, data: ImplicitData, num_factors: int = 10, num_iterations: int = 10, learn_rate: float = 0.01, u_regularization: float = 0.1, i_regularization: float = 0.1, j_regularization: float = 0.1, update_negative_items: bool = False, random_seed: int = 1):
        """
        Constructor.

        Keyword arguments:
        data -- ImplicitData object
        num_factors -- Number of latent features (int, default 10)
        num_iterations -- Maximum number of iterations (int, default 10)
        learn_rate -- Learn rate, aka step size (float, default 0.01)
        u_regularization -- User regularization factor (float, default 0.01)
        i_regularization -- Item regularization factor (float, default 0.01)
        j_regularization -- Negative item regularization factor (float, default 0.01)
        update_negative_items -- Whether to update the factors of negative items (bool, default False)
        random_seed -- Random seed (int, default 1)
        """
        self.data = data
        self.num_factors = num_factors
        self.num_iterations = num_iterations
        self.learn_rate = learn_rate
        self.user_regularization = u_regularization
        self.item_regularization = i_regularization
        self.negative_item_regularization = j_regularization
        self.update_negative_items = update_negative_items
        self.random_seed = random_seed
        np.random.seed(random_seed)
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

        user_id, item_id = self.data.AddFeedback(user, item)

        if len(self.user_factors) == self.data.maxuserid:
            self.user_factors.append(np.random.normal(0.0, 0.1, self.num_factors))
        if len(self.item_factors) == self.data.maxitemid:
            self.item_factors.append(np.random.normal(0.0, 0.1, self.num_factors))

        # There has to be at least one unobserved item
        user_items = self.data.GetUserItems(user_id)
        unobserved = np.setdiff1d(np.arange(self.data.maxitemid + 1), user_items)
        
        if len(unobserved) > 0 and (update_users or update_items):
            # Sample unobserved item
            negative_item_id = int(np.random.choice(unobserved))
            for _ in range(n_times):
                self._UpdateFactors(user_id, item_id, negative_item_id, update_users, update_items, self.update_negative_items)

    def _UpdateFactors(self, user_id, item_id, negative_item_id, update_users: bool = True, update_items: bool = True, update_negative_items: bool = False, target: int = 1):
        p_u = self.user_factors[user_id]
        q_i = self.item_factors[item_id]
        q_j = self.item_factors[negative_item_id]

        for _ in range(int(self.num_iterations)):
            x_ui = np.inner(p_u, q_i)
            x_uj = np.inner(p_u, q_j)
            x_uij = x_ui - x_uj
            sigmoid = 1 / (1 + np.e ** x_uij)

            if update_users:
                delta = self.learn_rate * (sigmoid * (q_i - q_j) - self.user_regularization * p_u)
                p_u += delta

            if update_items:
                delta = self.learn_rate * (sigmoid * p_u - self.item_regularization * q_i)
                q_i += delta

            if update_negative_items:
                delta = self.learn_rate * (sigmoid * -p_u - self.item_regularization * q_j)
                q_j += delta

        self.user_factors[user_id] = p_u
        self.item_factors[item_id] = q_i
        self.item_factors[negative_item_id] = q_j

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
        """

        recs = []

        user_id = self.data.GetUserInternalId(user)

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
        
        scores = np.inner(p_u, self.item_factors)

        recs = np.column_stack((self.data.itemset, scores))

        if exclude_known_items and user_id != -1:
            user_items = self.data.GetUserItems(user_id)
            recs = np.delete(recs, user_items, 0)

        if len(candidates):
            condition = np.isin(recs[:,0], candidates)
            recs = recs[condition]

        if n == -1 or n > len(recs) :
            n = len(recs)
        else:
            recs = recs[np.argpartition(-recs[:,1], n-1)[:n]]

        recs = recs[np.argsort(-recs[:,1])]

        return recs
