from data import ImplicitData, SymmetricMatrix
import numpy as np
import itertools as itt
from .Model import Model

class UserKNN(Model):
    """
    TODO doc
    """

    def __init__(self, data: ImplicitData, k: int = 10, similarity: str = "cosine"):
        """
        Constructor.

        Keyword arguments:
        data -- ImplicitData object
        k -- Number of neighbors to use (int, default 10)
        """
        self.data = data
        self.k = k
        self._InitModel()


    def _InitModel(self):
        self.ResetModel()

    def ResetModel(self):
        ''' 
        Creates objects user_freq and user_sim,  which contain square matrices of size ( (#num_entities + 1) * 2 ) x ( (#num_entities + 1) * 2 ) filled with zeros.
        Creates a list of 'maxuserid' 'k' sized arrays filled with -1.
        '''
        self.user_freq = SymmetricMatrix(len(self.data.userset)) # self.data.size
        self.user_sim = SymmetricMatrix(len(self.data.userset)) # self.data.size
        self.user_neighbors = [np.zeros(self.k, dtype=np.float32) - 1 for _ in range(self.data.maxuserid + 1)] # int

    def BatchTrain(self):
        """
        Trains a new model with the all the available data.
        """
        self._BuildFreqMatrix()
        self._ComputeSimilarities()
        self._ComputeNeighborhoods()

    def _BuildFreqMatrix(self):
        #Batch
        for item in range(self.data.maxitem + 1):
            users = self.data.GetItemUsers(item)
            for u, v in itt.combinations(users, 2):
                if u != v:
                    self.user_freq.Increment(u, v)
                self.user_freq.IncrementDiag(u)
                self.user_freq.IncrementDiag(v)

    def _ComputeSimilarities(self):
        # Batch: Iterate through all user pairs (u, v) 
        for u, v in itt.combinations(range(self.data.maxuserid + 1), 2):
            # Exclude u == v
            if u != v:
                f_uv = self.user_freq.Get(u, v)
                f_u = self.user_freq.Get(u, u)
                f_v = self.user_freq.Get(v, v)
                if f_uv: 
                    # Cosine
                    sim = f_uv / np.sqrt(f_u ** 2 * f_v ** 2)
                else:
                    sim = 0
                self.user_sims.Set(u, v, sim)

    def _ComputeNeighborhoods(self):
        # Batch
        for u in range(self.data.maxuserid + 1):
            self.user_neighbors[u] = self._ComputeUserNeighbors(u)

    def _ComputeUserNeighbors(self, u: int):
        # Batch + incremental

        # creates an array of size max_id x 2, column 0 contains users internal IDs, column 1 contains similarities of user u with the users from column 0
        sims = np.column_stack(([x for x in range(self.user_sim.max_id + 1)], self.user_sim.GetRow(u))) 
        sims = sims[np.argsort(-sims[:, 1], kind = 'heapsort')]
        neighbors = sims[:(self.k + 1)]
        if len(neighbors):
            neighbors = np.delete(neighbors, np.where(neighbors[:,0]==u), 0)
        return neighbors[:self.k]


    def IncrTrain(self, user, item):
        """
        Incrementally updates the model.

        Keyword arguments:
        user_id -- The ID of the user
        item_id -- The ID of the item
        """
        u, i = self.data.AddFeedback(user, item) # 
        self._UpdateSimilarities(u, i)
        self._UpdateNeighbors(u)


    def _UpdateSimilarities(self, u: int, i: int):
        item_users = self.data.GetItemUsers(i)
        self.user_freq.IncrementDiag(u)
        for v in item_users:
            if v != u:
                self.user_freq.Increment(u, v)

        f_u = self.user_freq.Get(u, u)
        for v in range(self.data.maxuserid + 1):
            if v != u:
                f_uv = self.user_freq.Get(u, v)
                f_v = self.user_freq.Get(v, v)
                if f_uv: 
                    # Cosine
                    sim = f_uv / np.sqrt(f_u ** 2 * f_v ** 2)
                else:
                    sim = 0
                self.user_sim.Set(u, v, sim)

    def _UpdateNeighbors(self, u: int, complete: bool = True):
        if u == len(self.user_neighbors):
            self.user_neighbors.append(np.zeros(self.k, dtype=np.int) - 1)

        # 1. Update user's own neighbors
        self.user_neighbors[u] = self._ComputeUserNeighbors(u)
        
        neighborhoods = [int(nn) for nn,_ in self.user_neighbors[u]]
        for v in range(self.data.maxuserid + 1):
            # Update u's neighbors' neighbors and users whose neighbors include u 
            #if(u in self.user_neighbors[v] or v in neighborhoods):
            #if(u in self.user_neighbors[v]):
            #    self.user_neighbors[v] = self._ComputeUserNeighbors(v)
            if v in neighborhoods:
                self.user_neighbors[v] = self._ComputeUserNeighbors(v)
        

    def Predict(self, user_id, item_id, internal: bool = True):
        """
        Return the prediction (float) of the user-item interaction score.

        Keyword arguments:
        user_id -- The external ID of the user
        item_id -- The external ID of the item
        """
        score = 0
        norm = 0
        if not internal:
            u = self.data.GetUserInternalId(user_id)
            i = self.data.GetItemInternalId(item_id)
        else:
            u, i = user_id, item_id
        for v, s in self.user_neighbors[u]:
            if np.isin(i, self.data.GetUserItems(int(v))):
                score += s
            norm += s
        if norm:
            return score / norm
        return 0


    def Recommend(self, user, n: int = -1, exclude_known_items: bool = True, default_user: str = 'none', sort_list: bool = True):
        """
        Returns an list of tuples in the form (item_id, score), ordered by score.

        Keyword arguments:
        user -- The external ID of the user
        """

        user_id = self.data.GetUserInternalId(user)

        if user_id == -1:
            return []

        recs = [[i, self.Predict(user, i, False)] for i in self.data.itemset]
        recs = np.array(recs)

        if exclude_known_items and user_id != -1:
            user_items = self.data.GetUserItems(user_id)
            recs = np.delete(recs, user_items, 0)

        if sort_list:
            float_scores = recs[:, 1].astype(float) # needed if data.itemset contains strings, and not int or float
#             recs = recs[np.argsort(-recs[:, 1], kind = 'heapsort')]
            recs = recs[np.argsort(-float_scores, kind = 'heapsort')]

        if n == -1 or n > len(recs) :
            n = len(recs)

        return recs[:n]
