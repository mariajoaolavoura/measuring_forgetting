import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # ver linha 72

class ImplicitData:
    '''
    transforma interações em objeto que contem mappings usuário-itens e item-usuários
    contém também métodos de suporte
    assume ratings implicitos
    '''
    def __init__(self, user_list: list, item_list: list):
        self.userlist = user_list # lista de usuarios
        self.itemlist = item_list # lista de itens
        self.size = len(self.userlist) # tamanho da lista de usuarios (total de interações)
        self.userset, self.userindices = np.unique(self.userlist, return_inverse=True) # lista de usuarios unicos, e indices que mapeiam interações aos usuários unicos
        self.itemset, self.itemindices = np.unique(self.itemlist, return_inverse=True) # lista de itens unicos, e indices que mapeiam interações aos itens unicos
        self.maxuserid = len(self.userset) - 1 # ID máximo de usuários
        self.maxitemid = len(self.itemset) - 1 # ID máximo de itens
        self.BuildMaps()

    def BuildMaps(self):
        '''
        cria listas para mapear usuário-itens e itens-usuário
        '''
        # cada lista contem listas vazias para cada usuário e item
        # cada lista em useritems representa um usuário, e é preenchida com os indices dos itens unicos que este usuario interagiu 
        # cada lista em itemusers representa um item, e é preenchida com os indices dos usuarios unicos que interagiram com o item
        self.useritems = []
        self.itemusers = []
        for u in range(self.maxuserid + 1):
            self.useritems.append([])
        for i in range(self.maxitemid + 1):
            self.itemusers.append([])
        for r in range(self.size):
            self.useritems[self.userindices[r]].append(self.itemindices[r])
            self.itemusers[self.itemindices[r]].append(self.userindices[r])
        # há uma lista para cada usuário que contém os itens com que ele interagiu
        # há uma lista para cada item que contém os usuários que interagiram com ele

    def GetUserItems(self, user_id, internal = True):
        '''
        Obtem lista de itens com que o usuário interagiu
        '''
        if internal:
            if user_id > -1 and user_id <= self.maxuserid:
                return self.useritems[user_id]
            return []
        uid = self.GetUserInternalId(user_id)
        if uid > -1:
            return self.itemset[self.useritems[uid]]
        return []

    def GetItemUsers(self, item_id, internal = True):
        '''
        Obtem lista de usuários que interagiram com o item
        '''
        if internal:
            if item_id > -1 and item_id <= self.maxitemid:
                return self.itemusers[item_id]
        iid = self.GetItemInternalId(item_id)
        if iid > -1:
            return self.userset[self.itemusers[iid]]
        return []


    def AddFeedback(self, user, item):
        '''
        Adiciona uma nova interação usuário-item às listas que mapeiam usuário-itens e itens-usuário
        '''
        self.size = self.size + 1
        self.userlist.append(user) # 
        self.itemlist.append(item)
        # if user not in self.userset:
        if not np.isin(user, self.userset, True):
            # if FutureWarning error happens, its bc of comparison between string and numpy arrays - which happens in Lastfm data since users and items IDs are strings
            # https://stackoverflow.com/questions/40659212/futurewarning-elementwise-comparison-failed-returning-scalar-but-in-the-futur
            # FutureWarning: elementwise comparison failed; returning scalar instead, but in the future will perform elementwise comparison
            self.userset = np.append(self.userset, user)
            self.maxuserid = self.maxuserid + 1
            self.useritems.append([])
        user_id = self.GetUserInternalId(user)
        self.userindices = np.append(self.userindices, user_id)
        # if item not in self.itemset:
        if not np.isin(item, self.itemset, True):
            self.itemset = np.append(self.itemset, item)
            self.maxitemid = self.maxitemid + 1
            self.itemusers.append([])
        item_id = self.GetItemInternalId(item)
        self.itemindices = np.append(self.itemindices, item_id)
        self.useritems[user_id].append(item_id)
        self.itemusers[item_id].append(user_id)
        return user_id, item_id

    def GetTuple(self, idx: int, internal: bool = False):
        '''
        Obtem uma tupla do usuário e item em uma interação (idx)
        Pode ser a representação interna ou externa
        '''
        if internal:
            return self.userindices[idx], self.itemindices[idx]
        return self.userlist[idx], self.itemlist[idx]

    def GetUserInternalId(self, user):
        '''
        Obtem ID interno do usuário
        '''
        user_id, = np.where(self.userset == user)
        if len(user_id):
            return user_id[0]
        return -1

    def GetItemInternalId(self, item):
        '''
        Obtem ID interno do item
        '''
        item_id, = np.where(self.itemset == item)
        if len(item_id):
            return item_id[0]
        return -1

    def GetItemInternalIds(self, items:set):
        item_ids, = np.where(np.isin(self.itemset, items))
        if len(item_ids):
            return item_ids
        return []

    def GetUserExternalId(self, user_id:int):
        '''
        Obtem ID externo do usuário (como descrito nos dados)
        '''
        if user_id > -1 and user_id <= self.maxuserid:
            return self.userset[user_id]
        return ""

    def GetItemExternalId(self, item_id:int):
        '''
        Obtem ID externo do item (como descrito nos dados)
        '''
        if item_id > -1 and item_id <= self.maxitemid:
            return self.itemset[item_id]
        return ""
