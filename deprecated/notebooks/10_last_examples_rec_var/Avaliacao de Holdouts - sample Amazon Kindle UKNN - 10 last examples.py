import os
import sys
sys.path.append(os.path.abspath('') + '/../..')

from data import ImplicitData
from plot_utils import recall_heatmap
from dataset_evaluation_utils import *
from recommenders_implicit import UserKNN  # ISGD framework, BISGD,
from eval_implicit import EvaluateHoldouts # EvaluateAndStore para guardar estados do modelo e holdouts, a avaliação prequencial de ratings implicitos é opcional, , EvalHoldout

import pandas as pd 
import numpy as np 
import seaborn as sns
sns.set_style('whitegrid')

def avg_recall(results_matrix): # Lopez-Paz e Ranzato GEM 2017
    return np.mean( np.diag(results_matrix) )

def compute_BWT(results_matrix): # Lopez-Paz e Ranzato GEM 2017
    BWT = []
    n_checkpoints = results_matrix.shape[0]
    for T in range(1, n_checkpoints): # 1 means holdout 2, 2 means 3, so on
        Rti = results_matrix.iloc[T, 0:T] # get models performances' on previous holdouts
        Rii = np.diag(results_matrix)[0:T] # get models performances' on their closest holdouts (diagonal)
        E = sum( Rti - Rii ) # future models performances' - performances' of models closest to holdouts (diagonal)
        BWT.append( E/T ) # store average BWT for model
    return BWT, np.mean( BWT ) # return BWT and average BWT for all models

def compute_FWT(results_matrix): # Díaz-Rodriguez et al. 2018
    upper_tri = results_matrix.to_numpy()[np.triu_indices(results_matrix.shape[0], k=1)]
    return np.mean(upper_tri)

# importa dataset 'movieles'
data = pd.read_csv('../output/amazonkindle_dump/2nd_sampled_amazon_kindle.csv')
user_col = 'user_id'
item_col = 'item_id'


data['date'] = data['date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))

# CODE TO GET LAST N INTERACTIONS FROM EACH USER AS HOLDOUT
# IF USER DID NOT INTERACT WITH AT LEAST N+1 ITEMS, THEN IT IS NOT USED FOR HOLDOUT

N = 10
cold_start_buckets = 0
#     print('0',data.shape[0]) # debug
print('Creating buckets. . .')
buckets = []

# create buckets based on months
months = data['date'].unique()
months.sort()
for interval in months:
    idx = (data['date'] == interval)
    buckets.append( data[idx] )

print('Creating holdouts. . .')
# create holdouts with last user interaction
holdouts = []

for i, b in enumerate( buckets ):
    if i >= cold_start_buckets:
        condition = (b[user_col].value_counts() > N)
        frequent_users = b[user_col].value_counts()[ condition ].index
        holdout_idx = []
        for u in frequent_users:
            tail_idx = list( b[b[user_col] == u].tail(N).index )
            holdout_idx += tail_idx
        holdout = b.loc[holdout_idx].reset_index(drop=True)
        holdouts.append(holdout)
        # buckets[i] = b.drop(index=holdout_idx).reset_index(drop=True)
        buckets[i] = b.reset_index(drop=True)

print('Converting to ImplicitData. . .')
for i, b in enumerate(buckets):
    buckets[i] = ImplicitData(user_list=b[user_col], item_list=b[item_col]) # convert to ImplicitData

for j, h in enumerate(holdouts):
    holdouts[j] = ImplicitData(user_list=h[user_col], item_list=h[item_col]) # convert to ImplicitData

print('Done!')
# return buckets, holdouts


# transforma interações em objeto que contem mappings usuário-itens e item-usuários, contém também métodos de suporte. recebe listas
# define hyperparameters
K = 10
similarity = 'cosine'
# O modelo deve ser iniciado com uma lista vazia
empty_stream = ImplicitData([], [])
# Se o stream for passado, ao excluir itens conhecidos o recall é sempre 0. Ao permitir a recomendação de itens já vistos, o recall não é 0.
model = UserKNN(empty_stream, k=K, similarity=similarity)

# criamos instancia de EvaluateHoldouts para treinar o modelo e criar checkpoints
eval = EvaluateHoldouts(model=model, buckets=buckets, holdouts=holdouts)

# 28min 20s
eval.Train_Evaluate(N_recommendations=20, exclude_known_items=False, default_user='none')

rm = eval.results_matrix
df_exp7 = pd.DataFrame(rm)

arecall = avg_recall(df_exp7)
BWT, meanBWT = compute_BWT(df_exp7)
FWT = compute_FWT(df_exp7)
# que itens que usuario utilizou no passado e deixou de consumir o sistema ainda pode recomendar

print('average recall:', arecall)
print('BWT:', BWT, 'meanBWT:', meanBWT)
print('FWT:', BWT)

allresults = [df_exp7]
allresults[0].to_csv('../output_10_last_examples/Kindle_UKNN_result_exp_7')

