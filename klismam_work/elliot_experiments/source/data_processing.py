import os

import pandas as pd 

def process_sample(data_path:str, path_to_datasets:str, user_col:str, item_col:str, date_conversion_function, test_run:bool=True):
    data = pd.read_csv(data_path)
    # CODE TO GET LAST N INTERACTIONS FROM EACH USER AS HOLDOUT
    # IF USER DID NOT INTERACT WITH AT LEAST N+1 ITEMS, THEN IT IS NOT USED FOR HOLDOUT
    data['date'] = data['date'].apply(date_conversion_function)
    N = 10
    cold_start_buckets = 0
    #     print('0',data.shape[0]) # debug
    print('Creating buckets. . .')
    buckets = []
    # assert interval_type in ['W', 'M', 'QS', 'F'], "interval must be one of W, M, QS, or F"
    # create buckets based on months
    months = data['date'].unique()
    months.sort()
    for interval in months: # type: ignore
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
    
    if test_run:
        buckets = [b.iloc[:1000] for b in buckets]
        holdouts = [h.iloc[:500] for h in holdouts]
    else:
        buckets = [b for b in buckets]
        holdouts = [h for h in holdouts]

    print('shape of buckets', [b.shape for b in buckets])
    print('shape of holdouts', [h.shape for h in holdouts])

    if not os.path.isdir(path_to_datasets):
        os.mkdir(path_to_datasets)

    for bi in range(len(buckets)):
        print(f'Size of training bucket {bi}', pd.concat(buckets[:bi+1]).shape)
        pd.concat(buckets[:bi+1]).to_csv(f'{path_to_datasets}/movielens_b{bi}.csv', sep='\t', columns=[user_col, item_col], header=False, index=False)
        holdouts[bi].to_csv(f'{path_to_datasets}/movielens_h{bi}.csv', sep='\t', columns=[user_col, item_col], header=False, index=False)