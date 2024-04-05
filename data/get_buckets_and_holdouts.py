from .implicit_data import ImplicitData

import pandas as pd
import numpy as np

def getBucketsHoldouts(data:pd.DataFrame, user_col:str, item_col:str, frequent_users:list, interval_type:str=None, intervals:list=None, cold_start_buckets:int=1):
    '''
    Creates lists with buckets and holdouts based on passed intervals.
    
    data - interactions, must contain 'date' column\n
    user_col - name of column with user IDs\n
    item_col - name of column with item IDs\n
    frequent_users - list of frequent users. Only their interactions go to holdout.\n
    interval_type - W for week, M for month, QS for quarter or semester, F representing fixed bucket size\n
    intervals - list containing tuple intervals. pos0-interval start, pos1-interval end. for QS these are dates, for F these are indexes. not necessary for Month interval type.\n
    cold_start_buckets - number of buckets to be used for training only\n
    '''
#     print('0',data.shape[0]) # debug
    print('Creating buckets. . .')
    buckets = []
    assert interval_type in ['W', 'M', 'QS', 'F'], "interval must be one of W, M, QS, or F"
    if interval_type == 'W':
        # create buckets based on months
        weeks = data['week'].unique()
        for interval in weeks:
            idx = (data['week'] == interval)
            buckets.append( data[idx] )
    elif interval_type == 'M':
        # create buckets based on months
        months = data['date'].unique()
        months.sort()
        for interval in months:
            idx = (data['date'] == interval)
            buckets.append( data[idx] )
    elif interval_type == 'QS':
        # create buckets based on quarters or semesters
        for s, e in intervals:
            idx = (data['date'] >= s) & (data['date'] <= e)
            buckets.append( data[idx] )

            # debug
            # print(str(s)+' to '+str(e)+'\n'+\
            #       str(data.loc[idx, 'date'].min()) + ' to ' +str(data.loc[idx, 'date'].max())+\
            #       '\nn interactions: '+str(data[idx].shape[0]))
            
        # i dont understand the reason why the following else-case is here (is not even paired with an if-case) 
        # and is creating an empty bucket...
        # else:
        #     idx = (data['date'] > e)
        #     buckets.append( data[idx] )
        # replaced with the following if-case code
        idx = (data['date'] > e)
        if data[idx].shape[0] > 0:
            buckets.append( data[idx] )
            
    else:
        # create buckets based on fixed number of examples
        for i, j in intervals:
            buckets.append( data.iloc[i:j] )
    
#     # debug
#     a = pd.concat( buckets ).set_index([user_col, item_col])
#     print('1',a.shape[0])
    
    print('Creating holdouts. . .')
    # create holdouts with last user interaction
    holdouts = []
    frequent_users_seen = [] # frequent users must have been seen at least once before being sent to holdouts. 
    # Imagine if the first frequent user interaction is the single interaction by this user in an interval, then this single interaction cant be sent to the holdout.
    for i, b in enumerate( buckets ):
        
        # debug
        # print('bucket '+str(i)+' - '+str(b['date'].min())+' to '+str(b['date'].max())+'\nstarting n interactions: '+str(len(buckets[i])))

        if i >= cold_start_buckets:
            last_interaction_idx = []
            for u in frequent_users:
                idx = b[user_col] == u
                if (idx.sum() == 1) and (u not in frequent_users_seen): # first condition to see if user appears once, second to see if user was not seen before - then it wont go to holdout, and it will be marked as seen
                    frequent_users_seen.append(u)
                    continue
                elif idx.sum() > 0: # else, if user appears at least once, append index to holdout
                    last_interaction_idx.append( b[ idx ].index[-1] )
                    if (u not in frequent_users_seen): # and if user hasnt been seen, mark as seen (he must appear at least twice then)
                        frequent_users_seen.append(u)
            holdout = b.loc[ last_interaction_idx ] # get last interactions as holdout
            holdout.reset_index(drop=True, inplace=True) # reset index required - implicitdata indexes user by their previous index
            holdouts.append(holdout) # append to holdouts
            buckets[i] = b.drop( index = last_interaction_idx).reset_index(drop=True) # remove last interactions from bucket
#             # debug
#             a = pd.concat( buckets ).set_index([user_col, item_col])
#             b = pd.concat( holdouts )[[user_col, item_col]].set_index([user_col, item_col])
#             print('2', a.reset_index().shape[0] + b.reset_index().shape[0] )
        else: # if bucket belongs to 'cold_start_buckets'
            # debug
            print('cold start bucket')
            
            buckets[i] = b.reset_index(drop=True)
            for u in frequent_users: # as before, we mark frequent users in the cold start bucket as seen
                idx = b[user_col] == u
                if (idx.sum() > 0):
                    frequent_users_seen.append(u)
        # debug
        # print('ending n interactions: '+str(len(buckets[i])))
    
    print('Cleaning holdouts. . .')
    # a verification is required to remove any items in the holdouts from the buckets
    # i.e. items that are in holdouts can never be used for training
    # can this be done while the holdouts and buckets are created?  

    # for each holdout:
    #   set interaction tuple as index 
    #   perform a inner join with the buckets dataframe.
    #   get the unique interactions that occur in both (i.e. resulting unique indexes)
    #   append these interactions to the respective bucket
    #   remove these interactions from the holdout

    for i, _ in enumerate(holdouts): 
        buckets_df = pd.concat( buckets )[[user_col, item_col]].set_index([user_col, item_col]) # concatenate all holdouts and set interaction tuple as index.
        temp_h = holdouts[i].set_index([user_col, item_col])
        common_interactions = temp_h.join(buckets_df, how='inner').index
        common_interactions = np.unique( common_interactions )
        print(f'common interactions between holdout {i+1} and all buckets: {len(common_interactions)}')
        bucket = buckets[i+cold_start_buckets].append(temp_h.loc[ common_interactions ].reset_index()).sort_values(by='timestamp').reset_index(drop=True)
        buckets[i+cold_start_buckets] = bucket
        holdouts[i] = temp_h.drop(index=common_interactions).reset_index()
                
#     for i, _ in enumerate(buckets): 
#         holdouts_df = pd.concat( holdouts )[[user_col, item_col]].set_index([user_col, item_col]) # concatenate all holdouts and set interaction tuple as index.
#         temp_b = buckets[i].set_index([user_col, item_col])
#         common_interactions = temp_b.join(holdouts_df, how='inner').index
#         common_interactions = np.unique( common_interactions )
#         print(f'common interactions between bucket {i+1} and all holdouts.')
#         print( len(common_interactions) )
#         # if bucket[i] is not a cold start bucket, the interactions are removed from the bucket to the holdout
#         if i >= cold_start_buckets:
#             holdout = holdouts[i-cold_start_buckets].append(temp_b.loc[ common_interactions ].reset_index()).sort_values(by='timestamp').reset_index(drop=True)
#             holdouts[i-cold_start_buckets] = holdout
#             buckets[i] = temp_b.drop(index=common_interactions).reset_index()
# #             # debug
# #             a = pd.concat( buckets ).set_index([user_col, item_col])
# #             b = pd.concat( holdouts )[[user_col, item_col]].set_index([user_col, item_col])
# #             print('3', a.reset_index().shape[0] + b.reset_index().shape[0] )
#         # if bucket[i] is a cold start bucket, the interactions are removed from the holdouts instead
#         else:
#             for j, _ in enumerate(holdouts):
#                 bucket = buckets[j+1].set_index([user_col, item_col])
#                 holdout = holdouts[j].set_index([user_col, item_col])
#                 ci_temp = []
#                 for ci in common_interactions:
#                     try:
#                         bucket = bucket.append(holdout.loc[ ci ])
#                         ci_temp.append(ci)
#                     except:
#                         continue
#                 buckets[j+1] = bucket.reset_index().sort_values(by='timestamp')
#                 holdouts[j] = holdout.drop(index=ci_temp).reset_index()
# #             # debug
# #             a = pd.concat( buckets ).set_index([user_col, item_col])
# #             b = pd.concat( holdouts )[[user_col, item_col]].set_index([user_col, item_col])
# #             print('4', a.reset_index().shape[0] + b.reset_index().shape[0] )
                
    print('Converting to ImplicitData. . .')
    for i, b in enumerate(buckets):
        buckets[i] = ImplicitData(user_list=b[user_col], item_list=b[item_col]) # convert to ImplicitData

    for j, h in enumerate(holdouts):
        holdouts[j] = ImplicitData(user_list=h[user_col], item_list=h[item_col]) # convert to ImplicitData
    
    print('Done!')
    return buckets, holdouts

        # holdouts[i-cold_start_buckets] = ImplicitData(user_list=holdout[user_col], item_list=holdout[item_col]) # convert holdout to ImplicitData
        # buckets[i] = ImplicitData(user_list=bucket[user_col], item_list=bucket[item_col])


    # # Problem - some users interact with the same items over and over
    # # Solution - remove their interactions from the holdouts
    # b_users = pd.concat(buckets)[user_col].unique()
    # h_users = pd.concat(holdouts)[user_col].unique()
    # absent_idx = [u not in b_users for u in h_users] # users in holdouts that are not in buckets - they would never be *seen* by the model
    # absent_users = h_users[absent_idx]
    # for i, b in enumerate(buckets):
    #     temp_b = b.copy()
    #     print(i)
    #     if i >= cold_start_buckets:
    #         holdout = holdouts[i-cold_start_buckets]#.set_index([user_col])
    #         for au in absent_users:
    #             idx = holdout[ holdout[user_col] == au ].index
    #             temp_b = temp_b.append( holdout.loc[idx] )
    #             holdout = holdout.drop(index=idx).reset_index(drop=True)
    #         holdouts[i-cold_start_buckets] = holdout
    #             # temp_b = temp_b.append( interactions )
    #             # try:

    #             #     interactions = holdout.loc[au].reset_index()
    #             #     holdout = holdout.drop(index=au).reset_index()
    #             #     holdouts[i-cold_start_buckets] = holdout
    #             #     temp_b = temp_b.append( interactions )
    #             # except:
    #             #     continue
    #     else:            
    #         holdout = cs_holdout[i]#.set_index([user_col])
    #         for au in absent_users:
    #             idx = holdout[ holdout[user_col] == au ].index
    #             temp_b = temp_b.append( holdout.loc[idx] )
    #             holdout = holdout.drop(index=idx).reset_index(drop=True)
    #         cs_holdout[i] = holdout
    #             # temp_b = temp_b.append( interactions )
    #             # try:
    #             #     interactions = holdout.loc[au].reset_index()
    #             #     holdout = holdout.drop(index=au).reset_index()
    #             #     cs_holdout[i] = holdout
    #             #     temp_b = temp_b.append( interactions )
    #             # except:
    #             #     continue
    #         # interactions = holdout.loc[absent_users].reset_index()
    #     buckets[i] = temp_b.sort_values(by='timestamp').reset_index(drop=True) 
    
def getBucketsHoldouts_lastNinteractions(data:pd.DataFrame, user_col:str, item_col:str, frequent_users:list, interval_type:str=None, intervals:list=None, cold_start_buckets:int=1):
    '''
    Creates lists with buckets and holdouts based on passed intervals.
    
    data - interactions, must contain 'date' column\n
    user_col - name of column with user IDs\n
    item_col - name of column with item IDs\n
    frequent_users - list of frequent users. Only their interactions go to holdout.\n
    interval_type - W for week, M for month, QS for quarter or semester, F representing fixed bucket size\n
    intervals - list containing tuple intervals. pos0-interval start, pos1-interval end. for QS these are dates, for F these are indexes. not necessary for Month interval type.\n
    cold_start_buckets - number of buckets to be used for training only\n
    '''
    N = 10
#     print('0',data.shape[0]) # debug
    print('Creating buckets. . .')
    buckets = []
    assert interval_type in ['W', 'M', 'QS', 'F'], "interval must be one of W, M, QS, or F"
    if interval_type == 'W':
        # create buckets based on months
        weeks = data['week'].unique()
        for interval in weeks:
            idx = (data['week'] == interval)
            buckets.append( data[idx] )
    elif interval_type == 'M':
        # create buckets based on months
        months = data['date'].unique()
        for interval in months:
            idx = (data['date'] == interval)
            buckets.append( data[idx] )
    elif interval_type == 'QS':
        # create buckets based on quarters or semesters
        for s, e in intervals:
            idx = (data['date'] >= s) & (data['date'] <= e)
            buckets.append( data[idx] )
        else:
            idx = (data['date'] > e)
            buckets.append( data[idx] )
    else:
        # create buckets based on fixed number of examples
        for i, j in intervals:
            buckets.append( data.iloc[i:j] )
    
#     # debug
#     a = pd.concat( buckets ).set_index([user_col, item_col])
#     print('1',a.shape[0])
    
    print('Creating holdouts. . .')
    # create holdouts with last user interaction
    holdouts = []
    frequent_users_seen = [] # frequent users must have been seen at least once before being sent to holdouts. 
    # Imagine if the first frequent user interaction is the single interaction by this user in an interval, then this single interaction cant be sent to the holdout.
    for i, b in enumerate( buckets ):
        if i >= cold_start_buckets:
            last_interaction_idx = []
            for u in frequent_users:
                idx = b[user_col] == u
                if (idx.sum() == 1) and (u not in frequent_users_seen): # first condition to see if user appears once, second to see if user was not seen before - then it wont go to holdout, and it will be marked as seen
                    frequent_users_seen.append(u)
                    continue
                elif idx.sum() > 0: # else, if user appears at least once, append index to holdout
                    last_interaction_idx.append( b[ idx ].index[-1] )
                    if (u not in frequent_users_seen): # and if user hasnt been seen, mark as seen (he must appear at least twice then)
                        frequent_users_seen.append(u)
            holdout = b.loc[ last_interaction_idx ] # get last interactions as holdout
            holdout.reset_index(drop=True, inplace=True) # reset index required - implicitdata indexes user by their previous index
            holdouts.append(holdout) # append to holdouts
            buckets[i] = b.drop( index = last_interaction_idx).reset_index(drop=True) # remove last interactions from bucket
#             # debug
#             a = pd.concat( buckets ).set_index([user_col, item_col])
#             b = pd.concat( holdouts )[[user_col, item_col]].set_index([user_col, item_col])
#             print('2', a.reset_index().shape[0] + b.reset_index().shape[0] )
        else: # if bucket belongs to 'cold_start_buckets'
            buckets[i] = b.reset_index(drop=True)
            for u in frequent_users: # as before, we mark frequent users in the cold start bucket as seen
                idx = b[user_col] == u
                if (idx.sum() > 0):
                    frequent_users_seen.append(u)
    
    print('Cleaning holdouts. . .')
    # a verification is required to remove any items in the holdouts from the buckets
    # i.e. items that are in holdouts can never be used for training
    # can this be done while the holdouts and buckets are created?  

    # for each holdout:
    #   set interaction tuple as index 
    #   perform a inner join with the buckets dataframe.
    #   get the unique interactions that occur in both (i.e. resulting unique indexes)
    #   append these interactions to the respective bucket
    #   remove these interactions from the holdout

    for i, _ in enumerate(holdouts): 
        buckets_df = pd.concat( buckets )[[user_col, item_col]].set_index([user_col, item_col]) # concatenate all holdouts and set interaction tuple as index.
        temp_h = holdouts[i].set_index([user_col, item_col])
        common_interactions = temp_h.join(buckets_df, how='inner').index
        common_interactions = np.unique( common_interactions )
        print(f'common interactions between holdout {i+1} and all buckets: {len(common_interactions)}')
        bucket = buckets[i+cold_start_buckets].append(temp_h.loc[ common_interactions ].reset_index()).sort_values(by='timestamp').reset_index(drop=True)
        buckets[i+cold_start_buckets] = bucket
        holdouts[i] = temp_h.drop(index=common_interactions).reset_index()
                
                
    print('Converting to ImplicitData. . .')
    for i, b in enumerate(buckets):
        buckets[i] = ImplicitData(user_list=b[user_col], item_list=b[item_col]) # convert to ImplicitData

    for j, h in enumerate(holdouts):
        holdouts[j] = ImplicitData(user_list=h[user_col], item_list=h[item_col]) # convert to ImplicitData
    
    print('Done!')
    return buckets, holdouts