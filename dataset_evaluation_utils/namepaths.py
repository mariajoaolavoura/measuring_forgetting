import os


def get_folderpaths(dump_foldername:str, sample_foldername:str=''):
    '''
        creates output and image filepaths following the rule: 
            what <image or output>/which_data_set/sample/ what <diversity_eval or heatmaps or other>/
    '''
    
    images_path = 'images/'+dump_foldername+sample_foldername
    output_path = 'output/'+dump_foldername+sample_foldername

    heatmaps_path = images_path+'heatmaps/'

    diversity_graphpath = images_path+'diversity_eval/'
    diversity_filepath = output_path+'diversity_eval/'

    validate_folderpath(images_path)
    validate_folderpath(output_path)
    validate_folderpath(heatmaps_path)
    validate_folderpath(diversity_graphpath)
    validate_folderpath(diversity_filepath)

    return images_path, output_path, heatmaps_path, diversity_graphpath, diversity_filepath


def validate_folderpath(folderpath):
    if not os.path.exists(folderpath):
        os.makedirs(folderpath)
        print(folderpath)


# def get_namepaths_Palco2010_ISGD(sample_year_month,
#                                         interval_type,
#                                         dump_filename,
#                                         use_data_unique_users,
#                                         frequent_users_thr,
#                                         cold_start_buckets,
#                                         to_grid_search,
#                                         num_factors,
#                                         num_iter,
#                                         learn_rate,
#                                         regularization,
#                                         random_seed):
    
#     return get_namepaths(data_name = '_palco2010',
#                           data_print_name = 'Palco2010',
#                           sample_year_month = sample_year_month,
#                           interval_type = interval_type,
#                           dump_filename = dump_filename,
#                           frequent_users_thr = frequent_users_thr,
#                           cold_start_buckets = cold_start_buckets,
#                           use_data_unique_users = use_data_unique_users,
#                           to_grid_search = to_grid_search,
#                           num_factors = num_factors,
#                           num_iter = num_iter,
#                           learn_rate = learn_rate,
#                           regularization = regularization,
#                           random_seed = random_seed,
#                           model_print_name = 'ISGD')


def get_namepaths(data_name:str,
                   data_print_name:str,
                   sample_year_month,
                   interval_type:str,
                   dump_filename:str,
                   use_data_unique_users,
                   frequent_users_thr,
                   cold_start_buckets,
                   to_grid_search,
                   num_factors,
                   num_iter,
                   learn_rate,
                   regularization,
                   random_seed,
                   model_print_name):
    
    sample_year_month_start = sample_year_month[0][0]
    sample_year_month_end = sample_year_month[1][0]

    if cold_start_buckets:
        sample_str = str(sample_year_month_start)+'_until_'+str(sample_year_month_end)+'+cold_start'
    else:
        sample_str = str(sample_year_month_start)+'_until_'+str(sample_year_month_end)

    

    dataset_name = 'sample_'+sample_str+data_name
    user_col = 'user_id'
    item_col = 'item_id'

    output_path = 'output/'+dump_filename+'/'

    data_path = output_path+''+dataset_name+'.csv'

    if frequent_users_thr:
        frequent_users_path = output_path+'sample_'+sample_str+'_frequent_users_'+str(frequent_users_thr)+'.joblib' 
        sample_str += '_fu_'+str(frequent_users_thr)

    else:    
        frequent_users_path = output_path+'sample_'+sample_str+'_frequent_users.joblib' 


    if interval_type == 'Q':
        intervals_path = output_path+'sample_'+sample_str+'_trimestres.joblib'
        bucket_freq =  'quarterly'
    elif interval_type == 'S':
        intervals_path = output_path+'sample_'+sample_str+'_semestres.joblib'
        bucket_freq = 'semesterly'
    else:
        # interval_type == 'M'
        intervals_path = None
        bucket_freq = 'monthly'


    buckets_path = output_path+'sample_'+sample_str+'_'+bucket_freq+'_buckets.joblib'
    holdouts_path = output_path+'sample_'+sample_str+'_'+bucket_freq+'_holdouts.joblib'

    results_matrix_path = output_path+''+dataset_name+' '+bucket_freq+'_bucket '+model_print_name+' results.csv'

    recall_heatmap_title = 'Recall@20 for '+model_print_name+' checkpoints across Holdouts ('+sample_str+'_'+bucket_freq+') - '+data_print_name
    recall_heatmap_path = 'images/heatmaps/'+dump_filename+'/'+dataset_name+' '+bucket_freq+'_bucket '+model_print_name+' heatmap.png'

    incrementalTraining_time_record_path = output_path+''+dataset_name+' '+bucket_freq+'_bucket '+model_print_name+' training time.joblib'
    evaluateHoldouts_time_record_path =  output_path+''+dataset_name+' '+bucket_freq+'_bucket '+model_print_name+' eval time.joblib'

    return {'sample_str': sample_str, 
            'dataset_name': dataset_name,
            'user_col': user_col, 
            'item_col': item_col,
            'output_path': output_path, 
            'data_path': data_path,
            'use_data_unique_users': use_data_unique_users,
            'frequent_users_path':frequent_users_path,
            'cold_start_buckets':cold_start_buckets,
            'to_grid_search':to_grid_search,
            'num_factors': num_factors,
            'num_iter': num_iter,
            'learn_rate': learn_rate,
            'regularization': regularization,
            'random_seed': random_seed,
            'interval_type': interval_type,
            'intervals_path': intervals_path,
            'bucket_freq': bucket_freq,
            'buckets_path': buckets_path,
            'holdouts_path': holdouts_path,
            'results_matrix_path': results_matrix_path,
            'recall_heatmap_title': recall_heatmap_title,
            'recall_heatmap_path': recall_heatmap_path,
            'incrementalTraining_time_record_path': incrementalTraining_time_record_path,
            'evaluateHoldouts_time_record_path': evaluateHoldouts_time_record_path}

