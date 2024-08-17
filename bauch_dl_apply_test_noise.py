'''
Code to generate ensemble predictions from the DL classifiers
on a give time series of residuals

'''


import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings("ignore")

import gc
import numpy as np
import pandas as pd
import random
import sys
import itertools

import tensorflow as tf
from tensorflow.keras.models import load_model
from datetime import datetime
tf.get_logger().setLevel('ERROR')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

random.seed(datetime.now())


import pickle
# load the data
with open("train_amit/envn_uncen_resids_dic.pickle", "rb") as f:
    data_dic_t1 = pickle.load(f)

with open("train_amit/whiten_uncen_resids_dic.pickle", "rb") as f:
    data_dic_t2 = pickle.load(f)

with open("train_amit/demn_uncen_resids_dic.pickle", "rb") as f:
    data_dic_t3 = pickle.load(f)
    
test_envn = data_dic_t1['test']
test_envn_target = data_dic_t1['test_target']
test_whiten = data_dic_t2['test']
test_whiten_target = data_dic_t2['test_target']
test_demn = data_dic_t3['test']
test_demn_target = data_dic_t3['test_target']


def get_dl_predictions(resids, model_type, kk):
    
    '''
    Generate DL prediction time series on resids
    from DL classifier with type 'model_type' and index kk.
    '''

    # Load in specific DL classifier
    model_name = "best_models/best_model_1_1_len1500.pkl"
    model = load_model(model_name)
    y_pred = []
    for ind in range(len(resids)):
        resid = resids[ind]
        resid = resid.reshape(1,-1,1)
        y_pred.append(model.predict(resid,verbose=0))
    # Delete model and do garbage collection to free up RAM
    tf.keras.backend.clear_session()
    del model
    gc.collect()
    
    return y_pred


preds_envn = get_dl_predictions(test_envn, 1, 1)
preds_whiten = get_dl_predictions(test_whiten, 1, 1)
preds_demn = get_dl_predictions(test_demn, 1, 1)

with open("bauch_apply_test_envn_uncen.pickle", "wb") as f:
    pickle.dump(preds_envn, f)

    
with open("bauch_apply_test_whiten_uncen.pickle", "wb") as f:
    pickle.dump(preds_whiten, f)

    
with open("bauch_apply_test_demn_uncen.pickle", "wb") as f:
    pickle.dump(preds_demn, f)

# # Compute DL predictions from all 20 trained models
# pred_dic = {}
# for model_type in [1,2]:                                
#     for kk in np.arange(1,11):
#         print('Compute DL predictions for model_type={}, kk={}'.format(
#             model_type,kk))
        
#         pred_dic[f"model_type-{model_type}_kk-{kk}"] = preds



# # Compute average prediction among all 20 DL classifiers
# list_df_preds = []
# for pred_ind in range(len(resids)):
#     temp_pred = []
#     for model_type in [1,2]:
#         for kk in np.arange(1,11):
#             temp_pred.append(pred_dic[f"model_type-{model_type}_kk-{kk}"][pred_ind])
#     list_df_preds.append(np.array(temp_pred).mean(axis=0))


# with open("amit_apply_test.pickle", "wb") as f:
#     pickle.dump(preds, f)


# pred = np.argmax(list_of_preds, axis=2)




