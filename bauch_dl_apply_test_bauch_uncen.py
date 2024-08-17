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
with open("preproccessed_data_uncensored.pickle", "rb") as f:
    data_dic = pickle.load(f)

test = data_dic['test']
test_target = data_dic['test_target']
resids = test
seq_len = len(resids)


def get_dl_predictions(resids, model_type, kk):
    
    '''
    Generate DL prediction time series on resids
    from DL classifier with type 'model_type' and index kk.
    '''

    # Load in specific DL classifier
    model_name = "best_models/best_model_7_2_len1500.pkl"
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


preds = get_dl_predictions(resids, 1, 1)
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


with open("bauch_apply_bauch_test_uncen.pickle", "wb") as f:
    pickle.dump(preds, f)


# pred = np.argmax(list_of_preds, axis=2)




