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


'''
10 of the DL models are trained on training set data that are censored (padded) 
on the left hand side only, hence they are trained on data that always includes 
the transition.  The other 10 were trained on training set data that are 
censored on both left and right, hence they do not include the transition.
kk runs from 1 to 10 and denotes the index of the 10 models of each type.
model_type is 1 or 2.  1 denotes the model that is trained on data censored on 
both the left and right.  2 is the model trained on data censored on the left only. 

'''


# Filepath to residual time series to make predictions on 
#filepath = '../test_models/may_fold_1500/data/resids/may_fold_null_1500_resids_1.csv'



filepath_out = f"bauch_20models_1500_test_preds.csv"
ts_len=1500

# Filepath to export ensemble DL predictions to
#filepath_out = '../test_models/may_fold_1500/data/ml_preds_test/ensemble_trend_probs_may_fold_null_forced_1_len1500.csv'

# Type of classifier to use (1500 or 500)
# ts_len=1500

'''  
The following two parameters control how many sample points along the 
timeseries are used, and the length between them.  For instance, for an input 
time series equal to or less then length 1500, mult_factor=10 and 
pad_samples=150 means that it will do the unveiling in steps of 10 datapoints, 
at 150 evenly spaced points across the entire time series (150 x 10 = 1500).
Needs to be adjusted according to whether you are using the trained 500 length 
or 1500 length classifier.
'''

# Steps of datapoints in between each DL prediction
mult_factor = 10

# Total number of DL predictions to make
# Use 150 for length 1500 time series. Use 50 for length 500 time series.
pad_samples = 150



# Load residual time series data
# df = pd.read_csv(filepath).dropna()
# resids = df['Residuals'].values.reshape(1,-1,1)
# Length of inupt time series
# seq_len = len(df)

import pickle
# load the data
with open("preproccessed_data.pickle", "rb") as f:
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
    model_name = 'best_models/best_model_{}_{}_len{}.pkl'.format(kk,model_type,ts_len)
    model = load_model(model_name)
    y_pred = []
    for ind in range(resids.shape[0]):
        resid = resids[ind,:,:]
        resid = resid.reshape(1,-1,1)
        y_pred.append(model.predict(resid,verbose=0))
    # Delete model and do garbage collection to free up RAM
    tf.keras.backend.clear_session()
    del model
    gc.collect()
    
    return y_pred



# Compute DL predictions from all 20 trained models
pred_dic = {}
for model_type in [1,2]:                                
    for kk in np.arange(1,11):
        print('Compute DL predictions for model_type={}, kk={}'.format(
            model_type,kk))
        preds = get_dl_predictions(resids, model_type, kk)
        pred_dic[f"model_type-{model_type}_kk-{kk}"] = preds



# Compute average prediction among all 20 DL classifiers
list_df_preds = []
for pred_ind in range(len(resids)):
    temp_pred = []
    for model_type in [1,2]:
        for kk in np.arange(1,11):
            temp_pred.append(pred_dic[f"model_type-{model_type}_kk-{kk}"][pred_ind])
    list_df_preds.append(np.array(temp_pred).mean(axis=0))


with open("bauch_apply_test.pickle", "wb") as f:
    pickle.dump(list_df_preds, f)


# pred = np.argmax(list_of_preds, axis=2)




