import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D, AveragePooling1D, concatenate, Permute
from tensorflow.keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, Reshape, multiply
from tensorflow.keras.layers import ConvLSTM2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import BatchNormalization

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras import regularizers

from datetime import datetime
import random 

random.seed(0)

import pickle
with open("./preproccessed_data.pickle", 'rb') as f:
    data = pickle.load(f)
    
train = data['train']
train_target = data['train_target']
validation = data['validation']
validation_target = data['validation_target']
test = data['test']
test_target = data['test_target']

# search with a subset of the data
from sklearn.model_selection import train_test_split
train, _, train_target, _ = train_test_split(train, train_target, test_size=0.8, stratify=train_target, random_state=0)
validation, _, validation_target, _ = train_test_split(validation, validation_target, test_size=0.8, stratify=validation_target, random_state=0)


def build_model(hp):
    n_lstm_layers = hp.Int("n_lstm_layers", 1, 3)
    mm_cells = hp.Int(f"mem_cells", min_value=10, max_value=100, step=10)

    n_conv_layers = hp.Int("n_conv_layers", 2, 8)    
    n_filters = hp.Int(f"n_filters", min_value=16, max_value=128, step=8)
    kernel_size = hp.Int(f"kernel_size", min_value=6, max_value=15, step=1)
    pool_size = hp.Int(f"pool_size", min_value=2, max_value=8, step=1)
    use_global_averaging = hp.Choice('use_global_averaging', [True, False])

    # recurrent_dropout = hp.Float(f'recurrent_dropout', 0, 0.8, step=0.05)
    recurrent_dropout = 0
    lstm_dropout = hp.Float(f'lstm_dropout', 0, 0.8, step=0.05, default=0.5)
    kernel_regulizer_l2 = hp.Float('kernel_regulizer_l2', 1e-7, 1e-1, sampling='log')
    cnn_dropout = hp.Float(f'cnn_dropout', 0, 0.8, step=0.05, default=0.5)

    learning_rate = hp.Float('learning_rate', 1e-7, 1e-2, sampling='log')


    ip = Input(shape=(1500, 1))
    for i in range(n_lstm_layers):
        if i == 0:
            x = LSTM(mm_cells,
                    recurrent_dropout = recurrent_dropout,
                    return_sequences= i!=(n_lstm_layers-1),
                    kernel_initializer = 'lecun_normal')(ip)
        else:
            x = LSTM(mm_cells,
                    recurrent_dropout =  recurrent_dropout,
                    return_sequences= i!=(n_lstm_layers-1),
                    kernel_initializer = 'lecun_normal')(x)
        
        if i!=(n_lstm_layers-1): # don't add dropout at the last layer
            x = Dropout(lstm_dropout)(x)
    
    # y = Permute((2, 1))(ip)
    for j in range(n_conv_layers):
        if j == 0:
            y = Conv1D(filters=n_filters*(2**j),
                    kernel_size=kernel_size,
                    kernel_initializer = 'lecun_normal',
                    kernel_regularizer=regularizers.l2(kernel_regulizer_l2),
                    padding='same')(ip)
        else:
            y = Conv1D(filters=n_filters*(2**j),
                    kernel_size=kernel_size,
                    kernel_initializer = 'lecun_normal',
                    kernel_regularizer=regularizers.l2(kernel_regulizer_l2),
                    padding='same')(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
        if (j+1) % 2 == 0: # every two layers one pooling
            y = AveragePooling1D(pool_size=pool_size)(y)
        y = Dropout(cnn_dropout)(y)
    
    if use_global_averaging:
        y = GlobalAveragePooling1D()(y)
    else:
        y = Flatten()(y)

    x = concatenate([x, y])
    out = Dense(4, activation='softmax')(x)
    model = Model(ip, out)
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    print("model parameters: " , model.count_params()/1000)
    # print(model.summary())
    return model


hp_name = "hp_lstmcnn_par"

import keras_tuner as kt

# tuner = kt.Hyperband(
#     build_model,
#     objective='val_accuracy',
#     max_epochs=500,
#     hyperband_iterations=2,
#     directory=f"./my_hp_results/{hp_name}",
#     overwrite=True
# )

# tuner.search(train,
#     train_target,
#     validation_data=(validation,validation_target),
#     epochs=500,
#     batch_size=128,
#     verbose=2,
#     callbacks=[EarlyStopping(patience=1)])

tuner = kt.BayesianOptimization(
    build_model,
    objective='val_accuracy',
    max_trials=200,
    executions_per_trial=2,
    directory=f"./my_hp_results/{hp_name}_bayesian",
    overwrite=True
)

tuner.search(train,
    train_target,
    validation_data=(validation,validation_target),
    epochs=200,
    batch_size=128,
    verbose=2,
    callbacks=[EarlyStopping(patience=10)])


print(tuner.results_summary())

top_hps = tuner.get_best_hyperparameters(5)

with open(f"./my_hp_results/tuner_{hp_name}_bayesian.pickle", "wb") as f:
    pickle.dump(top_hps, f)
