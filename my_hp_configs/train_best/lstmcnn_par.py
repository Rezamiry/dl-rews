import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D, AveragePooling1D, concatenate
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
from tensorflow.keras import backend as K

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
# from sklearn.model_selection import train_test_split
# train, _, train_target, _ = train_test_split(train, train_target, test_size=0.8, stratify=train_target, random_state=0)
# validation, _, validation_target, _ = train_test_split(validation, validation_target, test_size=0.8, stratify=validation_target, random_state=0)

def build_model(hp):
    # # best params
    ## loss: 0.6766 - accuracy: 0.7342 - val_loss: 0.7242 - val_accuracy: 0.7121
    ## loss: 1.1268 - accuracy: 0.5055 - val_loss: 1.1754 - val_accuracy: 0.4811
    # n_lstm_layers = 1
    # mm_cells = 90
    # n_conv_layers = 3
    # n_filters = 32
    # kernel_size = 15
    # pool_size = 4
    # use_global_averaging = 1
    # recurrent_dropout = 0
    # lstm_dropout = 0.4
    # kernel_regulizer_l2 = 0.00035813
    # cnn_dropout = 0.45
    # learning_rate = 0.003


    # # loss: 0.6955 - accuracy: 0.7254 - val_loss: 0.7257 - val_accuracy: 0.7119
    # # loss: 0.6735 - accuracy: 0.7345 - val_loss: 0.7460 - val_accuracy: 0.6979
    # n_lstm_layers = 1
    # mm_cells = 90
    # n_conv_layers = 4
    # n_filters = 64
    # kernel_size = 15
    # pool_size = 4
    # use_global_averaging = 1
    # recurrent_dropout = 0
    # lstm_dropout = 0.4
    # kernel_regulizer_l2 = 0.00035813
    # cnn_dropout = 0.45
    # learning_rate = 0.002

    # loss: 0.6552 - accuracy: 0.7424 - val_loss: 0.7136 - val_accuracy: 0.7156
    # loss: 0.6285 - accuracy: 0.7528 - val_loss: 0.7501 - val_accuracy: 0.7016
    # n_lstm_layers = 2
    # mm_cells = 90
    # n_conv_layers = 4
    # n_filters = 64
    # kernel_size = 15
    # pool_size = 4
    # use_global_averaging = 1
    # recurrent_dropout = 0
    # lstm_dropout = 0.4
    # kernel_regulizer_l2 = 0.00035813
    # cnn_dropout = 0.45
    # learning_rate = 0.001

    # loss: 0.6784 - accuracy: 0.7329 - val_loss: 0.7258 - val_accuracy: 0.7134
    # loss: 0.6442 - accuracy: 0.7455 - val_loss: 0.7268 - val_accuracy: 0.7095
    # n_lstm_layers = 2
    # mm_cells = 90
    # n_conv_layers = 3
    # n_filters = 32
    # kernel_size = 15
    # pool_size = 4
    # use_global_averaging = 1
    # recurrent_dropout = 0
    # lstm_dropout = 0.4
    # kernel_regulizer_l2 = 0.00035813
    # cnn_dropout = 0.45
    # learning_rate = 0.0001

    n_lstm_layers = 3
    mm_cells = 90
    n_conv_layers = 5
    n_filters = 64
    kernel_size = 15
    pool_size = 4
    use_global_averaging = 1
    recurrent_dropout = 0
    lstm_dropout = 0.4
    kernel_regulizer_l2 = 0.00035813
    cnn_dropout = 0.45
    learning_rate = 0.0001


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
    # print("model parameters: " , model.count_params()/1000)
    # print(model.summary())
    return model


model = build_model(None)
print(model.summary())
epoch_param = 200
batch_size = 640

model_name = "best_lstmcnn_par.pickle"
chk = ModelCheckpoint(model_name, monitor='val_accuracy', save_best_only=True, mode='max', verbose=2)
es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=2, patience=30)
model.fit(train, train_target, epochs=epoch_param, batch_size=batch_size, callbacks=[es, chk], verbose=2,  validation_data=(validation,validation_target))

