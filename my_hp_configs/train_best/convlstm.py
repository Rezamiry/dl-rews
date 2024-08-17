import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D, AveragePooling1D
from tensorflow.keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D
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


n_steps = 1
n_length = 1500
n_features = 1
n_outputs = 4
train = train.reshape((train.shape[0], n_steps, 1, n_length, n_features))
validation = validation.reshape((validation.shape[0], n_steps, 1, n_length, n_features))
test = test.reshape((test.shape[0], n_steps, 1, n_length, n_features))


def build_model(hp):
    
    # # best parameters
    # # bad overfitted performance
    # num_layers = 2
    # n_filters = 48
    # kernel_size = 16
    # kernel_regulizer_l2 = 6.2751e-07
    # recurrent_regulizer_l2 = 0.00021595
    # dense_regulizer_l2 = 1e-07
    # dropout = 0
    # # recurrent_dropout = hp.Float(f'recurrent_dropout', 0, 0.8, step=0.1, default=0)
    # recurrent_dropout = 0
    # dense_dropout = 0.5
    # learning_rate = 0.0045

    # # loss: 0.8347 - accuracy: 0.6646 - val_loss: 0.9469 - val_accuracy: 0.6087
    # # loss: 0.5397 - accuracy: 0.8077 - val_loss: 1.2004 - val_accuracy: 0.5670
    # num_layers = 2
    # n_filters = 24
    # kernel_size = 16
    # kernel_regulizer_l2 = 6.2751e-07
    # recurrent_regulizer_l2 = 0.00021595
    # dense_regulizer_l2 = 1e-07
    # dropout = 0.3
    # recurrent_dropout = 0.3
    # dense_dropout = 0.5
    # learning_rate = 0.0025

    num_layers = 2
    n_filters = 8
    kernel_size = 16
    kernel_regulizer_l2 = 6.2751e-07
    recurrent_regulizer_l2 = 0.00021595
    dense_regulizer_l2 = 1e-07
    dropout = 0.5
    recurrent_dropout = 0.5
    dense_dropout = 0.7
    learning_rate = 0.001

    model = Sequential()
    for i in range(num_layers):
        if i == 0:
            model.add(ConvLSTM2D(
            filters=n_filters*(2**i),
            kernel_size=kernel_size,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            kernel_regularizer=regularizers.l2(kernel_regulizer_l2),
            recurrent_regularizer=regularizers.l2(recurrent_regulizer_l2),
            activation='relu',
            padding='same',
            return_sequences=True,
            input_shape=(n_steps, 1, n_length, n_features)))
        elif i < (num_layers-1):
            model.add(ConvLSTM2D(
            filters=n_filters*(2**i),
            kernel_size=kernel_size,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            kernel_regularizer=regularizers.l2(kernel_regulizer_l2),
            recurrent_regularizer=regularizers.l2(recurrent_regulizer_l2),
            activation='relu',
            padding='same',
            return_sequences=True))
        else:
            model.add(ConvLSTM2D(
            filters=n_filters*(2**i),
            kernel_size=kernel_size,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            kernel_regularizer=regularizers.l2(kernel_regulizer_l2),
            recurrent_regularizer=regularizers.l2(recurrent_regulizer_l2),
            activation='relu',
            padding='same',
            return_sequences=False))
            
    model.add(Flatten())
    model.add(Dense(100, kernel_initializer = 'lecun_normal', kernel_regularizer=regularizers.l2(dense_regulizer_l2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dense_dropout))
    model.add(Dense(100, kernel_initializer = 'lecun_normal', kernel_regularizer=regularizers.l2(dense_regulizer_l2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dense_dropout))
    model.add(Dense(20, kernel_initializer = 'lecun_normal', kernel_regularizer=regularizers.l2(dense_regulizer_l2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(4, activation="softmax"))
    
    
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    print("model parameters: {}K".format(model.count_params()//1000))
    return model
    
    

model = build_model(None)
epoch_param = 200
batch_size = 1024

model_name = "best_convlstm.pickle"
chk = ModelCheckpoint(model_name, monitor='val_accuracy', save_best_only=True, mode='max', verbose=2)
es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=30)
model.fit(train, train_target, epochs=epoch_param, batch_size=batch_size, callbacks=[es, chk], validation_data=(validation,validation_target))

