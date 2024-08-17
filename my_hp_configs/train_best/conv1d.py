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
    
    # best params
    # loss: 0.7090 - accuracy: 0.7215 - val_loss: 0.7459 - val_accuracy: 0.7036
    # loss: 0.5616 - accuracy: 0.7823 - val_loss: 0.8852 - val_accuracy: 0.6489
    num_layers = 6
    n_filters = 32
    kernel_size = 10
    pool_size = 6
    kernel_regulizer_l2 = 6.4948e-07
    dropout = 0.15
    dense_dropout = 0.35
    learning_rate = 0.0002

    # loss: 0.6940 - accuracy: 0.7319 - val_loss: 0.7540 - val_accuracy: 0.7036
    # loss: 0.3674 - accuracy: 0.8676 - val_loss: 1.1946 - val_accuracy: 0.6510 
    # num_layers = 8
    # n_filters = 32
    # kernel_size = 10
    # pool_size = 6
    # kernel_regulizer_l2 = 6.4948e-07
    # dropout = 0.15
    # dense_dropout = 0.35
    # learning_rate = 0.0002

    # loss: 0.7111 - accuracy: 0.7283 - val_loss: 0.7877 - val_accuracy: 0.6927
    # loss: 0.5443 - accuracy: 0.8007 - val_loss: 0.9996 - val_accuracy: 0.6607
    # num_layers = 8
    # n_filters = 32
    # kernel_size = 10
    # pool_size = 6
    # kernel_regulizer_l2 = 6.4948e-07
    # dropout = 0.25
    # dense_dropout = 0.45
    # learning_rate = 0.0002

    
    # # loss: 0.7146 - accuracy: 0.7230 - val_loss: 0.7375 - val_accuracy: 0.7028
    # # loss: 0.3243 - accuracy: 0.8831 - val_loss: 1.2284 - val_accuracy: 0.6585
    # num_layers = 8
    # n_filters = 32
    # kernel_size = 10
    # pool_size = 6
    # kernel_regulizer_l2 = 6.4948e-07
    # dropout = 0.1
    # dense_dropout = 0.3
    # learning_rate = 0.0002

    num_layers = 6
    n_filters = 32
    kernel_size = 10
    pool_size = 6
    kernel_regulizer_l2 = 6.4948e-07
    dropout = 0.1
    dense_dropout = 0.3
    learning_rate = 0.0001

    input_x = Input(shape=(1500,1,))
    for i in range(num_layers):
        if i == 0:
            x = Conv1D(filters=n_filters*(2**i),
                    kernel_size=kernel_size,
                    kernel_initializer = 'lecun_normal',
                    kernel_regularizer=regularizers.l2(kernel_regulizer_l2),
                    padding='same')(input_x)
        else:
            x = Conv1D(filters=n_filters*(2**i),
                    kernel_size=kernel_size,
                    kernel_initializer = 'lecun_normal',
                    kernel_regularizer=regularizers.l2(kernel_regulizer_l2),
                    padding='same')(x)
        
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        if (i+1) % 2 == 0: # every two layers one pooling
            x = MaxPooling1D(pool_size=pool_size)(x)
        x = Dropout(dropout)(x)
    

    x = GlobalAveragePooling1D()(x)
    x = Dense(100, kernel_initializer = 'lecun_normal', kernel_regularizer=regularizers.l2(kernel_regulizer_l2))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(dense_dropout)(x)
    x = Dense(100, kernel_initializer = 'lecun_normal', kernel_regularizer=regularizers.l2(kernel_regulizer_l2))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(dense_dropout)(x)
    x = Dense(20, kernel_initializer = 'lecun_normal', kernel_regularizer=regularizers.l2(kernel_regulizer_l2))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    Y_HAT = Dense(4, activation="softmax")(x)
    model = Model(inputs=input_x, outputs=Y_HAT)
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    # trainable_count = int(
    #     np.sum([K.count_params(p) for p in set(model.trainable_weights)]))
    # print('Trainable params: {:,}'.format(trainable_count))
    # print("model parameters: {}M".format(model.count_params()//1000000))
    print(model.summary())
    return model


model = build_model(None)
epoch_param = 200
batch_size = 640

model_name = "best_conv1d.pickle"
chk = ModelCheckpoint(model_name, monitor='val_accuracy', save_best_only=True, mode='max', verbose=2)
es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=30)
model.fit(train, train_target, epochs=epoch_param, batch_size=batch_size, callbacks=[es, chk], validation_data=(validation,validation_target), verbose=2)

