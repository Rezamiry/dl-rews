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

def squeeze_excite_block(filters,input):                      
    se = GlobalAveragePooling1D()(input)
    se = Reshape((1, filters))(se) 
    se = Dense(filters//16, activation='relu')(se)
    se = Dense(filters, activation='sigmoid')(se)
    se = multiply([input, se])
    return se


def build_model(hp):
    
    # # best params
    # # loss: 0.7847 - accuracy: 0.7076 - val_loss: 0.7767 - val_accuracy: 0.7046
    # # loss: 0.7835 - accuracy: 0.7070 - val_loss: 0.7913 - val_accuracy: 0.6994
    # num_layers = 6
    # n_filters = 128
    # kernel_size = 9
    # pool_size = 7
    # kernel_regulizer_l2 = 3.9616e-07
    # dropout = 0.35
    # dense_dropout = 0.15
    # learning_rate = 0.009552

    # loss: 0.7634 - accuracy: 0.7156 - val_loss: 0.7653 - val_accuracy: 0.7125
    # loss: 0.7565 - accuracy: 0.7180 - val_loss: 0.8136 - val_accuracy: 0.6927
    # num_layers = 6
    # n_filters = 128
    # kernel_size = 9
    # pool_size = 7
    # kernel_regulizer_l2 = 3.9616e-07
    # dropout = 0.35
    # dense_dropout = 0.15
    # learning_rate = 0.005

    # loss: 0.7214 - accuracy: 0.7240 - val_loss: 0.7389 - val_accuracy: 0.7180
    # loss: 0.6857 - accuracy: 0.7491 - val_loss: 0.7981 - val_accuracy: 0.7050
    # num_layers = 6
    # n_filters = 128
    # kernel_size = 9
    # pool_size = 7
    # kernel_regulizer_l2 = 3.9616e-07
    # dropout = 0.35
    # dense_dropout = 0.15
    # learning_rate = 0.001

    # loss: 0.6288 - accuracy: 0.7534 - val_loss: 0.7421 - val_accuracy: 0.7143
    # loss: 0.4504 - accuracy: 0.8244 - val_loss: 0.9369 - val_accuracy: 0.6904
    # num_layers = 6
    # n_filters = 128
    # kernel_size = 9
    # pool_size = 7
    # kernel_regulizer_l2 = 3.9616e-07
    # dropout = 0.35
    # dense_dropout = 0.15
    # learning_rate = 0.0001

    num_layers = 6
    n_filters = 128
    kernel_size = 9
    pool_size = 7
    kernel_regulizer_l2 = 6.9616e-07
    dropout = 0.4
    dense_dropout = 0.2
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
        x = squeeze_excite_block(n_filters*(2**i),x)
        x = Activation('relu')(x)
        if (i+1) % 2 == 0: # every two layers one pooling
            x = AveragePooling1D(pool_size=pool_size)(x)
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
    return model


model = build_model(None)
print(model.summary())
epoch_param = 200
batch_size = 640

model_name = "best_conv1dse.pickle"
chk = ModelCheckpoint(model_name, monitor='val_accuracy', save_best_only=True, mode='max', verbose=2)
es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=30)
model.fit(train, train_target, epochs=epoch_param, batch_size=batch_size,verbose=2, callbacks=[es, chk], validation_data=(validation,validation_target))

