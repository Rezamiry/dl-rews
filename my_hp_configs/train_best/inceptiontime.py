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
from tensorflow.keras.layers import Flatten, Add
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

def inception_module(inputs, num_filters):
    conv1 = Conv1D(num_filters, kernel_size=1, activation='relu', padding='same')(inputs)

    conv3 = Conv1D(num_filters, kernel_size=3, activation='relu', padding='same')(inputs)

    conv5 = Conv1D(num_filters, kernel_size=5, activation='relu', padding='same')(inputs)

    maxpool = MaxPooling1D(pool_size=3, strides=1, padding='same')(inputs)
    convpool = Conv1D(num_filters, kernel_size=1, activation='relu', padding='same')(maxpool)

    output = concatenate([conv1, conv3, conv5, convpool], axis=2)
    return output


def build_model(hp):
    dropout_0 = 0.15
    # dropout_1 = hp.Float(f'dropout_1', 0, 0.8, step=0.05, default=0.5)
    # dropout_2 = hp.Float(f'dropout_2', 0, 0.8, step=0.05, default=0.5)
    # dropout_3 = hp.Float(f'dropout_3', 0, 0.8, step=0.05, default=0.5)
    n_filters_1 = 32
    # n_filters_2 = hp.Int(f"n_filters_2", min_value=32, max_value=256, step=32)
    # n_filters_3 = hp.Int(f"n_filters_3", min_value=32, max_value=256, step=32)
    # n_filters_4 = hp.Int(f"n_filters_4", min_value=32, max_value=256, step=32)
    learning_rate = 0.001

    inputs = Input(shape=(1500,1))
    
    res1x = Conv1D(n_filters_1*4, kernel_size=1, padding='same')(inputs)

    x = inception_module(inputs, n_filters_1)
    x = inception_module(x, n_filters_1)
    x = inception_module(x, n_filters_1)
    x = Add()([x, res1x])
    x = Activation("relu")(x)
    
    x = Dropout(dropout_0)(x)

    res2x = Conv1D(n_filters_1*8, kernel_size=1, padding='same')(x)

    x = inception_module(x, n_filters_1*2)
    x = inception_module(x, n_filters_1*2)
    x = inception_module(x, n_filters_1*2)
    x = Add()([x, res2x])
    x = Activation("relu")(x)

    x = Dropout(dropout_0)(x)

    x = inception_module(x, n_filters_1)

    x = BatchNormalization()(x)
    x = GlobalAveragePooling1D()(x)

    x = Dense(100, activation='relu')(x)
    x = Dropout(dropout_0)(x)
    x = Dense(20, activation='relu')(x)
    x = Dense(4, activation='softmax')(x)


    model = Model(inputs=inputs, outputs=x)
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # print("model parameters: {} K".format(model.count_params()//1000))
    return model


model = build_model(None)
print(model.summary())
epoch_param = 200
batch_size = 64

model_name = "best_inceptiontime.pickle"
chk = ModelCheckpoint(model_name, monitor='val_accuracy', save_best_only=True, mode='max', verbose=2)
es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=30)
model.fit(train, train_target, epochs=epoch_param, batch_size=batch_size, callbacks=[es, chk],verbose=2, validation_data=(validation,validation_target))

