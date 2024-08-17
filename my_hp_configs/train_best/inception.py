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

# # search with a subset of the data
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
    
    # # best params
    # # bad performance - val_acc=0.5945
    # n_layers = 3
    # n_filters = 64
    # dropout_inception = 0.05
    # dense_dropout = 0.7
    # learning_rate=3e-5

    # loss: 0.8636 - accuracy: 0.6502 - val_loss: 4.3850 - val_accuracy: 0.3490
    # loss: 0.8550 - accuracy: 0.6533 - val_loss: 4.4082 - val_accuracy: 0.3307
    # n_layers = 4
    # n_filters = 64
    # dropout_inception = 0.3
    # dense_dropout = 0.3
    # learning_rate=0.001

    n_layers = 3
    n_filters = 128
    dropout_inception = 0.05
    dense_dropout = 0.7
    learning_rate=3e-6
    
    inputs = Input(shape=(1500,1))
    for i in range(n_layers):
        if i == 0:
            x = inception_module(inputs, n_filters*(2**i))
        else:
            x = inception_module(x, n_filters*(2**i))
        
        x = Dropout(dropout_inception)(x)

    x = GlobalAveragePooling1D()(x)
    x = Dense(100, kernel_initializer = 'lecun_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(dense_dropout)(x)
    x = Dense(100, kernel_initializer = 'lecun_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(dense_dropout)(x)
    x = Dense(20, kernel_initializer = 'lecun_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    Y_HAT = Dense(4, activation="softmax")(x)
    model = Model(inputs=inputs, outputs=Y_HAT)
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    print("model parameters: {} M".format(model.count_params()//1000000))
    return model


model = build_model(None)
print(model.summary())
epoch_param = 200
batch_size = 256

model_name = "best_inception.pickle"
chk = ModelCheckpoint(model_name, monitor='val_accuracy', save_best_only=True, mode='max', verbose=2)
es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=30)
model.fit(train, train_target, epochs=epoch_param, batch_size=batch_size, callbacks=[es, chk], verbose=2, validation_data=(validation,validation_target))

