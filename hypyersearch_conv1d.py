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

from datetime import datetime
import random 

random.seed(0)

import pickle
with open("./preprocessed_data.pickle", 'rb') as f:
    data = pickle.load(f)
    
train = data['train']
train_target = data['train_target']
validation = data['validation']
validation_target = data['validation_target']
test = data['test']
test_target = data['test_target']

# search with a subset of the data
from sklearn.model_selection import train_test_split
train, _, train_target, _ = train_test_split(train, train_target, test_size=0.9, stratify=train_target, random_state=0)
validation, _, validation_target, _ = train_test_split(validation, validation_target, test_size=0.9, stratify=validation_target)


def build_model(hp):
    input_x = Input(shape=(1500,1,))
    for i in range(hp.Int("num_layers", 2, 5)):
        x = Conv1D(filters=hp.Int(f"n_filters_{i}", min_value=32, max_value=512, step=32),
                   kernel_size=hp.Int(f"kernel_size_{i}", min_value=3, max_value=12, step=1),
                   kernel_initializer = 'lecun_normal',
                   padding='same')(input_x)
        # if hp.Boolean("batch_normalization"):
        x = BatchNormalization()(x)
        # x = Activation(hp.Choice("activation", ['relu', 'gelu', 'tanh', 'sigmoid']))(x)
        x = Activation('relu')(x)
        if hp.Choice('pooling_' + str(i), ['avg', 'max']) == 'max':
            x = MaxPooling1D(pool_size=hp.Int(f"pool_{i}", min_value=2, max_value=6, step=1))(x)
        else:
            x = AveragePooling1D(pool_size=hp.Int(f"pool_{i}", min_value=2, max_value=6, step=1))(x)
        x = Dropout(hp.Float(f'dropout_{i}', 0, 0.8, step=0.1, default=0.5))(x)
    
    if hp.Choice('global_pooling_' + str(i), ['avg', 'max']) == 'avg':
        x = GlobalAveragePooling1D()(x)
    else:
        x = GlobalMaxPooling1D()(x)

    x = Dense(100, kernel_initializer = 'lecun_normal',)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(hp.Float(f'dropout_{i}', 0, 0.8, step=0.1, default=0.5))(x)
    x = Dense(100, kernel_initializer = 'lecun_normal',)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(hp.Float(f'dropout_{i}', 0, 0.8, step=0.1, default=0.5))(x)
    x = Dense(20, kernel_initializer = 'lecun_normal',)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    Y_HAT = Dense(4, activation="softmax")(x)
    model = Model(inputs=input_x, outputs=Y_HAT)
    model.compile(
        optimizer=Adam(learning_rate=hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model



hp_name = "hp_conv1d"

import keras_tuner as kt

tuner = kt.Hyperband(
    build_model,
    objective='val_accuracy',
    max_epochs=500,
    hyperband_iterations=2,
    directory="./my_hp_results/hp_conv1d",
    overwrite=True
)

tuner.search(train,
    train_target,
    validation_data=(validation,validation_target),
    epochs=500,
    callbacks=[EarlyStopping(patience=1), TensorBoard(f"/tmp/{hp_name}")])

print(tuner.results_summary())


with open(f"./my_hp_results/tuner_{hp_name}.pickle", "wb") as f:
    pickle.dump(tuner, f)
