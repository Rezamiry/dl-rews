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

# search with a subset of the data
from sklearn.model_selection import train_test_split
train, _, train_target, _ = train_test_split(train, train_target, test_size=0.8, stratify=train_target, random_state=0)
validation, _, validation_target, _ = train_test_split(validation, validation_target, test_size=0.8, stratify=validation_target, random_state=0)


n_steps = 1
n_length = 1500
n_features = 1
n_outputs = 4
train = train.reshape((train.shape[0], n_steps, 1, n_length, n_features))
validation = validation.reshape((validation.shape[0], n_steps, 1, n_length, n_features))
test = test.reshape((test.shape[0], n_steps, 1, n_length, n_features))


def build_model(hp):
    model = Sequential()
    num_layers = hp.Int("num_layers", 1, 2)
    n_filters = hp.Int(f"n_filters", min_value=16, max_value=64, step=16)
    kernel_size = hp.Int(f"kernel_size", min_value=4, max_value=16, step=4)
    kernel_regulizer_l2 = hp.Float('kernel_regulizer_l2', 1e-7, 1e-1, sampling='log')
    recurrent_regulizer_l2 = hp.Float('recurrent_regulizer_l2', 1e-7, 1e-1, sampling='log')
    dense_regulizer_l2 = hp.Float('dense_regulizer_l2', 1e-7, 1e-1, sampling='log')
    dropout = hp.Float(f'dropout', 0, 0.8, step=0.05, default=0.5)
    # recurrent_dropout = hp.Float(f'recurrent_dropout', 0, 0.8, step=0.1, default=0)
    recurrent_dropout = 0
    dense_dropout = hp.Float(f'dropout_dense', 0, 0.8, step=0.05)
    learning_rate = hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')

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
    
    

hp_name = "hp_convlstm"

import keras_tuner as kt

# tuner = kt.Hyperband(
#     build_model,
#     objective='val_accuracy',
#     max_epochs=100,
#     hyperband_iterations=2,
#     directory=f"./my_hp_results/{hp_name}",
#     overwrite=True
# )

# tuner.search(train,
#     train_target,
#     validation_data=(validation,validation_target),
#     epochs=100,
#     batch_size=256,
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
    epochs=100,
    batch_size=256,
    verbose=2,
    callbacks=[EarlyStopping(patience=10)])

print(tuner.results_summary())

top_hps = tuner.get_best_hyperparameters(5)

with open(f"./my_hp_results/tuner_{hp_name}_bayesian.pickle", "wb") as f:
    pickle.dump(top_hps, f)
