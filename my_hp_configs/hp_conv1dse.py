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
from sklearn.model_selection import train_test_split
train, _, train_target, _ = train_test_split(train, train_target, test_size=0.8, stratify=train_target, random_state=0)
validation, _, validation_target, _ = train_test_split(validation, validation_target, test_size=0.8, stratify=validation_target, random_state=0)

def squeeze_excite_block(filters,input):                      
    se = GlobalAveragePooling1D()(input)
    se = Reshape((1, filters))(se) 
    se = Dense(filters//16, activation='relu')(se)
    se = Dense(filters, activation='sigmoid')(se)
    se = multiply([input, se])
    return se


def build_model(hp):
    input_x = Input(shape=(1500,1,))
    num_layers = hp.Int("num_layers", 2, 8)
    n_filters = hp.Int(f"n_filters", min_value=16, max_value=128, step=8)
    kernel_size = hp.Int(f"kernel_size", min_value=4, max_value=15, step=1)
    pool_size = hp.Int(f"pool_size", min_value=2, max_value=8, step=1)
    kernel_regulizer_l2 = hp.Float('kernel_regulizer_l2', 1e-7, 1e-1, sampling='log')
    dropout = hp.Float(f'dropout', 0, 0.8, step=0.05, default=0.5)
    dense_dropout = hp.Float(f'dropout_dense', 0, 0.8, step=0.05, default=0.5)
    learning_rate=hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')

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
    print("model parameters: {}K".format(model.count_params()//1000))
    return model



hp_name = "hp_conv1dse"

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
