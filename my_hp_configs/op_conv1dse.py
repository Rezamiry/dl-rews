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

import optuna

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

classes = list(set(train_target))

def squeeze_excite_block(filters,input):                      
    se = GlobalAveragePooling1D()(input)
    se = Reshape((1, filters))(se) 
    se = Dense(filters//16, activation='relu')(se)
    se = Dense(filters, activation='sigmoid')(se)
    se = multiply([input, se])
    return se

def build_model(trial):
    
    num_layers = trial.suggest_int("num_layers", 2, 6)
    n_filters = trial.suggest_int(f"n_filters", 16, 128, step=8)
    kernel_size = trial.suggest_int(f"kernel_size", 4, 10, step=1)
    pool_size = trial.suggest_int(f"pool_size", 2, 4, step=1)
    kernel_regulizer_l2 = trial.suggest_float('kernel_regulizer_l2', 1e-7, 1e-4, log=True)
    dropout = trial.suggest_float(f'dropout', 0, 0.8, step=0.05)
    dense_dropout = trial.suggest_float(f'dropout_dense', 0, 0.8, step=0.05)
    


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
    
    return model

def batch_generator(X, y, batch_size):
    num_samples = X.shape[0]
    num_batches = num_samples // batch_size
    if num_samples % batch_size:
        num_batches += 1

    while True:  # Loop forever, the generator never ends
        for i in range(num_batches):
            start = i * batch_size
            end = min((i+1) * batch_size, num_samples)
            yield X[start:end], y[start:end]




def objective(trial):
    model = build_model(trial)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print("model parameters: {}K".format(model.count_params()//1000))
    batch_size = trial.suggest_categorical("batch_size", [128, 256, 512, 1024])
    generator = batch_generator(train, train_target, batch_size=batch_size)
    
    for step in range(100):
        num_batches = len(train) // batch_size
        for batch_id in range(num_batches):
            X_batch, y_batch = next(generator)
            # Now you can use X_batch and y_batch to train your model
            model.train_on_batch(X_batch, y_batch)
        
            # Progress bar
            print('\r', 'Training progress: ', '[{0}{1}]'.format('#' * ((step+1) * 50 // num_batches), '.' * (50 - ((step+1) * 50 // num_batches))), f' {((step+1) * 100 // num_batches)}%', end='')

    
        # Calculate the intermediate value by using evaluate instead of score
        loss, intermediate_value = model.evaluate(validation, validation_target, verbose=0)
        trial.report(intermediate_value, step)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.TrialPruned()

    # Calculate the final value using evaluate instead of score
    loss, final_value = model.evaluate(validation, validation_target, verbose=0)
    return final_value  # return accuracy


study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.HyperbandPruner())
study.optimize(objective, n_trials=500)

best_trial = study.best_trial

print("Best trial:")
print(" Value: ", best_trial.value)
print(" Params: ")
for key, value in best_trial.params.items():
    print(f"    {key}: {value}")