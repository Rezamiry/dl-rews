# Python script to train a neural network using Keras library.

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling2D, MaxPooling3D
from tensorflow.keras.layers import ConvLSTM2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import BatchNormalization

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.regularizers import l1
from tensorflow.keras.layers import Input, Dense, LSTM, Conv1D, Activation, Layer
from tensorflow.keras.layers import AlphaDropout, BatchNormalization, concatenate
from tensorflow.keras.layers import GlobalAveragePooling1D, Reshape, multiply, AveragePooling1D, GlobalMaxPooling1D, GlobalAveragePooling1D
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from tensorflow.keras.regularizers import l1
import numpy as np

from datetime import datetime

run = "run16"
import os
if not os.path.exists(f"my_best_models/{run}"):
    os.makedirs(f"my_best_models/{run}")


import pickle
# load the data
with open("preproccessed_data.pickle", "rb") as f:
    data_dic = pickle.load(f)

train = data_dic['train']
train_target = data_dic['train_target']
validation = data_dic['validation']
validation_target = data_dic['validation_target']
test = data_dic['test']
test_target = data_dic['test_target']


n_steps = 1
n_length = 1500
n_features = 1
n_outputs = 4
# train = train.reshape((train.shape[0], n_steps, 1, n_length, n_features))
# validation = validation.reshape((validation.shape[0], n_steps, 1, n_length, n_features))
# test = test.reshape((test.shape[0], n_steps, 1, n_length, n_features))


# hyperparameter settings
learning_rate_param = 0.000001
batch_param = 1000
dropout_percent = 0.1
epoch_param = 1500
initializer_param = 'lecun_normal'
kernel_size = 6
n_kernels = 128
n_length = 1500
n_outputs = 4
pool_size_param = 6


def squeeze_excite_block(filters,input):                      
    se = GlobalAveragePooling1D()(input)
    se = Reshape((1, filters))(se) 
    se = Dense(filters//16, activation='relu')(se)
    se = Dense(filters, activation='sigmoid')(se)
    se = multiply([input, se])
    return se

def make_model(shape):
        s = Input(shape=shape) 
        x = Conv1D(n_kernels,kernel_size,activation='relu',padding='same')(s)
        x = Conv1D(n_kernels,kernel_size,activation='relu',padding='same')(x)
        x = Conv1D(n_kernels,kernel_size,activation='relu',padding='same')(x)
        x = BatchNormalization()(x)
        x = squeeze_excite_block(n_kernels,x)
        x = AveragePooling1D(pool_size_param)(x)
        x = Dropout(dropout_percent)(x)

        x = Conv1D(n_kernels,kernel_size,activation='relu',padding='same')(x)
        x = Conv1D(n_kernels,kernel_size,activation='relu',padding='same')(x)
        x = Conv1D(n_kernels,kernel_size,activation='relu',padding='same')(x)
        x = BatchNormalization()(x)
        x = squeeze_excite_block(n_kernels,x)
        x = AveragePooling1D(pool_size_param)(x)
        x = Dropout(dropout_percent)(x)

        x = Conv1D(n_kernels,kernel_size,activation='relu',padding='same')(x)
        x = Conv1D(n_kernels,kernel_size,activation='relu',padding='same')(x)
        x = Conv1D(n_kernels,kernel_size,activation='relu',padding='same')(x)
        x = BatchNormalization()(x)
        x = squeeze_excite_block(n_kernels,x)
        x = AveragePooling1D(pool_size_param)(x)     
        x = Dropout(dropout_percent)(x)


        x = Conv1D(n_kernels,kernel_size,activation='relu',padding='same')(x)
        x = Conv1D(n_kernels,kernel_size,activation='relu',padding='same')(x)
        x = Conv1D(n_kernels,kernel_size,activation='relu',padding='same')(x)
        x = BatchNormalization()(x)
        x = squeeze_excite_block(n_kernels,x)
        x = AveragePooling1D(pool_size_param)(x)
        x = Dropout(dropout_percent)(x)


        x = concatenate([GlobalMaxPooling1D()(x),
                                         GlobalAveragePooling1D()(x)])
        x = Dense(100, activation='relu')(x)
        x = Dropout(dropout_percent)(x)
        x = Dense(100, activation='relu')(x)
        x = Dropout(dropout_percent)(x)
        x = Dense(20, activation='relu')(x)
        x = Dropout(dropout_percent)(x)
        x = Dense(n_outputs, activation='softmax')(x)
        return Model(inputs=s, outputs=x)
    
model=make_model(shape=(n_length,1,))


# name for output pickle file containing model info
model_name = f'my_best_models/{run}/{run}.pkl'
history_savefile = f"my_best_models/{run}/train_history_{run}.pkl"

# Set up optimiser
adam = Adam(learning_rate=learning_rate_param)
chk = ModelCheckpoint(model_name, monitor='val_accuracy', save_best_only=True, mode='max', verbose=2)
es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=200)
model.compile(loss='sparse_categorical_crossentropy', optimizer=adam, metrics=['accuracy', 'sparse_categorical_accuracy'])

print(model.summary())
