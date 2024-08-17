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
from tensorflow.keras.layers import Input, Dense, LSTM, Conv1D, Activation, Layer, Permute, MaxPooling1D
from tensorflow.keras.layers import AlphaDropout, BatchNormalization, concatenate
from tensorflow.keras.layers import GlobalAveragePooling1D, Reshape, multiply, AveragePooling1D, GlobalMaxPooling1D, GlobalAveragePooling1D
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from tensorflow.keras.regularizers import l1

from attention import Attention

import numpy as np

from datetime import datetime

run = "run21"
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
train = train.reshape((-1, 1, n_length))
validation = validation.reshape((-1, 1, n_length))
test = test.reshape((-1, 1, n_length))


# hyperparameter settings
learning_rate_param = 0.0001
batch_param = 1000
lstm_dropout = 0.2
cnn_dropout = 0.8
epoch_param = 1500
initializer_param = 'lecun_normal'
kernel_size = 6
n_kernels = 128
n_length = 1500
n_outputs = 4
pool_size_param = 6
lstm_cells = 25
attention_cells = 25
recurrent_dropout_percent = 0.6


ip = Input(shape=(1, n_length))

x = LSTM(lstm_cells, return_sequences=True, recurrent_dropout = recurrent_dropout_percent)(ip)
x = LSTM(lstm_cells, return_sequences=True, recurrent_dropout = recurrent_dropout_percent)(x)
x = Attention(units=attention_cells)(x)
x = Dropout(lstm_dropout)(x)

y = Permute((2, 1))(ip)
y = Conv1D(filters=n_kernels, kernel_size=8, activation='relu', padding='same', kernel_initializer='he_uniform')(y)
y = BatchNormalization()(y)
y = Activation('relu')(y)
y = Dropout(cnn_dropout)(y)

y = Conv1D(filters=2*n_kernels, kernel_size=5, activation='relu', padding='same', kernel_initializer='he_uniform')(y)
y = BatchNormalization()(y)
y = Activation('relu')(y)
y = Dropout(cnn_dropout)(y)

y = Conv1D(filters=4*n_kernels, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_uniform')(y)
y = BatchNormalization()(y)
y = Activation('relu')(y)
y = GlobalAveragePooling1D()(y)

x = concatenate([x, y])
x = Dense(50, activation='relu')(x)
out = Dense(n_outputs, activation='softmax')(x)

model = Model(ip, out)


# name for output pickle file containing model info
model_name = f'my_best_models/{run}/{run}.pkl'
history_savefile = f"my_best_models/{run}/train_history_{run}.pkl"

# Set up optimiser
adam = Adam(learning_rate=learning_rate_param)
chk = ModelCheckpoint(model_name, monitor='val_accuracy', save_best_only=True, mode='max', verbose=2)
es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=200)
model.compile(loss='sparse_categorical_crossentropy', optimizer=adam, metrics=['accuracy', 'sparse_categorical_accuracy'])

print(model.summary())
