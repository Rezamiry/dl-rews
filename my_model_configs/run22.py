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
from tensorflow.keras.layers import Input, Dense, LSTM, Conv1D, Activation, Layer, Permute
from tensorflow.keras.layers import AlphaDropout, BatchNormalization, concatenate
from tensorflow.keras.layers import GlobalAveragePooling1D, Reshape, multiply, AveragePooling1D, GlobalMaxPooling1D, GlobalAveragePooling1D
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from tensorflow.keras.regularizers import l1

from tensorflow import keras
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, concatenate, Flatten, Dense, Dropout
from tensorflow.keras.models import Model


import numpy as np

from datetime import datetime

run = "run22"
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
learning_rate_param = 0.0001
batch_param = 250
dropout_percent = 0.8
epoch_param = 1500
initializer_param = 'lecun_normal'
kernel_size = 6
n_kernels = 16
n_length = 1500
n_outputs = 4
pool_size_param = 6
lstm_cells = 32


def inception_module(inputs, num_filters):
    conv1 = Conv1D(num_filters, kernel_size=1, activation='relu', padding='same')(inputs)

    conv3 = Conv1D(num_filters, kernel_size=3, activation='relu', padding='same')(inputs)

    conv5 = Conv1D(num_filters, kernel_size=5, activation='relu', padding='same')(inputs)

    maxpool = MaxPooling1D(pool_size=3, strides=1, padding='same')(inputs)
    convpool = Conv1D(num_filters, kernel_size=1, activation='relu', padding='same')(maxpool)

    output = concatenate([conv1, conv3, conv5, convpool], axis=2)
    return output

def InceptionTime(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    x = inputs
    x = Conv1D(32, kernel_size=1, activation='relu')(x)

    x = inception_module(x, n_kernels)
    x = inception_module(x, n_kernels)
    x = Dropout(dropout_percent)(x)

    x = inception_module(x, n_kernels*2)
    x = inception_module(x, n_kernels*2)
    x = Dropout(dropout_percent)(x)

    x = inception_module(x, n_kernels*4)
    x = inception_module(x, n_kernels*4)
    x = Dropout(dropout_percent)(x)

    x = inception_module(x, n_kernels*8)
    x = inception_module(x, n_kernels*8)
    # x = Dropout(dropout_percent)(x)
    
    # x = concatenate([GlobalAveragePooling1D()(x), GlobalMaxPooling1D()(x)])

    x = Flatten()(x)
    # x = Dropout(dropout_percent)(x)
    x = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=x)
    return model

model = InceptionTime((n_length, 1), n_outputs)



# name for output pickle file containing model info
model_name = f'my_best_models/{run}/{run}.pkl'
history_savefile = f"my_best_models/{run}/train_history_{run}.pkl"

# Set up optimiser
adam = Adam(learning_rate=learning_rate_param)
chk = ModelCheckpoint(model_name, monitor='val_accuracy', save_best_only=True, mode='max', verbose=2)
es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=200)
model.compile(loss='sparse_categorical_crossentropy', optimizer=adam, metrics=['accuracy', 'sparse_categorical_accuracy'])

print(model.summary())
