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
from tensorflow.keras.layers import Input, Dense, LSTM, Conv1D, Activation, Layer, MaxPooling1D
from tensorflow.keras.layers import AlphaDropout, BatchNormalization, concatenate
from tensorflow.keras.layers import GlobalAveragePooling1D, Reshape, multiply, AveragePooling1D, GlobalMaxPooling1D, GlobalAveragePooling1D
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from tensorflow.keras.regularizers import l1
import numpy as np

from datetime import datetime

run = "urun0"
import os
if not os.path.exists(f"my_best_models/{run}"):
    os.makedirs(f"my_best_models/{run}")


import pickle
with open("preproccessed_data_uncensored.pickle", "rb") as f:
    data_dic = pickle.load(f)


train = data_dic['train']
train_target = data_dic['train_target']
validation = data_dic['validation']
validation_target = data_dic['validation_target']
test = data_dic['test']
test_target = data_dic['test_target']

train = np.array(train)
validation = np.array(validation)
test = np.array(test)

train_target = np.array(train_target)
validation_target = np.array(validation_target)
test_target = np.array(test_target)



# hyperparameter settings
CNN_layers = 2
LSTM_layers = 2  
pool_size_param = 2
learning_rate_param = 0.0005     
batch_param = 1000
dropout_percent = 0.10
filters_param = 50  
mem_cells = 50
mem_cells2 = 10
kernel_size_param = 12
epoch_param = 1500
ts_len = 1500
initializer_param = 'lecun_normal'


model = Sequential()

# add layers
if CNN_layers == 1:
	model.add(Conv1D(filters=filters_param, kernel_size=kernel_size_param, activation='relu', padding='same',input_shape=(ts_len, 1),kernel_initializer = initializer_param))
elif CNN_layers == 2:
	model.add(Conv1D(filters=filters_param, kernel_size=kernel_size_param, activation='relu', padding='same',input_shape=(ts_len, 1)))
	model.add(Conv1D(filters=2*filters_param, kernel_size=kernel_size_param, activation='relu', padding='same'))
	
model.add(Dropout(dropout_percent))
model.add(MaxPooling1D(pool_size=pool_size_param))

if LSTM_layers == 1:
	model.add(LSTM(mem_cells, return_sequences=True, kernel_initializer = initializer_param))
	model.add(Dropout(dropout_percent))
elif LSTM_layers == 2:
	model.add(LSTM(mem_cells, return_sequences=True))
	model.add(LSTM(mem_cells, return_sequences=True))
	model.add(Dropout(dropout_percent))

model.add(LSTM(mem_cells2,kernel_initializer = initializer_param))
model.add(Dropout(dropout_percent))
model.add(Dense(4, activation='softmax',kernel_initializer = initializer_param))


# name for output pickle file containing model info
model_name = f'my_best_models/{run}/{run}.pkl'
history_savefile = f"my_best_models/{run}/train_history_{run}.pkl"

# Set up optimiser
adam = Adam(learning_rate=learning_rate_param)
chk = ModelCheckpoint(model_name, monitor='val_accuracy', save_best_only=True, mode='max', verbose=2)
es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=1000)
model.compile(loss='sparse_categorical_crossentropy', optimizer=adam, metrics=['accuracy', 'sparse_categorical_accuracy'])

print(model.summary())
