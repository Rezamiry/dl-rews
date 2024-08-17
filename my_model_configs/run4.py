# Python script to train a neural network using Keras library.

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import ConvLSTM2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from datetime import datetime

run = "run4"
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


n_length = 1500
n_features = 1
n_outputs = 4


# hyperparameter settings
learning_rate_param = 0.001  
batch_param = 500
dropout_percent = 0.10
epoch_param = 500
initializer_param = 'lecun_normal'
kernel_size = 12
n_kernels = 32


# define model
x = Input(shape=(1,n_length))
x = Conv1D(filters=n_kernels, kernel_size=kernel_size, activation='softplus', padding='valid')(x)
x = Dropout(dropout_percent)(x)
x = Conv1D(filters=n_kernels, kernel_size=2*kernel_size, activation='softplus', padding='valid')(x)
x = Dropout(dropout_percent)(x)
x = Conv1D(filters=n_kernels, kernel_size=4*kernel_size, activation='softplus', padding='valid')(x)
x = Dropout(dropout_percent)(x)
x = Flatten()(x)
x = Dense(100, activation='softplus')(x)
x = Dropout(dropout_percent)(x)
x = Dense(20, activation='softplus')(x)
Y_HAT = Dense(n_outputs, activation="softmax")(x)
model = Model(inputs=x, outputs=Y_HAT)


# name for output pickle file containing model info
model_name = f'my_best_models/{run}/{run}.pkl'

# Set up optimiser
adam = Adam(learning_rate=learning_rate_param)
chk = ModelCheckpoint(model_name, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=100)
model.compile(loss='sparse_categorical_crossentropy', optimizer=adam, metrics=['accuracy', 'sparse_categorical_accuracy'])
