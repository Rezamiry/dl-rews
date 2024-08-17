# Python script to train a neural network using Keras library.

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import ConvLSTM2D
from tensorflow.keras.layers import Flatten

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from datetime import datetime

run = "run1"
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
train = train.reshape((train.shape[0], n_steps, 1, n_length, n_features))
validation = validation.reshape((validation.shape[0], n_steps, 1, n_length, n_features))
test = test.reshape((test.shape[0], n_steps, 1, n_length, n_features))


# hyperparameter settings
learning_rate_param = 0.001  
batch_param = 256
dropout_percent = 0.10
epoch_param = 500
initializer_param = 'lecun_normal'


model = Sequential()
model.add(ConvLSTM2D(filters=128, kernel_size=(1,12), activation='relu',return_sequences=True, input_shape=(n_steps, 1, n_length, n_features)))
model.add(Dropout(0.5))
model.add(ConvLSTM2D(filters=64, kernel_size=(1,3), activation='relu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(20, activation='relu'))
model.add(Dense(n_outputs, activation='softmax'))


# name for output pickle file containing model info
model_name = f'my_best_models/{run}/{run}.pkl'

# Set up optimiser
adam = Adam(learning_rate=learning_rate_param)
chk = ModelCheckpoint(model_name, monitor='val_accuracy', save_best_only=True, mode='max', verbose=2)
es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=100)
model.compile(loss='sparse_categorical_crossentropy', optimizer=adam, metrics=['accuracy', 'sparse_categorical_accuracy'])
