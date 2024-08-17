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

run = "run12"
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
learning_rate_param = 0.000005     
batch_param = 500
dropout_percent = 0.5
epoch_param = 500
initializer_param = 'lecun_normal'
recurrent_dropout_percent = 0.3
n_filters = 32


# define model
model = Sequential()
model.add(ConvLSTM2D(filters=n_filters, recurrent_dropout = recurrent_dropout_percent, kernel_size=(1,12), activation='relu',return_sequences=True, input_shape=(n_steps, 1, n_length, n_features)))
model.add(Flatten())
model.add(Dropout(dropout_percent))
model.add(Dense(100, activation='relu'))
model.add(Dense(n_outputs, activation='softmax'))


# name for output pickle file containing model info
model_name = f'my_best_models/{run}/{run}.pkl'
history_savefile = f"my_best_models/{run}/train_history_{run}.pkl"

# Set up optimiser
adam = Adam(learning_rate=learning_rate_param)
chk = ModelCheckpoint(model_name, monitor='val_accuracy', save_best_only=True, mode='max', verbose=2)
es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=50)
model.compile(loss='sparse_categorical_crossentropy', optimizer=adam, metrics=['accuracy', 'sparse_categorical_accuracy'])

print(model.summary())



# # Python script to train a neural network using Keras library.

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.layers import LSTM
# from tensorflow.keras.layers import Dropout
# from tensorflow.keras.layers import Conv1D
# from tensorflow.keras.layers import MaxPooling2D, MaxPooling3D
# from tensorflow.keras.layers import ConvLSTM2D
# from tensorflow.keras.layers import Flatten
# from tensorflow.keras.layers import TimeDistributed
# from tensorflow.keras.layers import BatchNormalization

# from tensorflow.keras.optimizers import Adam, Nadam
# from tensorflow.keras.models import load_model
# from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
# from tensorflow.keras.regularizers import l1
# from datetime import datetime

# run = "run12"
# import os
# if not os.path.exists(f"my_best_models/{run}"):
#     os.makedirs(f"my_best_models/{run}")


# import pickle
# # load the data
# with open("preproccessed_data.pickle", "rb") as f:
#     data_dic = pickle.load(f)

# train = data_dic['train']
# train_target = data_dic['train_target']
# validation = data_dic['validation']
# validation_target = data_dic['validation_target']
# test = data_dic['test']
# test_target = data_dic['test_target']


# n_steps = 1
# n_length = 1500
# n_features = 1
# n_outputs = 4
# train = train.reshape((train.shape[0], n_steps, 1, n_length, n_features))
# validation = validation.reshape((validation.shape[0], n_steps, 1, n_length, n_features))
# test = test.reshape((test.shape[0], n_steps, 1, n_length, n_features))


# # hyperparameter settings
# learning_rate_param = 0.000005
# batch_param = 250
# epoch_param = 500
# initializer_param = 'lecun_normal'
# pool_size = (1,1,2)
# dropout_percent = 0.4


# # define model
# model = Sequential()
# model.add(ConvLSTM2D(filters=32, padding="same", data_format='channels_first', kernel_initializer=initializer_param,  kernel_size=(1,6), activation='relu',return_sequences=True, input_shape=(n_steps, 1, n_length, n_features)))
# model.add(BatchNormalization())
# model.add(Dropout(dropout_percent))
# model.add(ConvLSTM2D(filters=32, padding="same", data_format='channels_first', kernel_initializer=initializer_param,  kernel_size=(1,6), activation='relu',return_sequences=True))
# model.add(BatchNormalization())
# model.add(MaxPooling3D(pool_size=pool_size, padding='same', data_format='channels_first'))
# model.add(Dropout(dropout_percent))

# model.add(ConvLSTM2D(filters=32, padding="same", data_format='channels_first', kernel_initializer=initializer_param,  kernel_size=(1,6), activation='relu',return_sequences=True))
# model.add(BatchNormalization())
# model.add(Dropout(dropout_percent))
# model.add(ConvLSTM2D(filters=32, padding="same", data_format='channels_first', kernel_initializer=initializer_param,  kernel_size=(1,6), activation='relu',return_sequences=True))
# model.add(BatchNormalization())
# model.add(MaxPooling3D(pool_size=pool_size, padding='same', data_format='channels_first'))
# model.add(Dropout(dropout_percent))

# model.add(ConvLSTM2D(filters=32, padding="same", data_format='channels_first', kernel_initializer=initializer_param,  kernel_size=(1,6), activation='relu',return_sequences=True))
# model.add(BatchNormalization())
# model.add(Dropout(dropout_percent))
# model.add(ConvLSTM2D(filters=32, padding="same", data_format='channels_first', kernel_initializer=initializer_param,  kernel_size=(1,6), activation='relu',return_sequences=True))
# model.add(BatchNormalization())
# model.add(MaxPooling3D(pool_size=pool_size, padding='same', data_format='channels_first'))
# model.add(Dropout(dropout_percent))

# model.add(ConvLSTM2D(filters=32, padding="same", data_format='channels_first', kernel_initializer=initializer_param,  kernel_size=(1,6), activation='relu',return_sequences=True))
# model.add(BatchNormalization())
# model.add(Dropout(dropout_percent))
# model.add(ConvLSTM2D(filters=32, padding="same", data_format='channels_first', kernel_initializer=initializer_param,  kernel_size=(1,6), activation='relu',return_sequences=True))
# model.add(BatchNormalization())
# model.add(MaxPooling3D(pool_size=pool_size, padding='same', data_format='channels_first'))
# model.add(Dropout(dropout_percent))

# model.add(Flatten())
# model.add(Dense(100, activation='relu'))
# model.add(BatchNormalization())
# model.add(Dropout(dropout_percent))
# model.add(Dense(100, activation='relu'))
# model.add(BatchNormalization())
# model.add(Dropout(dropout_percent))
# model.add(Dense(20, activation='relu'))
# model.add(BatchNormalization())
# model.add(Dropout(dropout_percent))
# model.add(Dense(n_outputs, activation='softmax', use_bias=False, kernel_regularizer=l1(0.00025)))


# # name for output pickle file containing model info
# model_name = f'my_best_models/{run}/{run}.pkl'
# history_savefile = f"my_best_models/{run}/train_history_{run}.pkl"

# # Set up optimiser
# adam = Nadam(learning_rate=learning_rate_param)
# chk = ModelCheckpoint(model_name, monitor='val_accuracy', save_best_only=True, mode='max', verbose=2)
# es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=200)
# model.compile(loss='sparse_categorical_crossentropy', optimizer=adam, metrics=['accuracy', 'sparse_categorical_accuracy'])

# print(model.summary())
