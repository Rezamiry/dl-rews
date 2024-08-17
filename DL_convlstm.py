# Python script to train a neural network using Keras library.
run = "run5"
print(f"This is run {run}")
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

import pandas
import pickle
import numpy as np
import random
import sys

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

random.seed(0)

# # keeps track of training metrics
# f1_name = 'training_results_convlstm_{}.txt'.format(str(datetime.now()))
# f2_name = 'training_results_convlstm_{}.csv'.format(str(datetime.now()))

# f_results= open(f1_name, "w")
# f_results2 = open(f2_name, "w")


with open("preproccessed_data.pickle", "rb") as f:
    data_dic = pickle.load(f)

train = data_dic['train']
train_target = data_dic['train_target']
validation = data_dic['validation']
validation_target = data_dic['validation_target']
test = data_dic['test']
test_target = data_dic['test_target']

# reshape to convlstm input
n_steps = 1
n_length = 1500
n_features = 1
n_outputs = 4
train = train.reshape((train.shape[0], n_steps, 1, n_length, n_features))
validation = validation.reshape((validation.shape[0], n_steps, 1, n_length, n_features))
test = test.reshape((test.shape[0], n_steps, 1, n_length, n_features))



# hyperparameter settings
learning_rate_param = 0.0003     
batch_param = 500
dropout_percent = 0.20
epoch_param = 500
initializer_param = 'lecun_normal'


# define model
model = Sequential()
model.add(ConvLSTM2D(filters=128, kernel_size=(1,12), activation='relu',return_sequences=True, input_shape=(n_steps, 1, n_length, n_features)))
model.add(Dropout(dropout_percent))
# model.add(ConvLSTM2D(filters=64, kernel_size=(1,3), activation='relu'))
# model.add(Dropout(0.1))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dropout(dropout_percent))
# model.add(Dense(20, activation='relu'))
model.add(Dense(n_outputs, activation='softmax'))



# name for output pickle file containing model info
model_name = f'my_best_models/{run}/{run}.pkl'

# Set up optimiser
adam = Adam(learning_rate=learning_rate_param)
chk = ModelCheckpoint(model_name, monitor='val_accuracy', save_best_only=True, mode='max', verbose=2)
es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=100)
model.compile(loss='sparse_categorical_crossentropy', optimizer=adam, metrics=['accuracy', 'sparse_categorical_accuracy'])

# Train model
history = model.fit(train, train_target, epochs=epoch_param, batch_size=batch_param, callbacks=[chk, es], validation_data=(validation,validation_target))

model = load_model(model_name)

# generate test metrics
from sklearn.metrics import accuracy_score
test_preds = model.predict(test)
test_preds = np.argmax(test_preds, axis=1)
accuracy_score(test_target, test_preds)

from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report 

print(classification_report(test_target, test_preds, digits=3))
print(history.history['accuracy'])
print(history.history['val_accuracy'])
print(history.history['loss'])
print(history.history['val_loss'])
print("F1 score:",f1_score(test_target, test_preds, average='macro'))
print("Precision: ",precision_score(test_target, test_preds, average="macro"))
print("Recall: ",recall_score(test_target, test_preds, average="macro"))    
print("Confusion matrix: \n",confusion_matrix(test_target, test_preds))