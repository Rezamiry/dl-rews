# Python script to train a neural network using Keras library.
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# import pandas
import pickle
import numpy as np
import random
import sys
from datetime import datetime
random.seed(0)
import importlib

run = sys.argv[1]
globals().update(importlib.import_module(f'my_model_configs.{run}').__dict__)


# load model configuration
# from my_model_configs.run3 import model, chk, es, epoch_param, batch_param, model_name, train, train_target, validation, validation_target, test, test_target
from tensorflow.keras.models import load_model


# Train model
history = model.fit(train, train_target, epochs=epoch_param, batch_size=batch_param, callbacks=[chk, es], validation_data=(validation,validation_target))
history_dic = {"accuracy": history.history['accuracy'],
               "val_accuracy": history.history['val_accuracy'],
               "loss": history.history['loss'],
               "val_loss": history.history['val_loss']}
with open(history_savefile, "wb") as f:
    pickle.dump(history_dic, f)


model = load_model(model_name)

# generate test metrics
from sklearn.metrics import accuracy_score
test_preds = model.predict(test)
test_preds = np.argmax(test_preds, axis=1)
print("accuracy: ", accuracy_score(test_target, test_preds))

from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report 

print(classification_report(test_target, test_preds, digits=3))
print("F1 score:",f1_score(test_target, test_preds, average='macro'))
print("Precision: ",precision_score(test_target, test_preds, average="macro"))
print("Recall: ",recall_score(test_target, test_preds, average="macro"))    
print("Confusion matrix: \n",confusion_matrix(test_target, test_preds))


