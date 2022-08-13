# sound event class inference


import os
import logging
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout
from tensorflow.keras.layers import GlobalMaxPooling2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model


os.system('clear')
logging.getLogger("tensorflow").setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


# load model
proj_path = os.path.abspath(os.getcwd())
f_name = proj_path + '/models/cnn_20220802.h5'
model = load_model(f_name)
model.summary()
# weight_file = proj_path + '/models/cnn_weight_20220802.data-00000-of-00001'
# cnn_weights = cnn_model.load_weights(weight_file)
print('Model Loaded!')


# loading data
proj_path = os.path.abspath(os.getcwd())
npz_path = proj_path + '/data/npz/'
npz_file_name = npz_path + 'data_20220802_2037.npz'
data = np.load(npz_file_name)
x_test = data['x_test']
y_test = data['y_test']


# Evaluation
print('Start evaluation: ')
model.evaluate(x_test, y_test)


