# record 3s audio and predict the category without saving audio

import sys
import os

proj_path = os.path.abspath(os.getcwd())
util_path = proj_path + '/util/'
sys.path.insert(0, util_path)

from preprocessing import get_features
import pyaudio
import wave
import numpy as np
import csv
import logging
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from scipy.io import wavfile
import struct
import sounddevice as sd

os.system('clear')
logging.getLogger("tensorflow").setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# parameters
FRAME_SIZE = 512
sample_format = pyaudio.paInt16
N_CHANNELS = 1
FS = 44100
CLIP_DURATION = 3   # seconds
CONFIDENCE_THRD = 2


# sound event categories
proj_path = os.path.abspath(os.getcwd())
with open(proj_path + '/csv/categories.csv', newline='') as csvfile:
    categories = np.array(list(csv.reader(csvfile)))
n_categories = len(categories)


# load the target input device
device_list = sd.query_devices()
device_name = 'Sennheiser'
device_idx = []
for p in range(len(device_list)):
    if device_name in device_list[p]['name']:
        device_idx = p


# initialize recorder
audio_obj = pyaudio.PyAudio()  # portaudio interface
print('Recording')
stream = audio_obj.open(format=sample_format,
                channels=N_CHANNELS,
                rate=FS,
                frames_per_buffer=FRAME_SIZE,
                input_device_index=device_idx,
                input=True)


# audio streaming and recording
frames = []
stream.start_stream()
for p in range(0, int(FS / FRAME_SIZE * CLIP_DURATION)):
    data = stream.read(FRAME_SIZE)
    count = len(data)/2
    format = "%dh"%(count)
    frame_data = struct.unpack(format, data)
    frames.extend(frame_data)
frames = np.array(frames)


# sStop and close the stream 
stream.stop_stream()
stream.close()
audio_obj.terminate()
print('Finished recording')


# load model
proj_path = os.path.abspath(os.getcwd())
f_name = proj_path + '/models/cnn_20220802.h5'
model = load_model(f_name)


# preprocessing
frames_float = frames.astype(np.float32, order='C') / 32768.0
features = get_features(frames_float, FS)    # shape: (128, 130)
model_input = np.expand_dims(features, 0)


# inference
y_pred = model.predict(model_input)
y_pred = np.squeeze(y_pred)
CONFIDENCE_THRD = 2
y_max = y_pred.max()
if y_max > CONFIDENCE_THRD:
    event_idx = np.argmax(y_pred)
    event = categories[event_idx, 1]
    print('Event detected: ' + event)
else:
    print('No event detected.')

