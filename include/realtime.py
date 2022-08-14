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
FRAME_SIZE = 32768    # samples
sample_format = pyaudio.paInt16
N_CHANNELS = 1
FS = 44100
CLIP_DURATION = 3   # seconds
CONFIDENCE_THRD = 5
MAX_DURATION = 30   # seconds


# sound event categories
proj_path = os.path.abspath(os.getcwd())
with open(proj_path + '/csv/categories.csv', newline='') as csvfile:
    categories = np.array(list(csv.reader(csvfile)))
n_categories = len(categories)


# load model
proj_path = os.path.abspath(os.getcwd())
f_name = proj_path + '/models/cnn_20220802.h5'
model = load_model(f_name)


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
clip = [0.0] * FS * CLIP_DURATION   # 3 seconds of audio samples
clip = np.array(clip)
clip = clip.astype(np.float32, order='C')


# audio streaming and recording
stream.start_stream()
for p in range(0, int(FS / FRAME_SIZE * MAX_DURATION)):
    data = stream.read(FRAME_SIZE)
    count = len(data)/2
    format = "%dh"%(count)
    frame_data = struct.unpack(format, data)
    frame_data = np.array(frame_data)
    frame_float = frame_data.astype(np.float32, order='C') / 32768.0

    clip = np.append(clip[len(frame_data):], frame_float)
    features = get_features(clip, FS)

    model_input = np.expand_dims(features, 0)
    # inference
    y_pred = model.predict(model_input)
    y_pred = np.squeeze(y_pred)
    y_max = y_pred.max()
    if y_max > CONFIDENCE_THRD:
        event_idx = np.argmax(y_pred)
        event = categories[event_idx, 1]
        print('Event detected: ' + event)
    else:
        print('No event detected.')


# stop and close the stream 
stream.stop_stream()
stream.close()
audio_obj.terminate()
print('Finished recording')


