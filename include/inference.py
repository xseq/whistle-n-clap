# Real time sound event detection

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
import struct
import logging
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from scipy.io import wavfile
import sounddevice as sd
import librosa.display


os.system('clear')
logging.getLogger("tensorflow").setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# parameters
FRAME_SIZE = 32768    # samples, 32768 does not cause severe overflow
sample_format = pyaudio.paInt16
N_CHANNELS = 1
FS = 44100
CLIP_DURATION = 3   # seconds
CONFIDENCE_THRD = 5
MAX_DURATION = 60   # seconds


# sound event categories
proj_path = os.path.abspath(os.getcwd())
with open(proj_path + '/csv/categories.csv', newline='') as csvfile:
    categories = np.array(list(csv.reader(csvfile)))
n_categories = len(categories)


# display introduction
print('Make the following sounds close to the microphone, and they would be detected:')
print(categories[:, 1])
print(' ')


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
print('Started.')
stream = audio_obj.open(format=sample_format,
                channels=N_CHANNELS,
                rate=FS,
                frames_per_buffer=FRAME_SIZE,
                input_device_index=device_idx,
                input=True)
clip = [0.0] * FS * CLIP_DURATION   # 3 seconds of audio samples
clip = np.array(clip)
clip = clip.astype(np.float32, order='C')


# initializing plotting
plt.ion()
fig, (ax1, ax2) = plt.subplots(2)
fig = plt.gcf()
fig.set_size_inches(5, 11, forward=True)  # inches
plt.show()


# audio streaming and recording
stream.start_stream()
for p in range(0, int(FS / FRAME_SIZE * MAX_DURATION)):
    # reading audio buffer and formatting data
    data = stream.read(FRAME_SIZE, 
            exception_on_overflow = False) # ignore overflow
    count = len(data)/2
    format = "%dh"%(count)
    frame_data = struct.unpack(format, data)
    frame_data = np.array(frame_data)
    frame_float = frame_data.astype(np.float32, order='C') / 32768.0

    # circular buffer
    # TODO: improve efficiency
    clip = np.append(clip[len(frame_data):], frame_float)

    # preprocessing
    features = get_features(clip, FS)
    model_input = np.expand_dims(features, 0)

    # plotting
    ax1.clear() 
    ax1.axis([0, FS * CLIP_DURATION, -0.3, 0.3])
    ax1.plot(clip)
    ax2.clear() 
    flip_features = np.flip(features, axis=0)
    plt.imshow(flip_features)
    plt.pause(0.001)

    # inference
    y_pred = model.predict(model_input)
    y_pred = np.squeeze(y_pred)
    y_max = y_pred.max()
    if y_max > CONFIDENCE_THRD:
        event_idx = np.argmax(y_pred)
        event = categories[event_idx, 1]
        fig.suptitle('Event detected: ' + event, fontsize=20)
    else:
        fig.suptitle('No event detected.', fontsize=20)


# stop and close the stream 
stream.stop_stream()
stream.close()
audio_obj.terminate()
print('Finished.')


