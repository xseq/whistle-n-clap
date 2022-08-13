# record audio and predict the category

import pyaudio
import wave
import numpy as np
import os
import csv
import logging
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from scipy.io import wavfile
import sys
import struct
import sounddevice as sd

proj_path = os.path.abspath(os.getcwd())
util_path = proj_path + '/util/'
sys.path.insert(0, util_path)

from preprocessing import get_features


os.system('clear')
logging.getLogger("tensorflow").setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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

frame_size = 512
sample_format = pyaudio.paInt16
n_channels = 1
FS = 44100
FILE_NAME = "temp.wav"
CLIP_DURATION = 3   # seconds


audio_obj = pyaudio.PyAudio()  # portaudio interface
print('Recording')

stream = audio_obj.open(format=sample_format,
                channels=n_channels,
                rate=FS,
                frames_per_buffer=frame_size,
                input_device_index=device_idx,
                input=True)


record_frames = []
stream.start_stream()
for p in range(0, int(FS / frame_size * CLIP_DURATION)):
    data = stream.read(frame_size)
    record_frames.append(data)


# Stop and close the stream 
stream.stop_stream()
stream.close()
# Terminate the PortAudio interface
audio_obj.terminate()
print('Finished recording')



samples = []
for x in record_frames:
    count = len(x)/2
    format = "%dh"%(count)
    sample = struct.unpack(format, x)
    samples.extend(sample)
record_frames = np.array(samples)


# # Save the recorded data as a WAV file
# wf = wave.open(FILE_NAME, 'wb')
# wf.setnchannels(n_channels)
# wf.setsampwidth(audio_obj.get_sample_size(sample_format))
# wf.setframerate(FS)
# wf.writeframes(b''.join(record_frames))
# wf.close()


# load model
proj_path = os.path.abspath(os.getcwd())
f_name = proj_path + '/models/cnn_20220802.h5'
model = load_model(f_name)
# model.summary()
print('Model Loaded!')


# _, data = wavfile.read(FILE_NAME)
data = record_frames.astype(np.float32, order='C') / 32768.0
# data = np.array(data)
features = get_features(data, FS)    # shape: (128, 130)
model_input = np.expand_dims(features, 0)
print('input shape: ')
print(model_input.shape)
y_pred = model.predict(model_input)
# print('Prediction: '  + str(y_pred))
y_pred = np.squeeze(y_pred)

THRD = 2
y_max = y_pred.max()
if y_max > THRD:
    event_idx = np.argmax(y_pred)
    event = categories[event_idx, 1]
    print('Event detected: ' + event)
else:
    print('No event detected.')

