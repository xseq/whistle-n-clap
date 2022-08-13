# record audio for a limited period of time
# reference: https://stackoverflow.com/questions/40704026/voice-recording-using-pyaudio

import pyaudio
import wave
import numpy as np
import os
import logging
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import sys
import sounddevice as sd

proj_path = os.path.abspath(os.getcwd())
util_path = proj_path + '/util/'
sys.path.insert(0, util_path)

from preprocessing import get_features


os.system('clear')
logging.getLogger("tensorflow").setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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


# Save the recorded data as a WAV file
wf = wave.open(FILE_NAME, 'wb')
wf.setnchannels(n_channels)
wf.setsampwidth(audio_obj.get_sample_size(sample_format))
wf.setframerate(FS)
wf.writeframes(b''.join(record_frames))
wf.close()
print('Finished recording')

