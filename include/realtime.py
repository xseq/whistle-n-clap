import pyaudio
import wave
import numpy as np
import os
import logging
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import sys


proj_path = os.path.abspath(os.getcwd())
util_path = proj_path + '/util/'
sys.path.insert(0, util_path)


from preprocessing import get_features


os.system('clear')
logging.getLogger("tensorflow").setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

frame_size = 1024
sample_format = pyaudio.paInt16
n_channels = 1
FS = 44100
filename = "output.wav"
CLIP_DURATION = 3   # seconds
MAX_DURATION = 60   # seconds


# load model
proj_path = os.path.abspath(os.getcwd())
f_name = proj_path + '/models/cnn_20220802.h5'
model = load_model(f_name)
model.summary()
print('Model Loaded!')


audio_obj = pyaudio.PyAudio()  # portaudio interface
print('Recording')

stream = audio_obj.open(format=sample_format,
                channels=1,
                rate=FS,
                frames_per_buffer=frame_size,
                input_device_index=2,
                input=True)


n_buffer_samples = MAX_DURATION * FS
buffer = [0.0] * n_buffer_samples  # Initialize queue


plt.figure(1)
# Store data in chunks for 3 seconds
for p in range(0, int(FS / frame_size * MAX_DURATION)):
    frame_data = np.frombuffer(stream.read(frame_size),dtype=np.float32)
    # frame = np.array(stream.read(frame_size))
    # frame_data = frame.astype(np.float32, order='C') / 32768.0
    buffer = np.append(buffer[len(frame_data):], frame_data)
    features = get_features(buffer, FS)
    y_pred = model.predict(features)
    print('Prediction: '  + str(y_pred))



# Stop and close the stream 
stream.stop_stream()
stream.close()
# Terminate the PortAudio interface
p.terminate()

print('Finished recording')

# # Save the recorded data as a WAV file
# wf = wave.open(filename, 'wb')
# wf.setnchannels(n_channels)
# wf.setsampwidth(p.get_sample_size(sample_format))
# wf.setframerate(FS)
# wf.writeframes(b''.join(frames))
# wf.close()
