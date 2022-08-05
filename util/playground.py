import pyaudio
import wave

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "voice.wav"

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("* recording")

frames = []
plt.figure(1)
for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    buffer = np.frombuffer(stream.read(frame_size),dtype=np.float32)
    # frames.append(data)
    

print("* done recording")

stream.stop_stream()
stream.close()
p.terminate()

# wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
# wf.setnchannels(CHANNELS)
# wf.setsampwidth(p.get_sample_size(FORMAT))
# wf.setframerate(RATE)
# wf.writeframes(b''.join(frames))
# wf.close()


# import numpy as np
# import pyaudio
# from matplotlib import pyplot as plt
# from matplotlib.animation import FuncAnimation
# plt.style.use('bmh')

# SAMPLESIZE = 4096 # number of data points to read at a time
# SAMPLERATE = 44100 # time resolution of the recording device (Hz)

# p = pyaudio.PyAudio() # instantiate PyAudio

# # info = p.get_host_api_info_by_index(0)
# # numdevices = info.get('deviceCount')

# # for i in range(0, numdevices):
# #     if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
# #         print("Input Device id ", i, " - ", p.get_device_info_by_host_api_device_index(0, i).get('name'))

# stream=p.open(format=pyaudio.paInt16,channels=1,rate=SAMPLERATE,input=True,input_device_index=2,
#                frames_per_buffer=SAMPLESIZE) # use default input device to open audio stream

# # set up plotting
# fig = plt.figure()
# ax = plt.axes(xlim=(0, SAMPLESIZE-1), ylim=(-9999, 9999))
# line, = ax.plot([], [], lw=1)

# # x axis data points
# x = np.linspace(0, SAMPLESIZE-1, SAMPLESIZE)

# # methods for animation
# def init():
#     line.set_data([], [])
#     return line,
# def animate(i):
#     y = np.frombuffer(stream.read(SAMPLESIZE), dtype=np.int16)
#     line.set_data(x, y)
#     return line,

# FuncAnimation(fig, animate, init_func=init, frames=200, interval=20, blit=True)

# plt.show()

# # stop and close the audio stream
# stream.stop_stream()
# stream.close()
# p.terminate()