# Do the following to prepare array inputs for the neural network model
# Shuffle wav files
# Split to training and evaluation sets
# Load wav files
# Zero padding
# Extract features
# Saving to npz file


import numpy as np
import csv
import os
from playsound import playsound
from scipy.io import wavfile


# parameters
N_TRAIN_FILES = 24  # per class
N_TEST_FILES = 6    # per class
FS = 44100

os.system('clear')
proj_path = os.path.abspath(os.getcwd())
wav_path = proj_path + '/data/selected_wav/'
npz_path = proj_path + '/data/npz/'

with open(proj_path + '/csv/categories.csv', newline='') as csvfile:
    categories = np.array(list(csv.reader(csvfile)))

n_categories = len(categories)
n_file_per_label = N_TRAIN_FILES + N_TEST_FILES
x_train = []
x_test = []
y_train = []
y_test = []

for p in range(n_categories):
    label_txt = categories[p, 1]
    print('Processing label: ' + label_txt)
    label_folder = wav_path + label_txt + '/'
    wav_file_list = os.listdir(label_folder)
    for q in range(N_TEST_FILES):
        f_name = label_folder + wav_file_list[q]
        _, data = librosa.Load(f_name)
        #_, data = wavfile.read(f_name)
        print('processing' + f_name)
        x_test.append(get_features())
        y_test.append(categories[p, 2])   # a number that stands for the category


# zero padding to the intended length
def zero_padding(data_in, FS_in):
    CLIP_DURATION = 3    # seconds
    data_out = [0] * (CLIP_DURATION * FS_in)
    n_copied_samples = min(len(data_in), len(data_out))  # throw away extra samples
    data_out[:n_copied_samples] = data_in[:n_copied_samples]
    return data_out


# get audio features; using a wrapper of librosa for now
def get_features(data_in, FS_in):
    melspectrogram = librosa.feature.melspectrogram(
        y=data_in,
        sr=FS_in,
        n_fft=2048,  # 46 ms
        hop_length=1024,  # 23 ms
        n_mels=128
    )
    return librosa.power_to_db(melspectrogram, ref=np.max)

