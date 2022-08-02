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
        _, data = wavfile.read(f_name)
        print('processing' + f_name)
        y_test.append(categories[p, 2])   # a number that stands for the category


