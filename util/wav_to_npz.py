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


# parameters
N_TRAIN_FILES = 24  # per class
N_EVAL_FILES = 6    # per class


# os.system('clear')
proj_path = os.path.abspath(os.getcwd())
wav_path = proj_path + '/data/wav/selected_wav'
npz_path = proj_path + '/data/npz/'
