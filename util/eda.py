# Exploratory Data Analysis

# Do the following to prepare array inputs for the neural network model
# Shuffle wav files
# Split to training and evaluation sets
# Load wav files
# Zero padding
# Extract features
# Saving to a large csv file

import csv
import os
from playsound import playsound


# parameters
N_TRAIN_FILES = 24  # per class
N_EVAL_FILES = 6    # per class


os.system('clear')
proj_path = os.path.abspath(os.getcwd())
wav_path = proj_path + '/data/wav/'


with open(proj_path + '/csv/categories.csv', newline='') as csvfile:
    categories = np.array(list(csv.reader(csvfile)))

n_categories = len(categories)
for p in range(n_categories):
    label = categories[p, 1]
    print('Processing label: ' + label)
    label_folder = wav_path + label
    wav_file_list = os.listdir(label_folder)
    for q in range(38):
        wav_name = label_folder + '/' +  wav_file_list[q]
        print('Playing file #: ' + str(q) + '  ' +wav_file_list[q])
        playsound(wav_name)

