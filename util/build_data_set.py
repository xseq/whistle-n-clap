# copying a subset of FSD50k audio data from the dataset folder to a local folder
# use conda environment spoken_numbers

import numpy as np
import csv
import os
import shutil


os.system('clear')
proj_path = os.path.abspath(os.getcwd())

# parameters
MAX_WAV_FILE_SIZE = 500000

# change the following to the source path name
wav_src_path = '.../Dataset/FSD50k/FSD50K.dev_audio/'
wav_dst_path = proj_path + '/data/wav/'

with open(proj_path + '/csv/categories.csv', newline='') as csvfile:
    categories = np.array(list(csv.reader(csvfile)))
with open(proj_path + '/csv/dev.csv', newline='') as csvfile:
    dev_list = np.array(list(csv.reader(csvfile)))

n_categories = len(categories)
n_dev_data = len(dev_list)
file_count = [0] * n_categories

# moving files
for p in range(1, n_dev_data):
    category_name = dev_list[p, 1]
    src_name = wav_src_path + str(dev_list[p, 0]) + '.wav'
    file_size = os.path.getsize(src_name)
    if file_size < MAX_WAV_FILE_SIZE:
        for q in range(n_categories):
            if category_name == categories[q, 0]:
                file_count[q] += 1
                dst_name = wav_dst_path + str(dev_list[p, 0]) + '.wav'
                print('Copying file of category ' + 
                    categories[q, 1] + ': ' + str(dev_list[p, 0]) + '.wav')
                shutil.copy(src_name, dst_name)

# display results
print(' ')
print('Task complete.')
for p in range(n_categories):
    print(categories[p, 1] + ': ' + str(file_count[p]))
print('The number of wav files copied: ' + str(sum(file_count)))

# results
# Chink and Clink: 83
# Clapping Hands: 293
# Dropping Coins: 289
# Coughing: 168
# Opening Drawer: 66
# Snapping Fingers: 126
# Jangling Keys: 86
# Knocking Doors: 35
# Laughing: 516
# Walking: 150
# The number of wav files copied: 1812

