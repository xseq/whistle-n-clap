# copying a subset of audio data from the dataset folder to a local folder
# use conda environment spoken_numbers

import numpy as np
import csv
import os
import shutil


os.system('clear')
proj_path = os.path.abspath(os.getcwd())
wav_src_path = '/media/xuan/XZ/Dataset/FSD50k/FSD50K.dev_audio/'
wav_dst_path = proj_path + '/data/wav/'

with open(proj_path + '/csv/categories.csv', newline='') as csvfile:
    categories = np.array(list(csv.reader(csvfile)))
with open(proj_path + '/csv/dev.csv', newline='') as csvfile:
    dev_list = np.array(list(csv.reader(csvfile)))

n_dev_data = len(dev_list)
n_wav_files = 0

for p in range(n_dev_data):
    if dev_list[p, 1] in categories: 
        n_wav_files += 1
        src_name = wav_src_path + str(dev_list[p, 0]) + '.wav'
        dst_name = wav_dst_path + str(dev_list[p, 0]) + '.wav'
        print('Copying file: '+ str(dev_list[p, 0]) + '.wav')
        shutil.copy(src_name, dst_name)

print('The number of wav files copied: ' + str(n_wav_files))
