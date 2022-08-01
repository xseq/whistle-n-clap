# zero padding audio files

import numpy as np
import csv
import os
from playsound import playsound


# parameters
N_FILES_TO_PLAY = 38
FS = 44100
DURATION = 5


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
    for q in range(N_FILES_TO_PLAY):
        wav_name = label_folder + '/' +  wav_file_list[q]
        print('Playing file #: ' + str(q) + '  ' +wav_file_list[q])
        playsound(wav_name)
        shutil.copy(src_name, dst_name)
print('Done.')

