# copying a subset of audio data from the dataset folder to a local folder
# use conda environment spoken_numbers

import numpy as np
import csv
import os


os.system('clear')
path = os.path.abspath(os.getcwd())

with open(path + '/csv/categories.csv', newline='') as csvfile:
    categories = list(csv.reader(csvfile))

print(categories[0][1])
