

import numpy as np
import librosa
import csv
import os
from playsound import playsound



# zero padding to the intended length
def zero_padding(data_in, FS_in):
    CLIP_DURATION = 3    # seconds
    data_out = [0] * (CLIP_DURATION * FS_in)
    n_copied_samples = min(len(data_in), len(data_out))  # throw away extra samples
    data_out[:n_copied_samples] = data_in[:n_copied_samples]
    return data_out


# get audio features; using a wrapper of librosa for now
def get_features(data_in, FS_in):
    data_in = zero_padding(data_in, FS_in)
    data_in = np.array(data_in)
    melspectrogram = librosa.feature.melspectrogram(
        y=data_in,
        sr=FS_in,
        n_fft=2048,  # 46 ms
        hop_length=1024,  # 23 ms
        n_mels=128
    )
    return librosa.power_to_db(melspectrogram, ref=np.max)

