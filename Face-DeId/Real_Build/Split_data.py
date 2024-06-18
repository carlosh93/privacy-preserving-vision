import torch
import torchvision.io
import os
from scipy.io.wavfile import read, write

path = '/home/jhon/Desktop/Data_Sets/FACE2FACE_LAB/Processed/'

all_files = []
for path, subdirs, files in os.walk(path):
    sub_files = []
    if len(files) != 0:
        for name in files:
            sub_files.append(os.path.join(path, name))
        all_files.append(sub_files)

video_1, audio_1, info_2 = torchvision.io.read_video(all_files[0][0])
video_2, audio_2, info_2 = torchvision.io.read_video(all_files[0][0])
print(info_2)
