import os
import torch
from PIL import Image
from PIL import ImageOps
import pandas as pd
import constants
import torchvision
from pathlib import Path
import csv


data = pd.read_csv(constants.DATA + "/train.csv")
data.rename(columns=data.iloc[0]).drop(data.index[0])
images = data.iloc[:, 0]
labels = data.iloc[:, 1]

whaledict = {}
for label in labels:
    if label in whaledict:
        whaledict[label] += 1
    else:
        whaledict[label] = 1

filepath = Path(constants.DATA + "/whalecount.csv")
# filepath.parent.mkdir(parents = True, exist_ok = True)
# writer = pd.DataFrame.to_csv(whaledict)
# writer.to_csv(filepath)

with open(filepath, 'w', newline = '') as csvfile:
    fieldnames = ['whale', 'count']

    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    for whale in whaledict:
        writer.writerheader()
        writer.writerow({'whale' : whale, 'count' : whaledict[whale]})