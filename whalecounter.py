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
for i in range(len(labels)):
    label = labels[i]
    image = images[i]
    if label == "new_whale":
        whaledict[image] = 1
    elif label in whaledict:
        whaledict[label] += 1
    else:
        whaledict[label] = 1


anchor = {}
negative = {}

for i in range(len(labels)):
    label = labels[i]
    image = images[i]

    if image in whaledict:
        negative[image] = label
    elif label in whaledict:
        if whaledict[label] > 1:
            anchor[image] = label
        else: 
            negative[image] = label
    

anchorpath = Path(constants.DATA + "/anchorwhales.csv")
negativepath = Path(constants.DATA + "/negativewhales.csv")

with open(anchorpath, 'w', newline = '') as csvfile:
    fieldnames = ['Image', 'Id', 'Augment']

    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for whale in anchor:
        writer.writerow({'Image' : whale, 'Id' : anchor[whale], 'Augment' : 'True'})
        writer.writerow({'Image' : whale, 'Id' : anchor[whale], 'Augment' : 'False'})

with open(negativepath, 'w', newline = '') as csvfile:
    fieldnames = ['Image', 'Id', 'Augment']

    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for whale in negative:
        writer.writerow({'Image' : whale, 'Id' : negative[whale], 'Augment' : 'True'})
        writer.writerow({'Image' : whale, 'Id' : negative[whale], 'Augment' : 'False'})

