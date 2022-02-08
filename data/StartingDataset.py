import os
import torch
from PIL import Image
from PIL import ImageOps
import pandas as pd
import constants
import torchvision


class StartingDataset(torch.utils.data.Dataset):
    """
    Dataset that contains 100000 3x224x224 black images (all zeros).
    """

    def __init__(self, path):
        self.path = path
        self.data = pd.read_csv(constants.DATA + "/train.csv")
        self.data.rename(columns=self.data.iloc[0]).drop(self.data.index[0])
        self.images = self.data.iloc[:, 0]
        self.labels = self.data.iloc[:, 1]
        self.transition = list(set(self.labels))
        self.whales = self.labels.replace(self.transition, list(range(5005)))

    def __getitem__(self, index):
        image = Image.open(constants.DATA + self.path + self.images[index])
        label = self.whales[index]

        image = image.resize((448, 224))
        image = ImageOps.grayscale(image)

        return torchvision.transforms.functional.pil_to_tensor(image), label


    def __len__(self):
        return len(self.labels)
