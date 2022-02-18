import torch
from PIL import Image
from PIL import ImageOps
import pandas as pd
import constants
from torchvision import transforms

class SiameseDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        self.path = path
        self.data = pd.read_csv(constants.DATA + "/train.csv")
        self.data.rename(columns=self.data.iloc[0]).drop(self.data.index[0])
        self.images = self.data.iloc[:, 0]
        self.labels = self.data.iloc[:, 1]
        self.transition = list(set(self.labels))
        self.whales = self.labels.replace(self.transition, list(range(5005)))

        self.transform = transforms.Compose([transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]),
        ])

    def __getitem__(self, index):
        image = Image.open(constants.DATA + self.path + self.images[index]).convert('RGB') 
        label = self.whales[index]

        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.labels)
