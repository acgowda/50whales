import torch
import torch.nn as nn
import torch.nn.functional as F


class TransferNetwork(nn.Module):
    """
    Basic logistic regression on 224x224x3 images.
    """

    def __init__(self):
        super().__init__()
        model_a = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        self.model_a = torch.nn.Sequential(*(list(model_a.children())[:-1]))
        self.fc = nn.Linear(512, 5005)

    def forward(self, x):
        with torch.no_grad(): # Freeze weights
            features = self.model_a(x)

        features = torch.reshape(features, (-1, 512))
        
        prediction = self.fc(features)
        
        return prediction