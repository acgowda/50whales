import torch
import torch.nn as nn
import torch.nn.functional as F


class StartingNetwork(nn.Module):
    """
    Basic logistic regression on 224x224x3 images.
    """

    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(1, 4, kernel_size = 5, padding = 2)
        self.conv2 = nn.Conv2d(4, 8, kernel_size = 3, padding = 1)
        self.conv3 = nn.Conv2d(8, 16, kernel_size = 3, padding = 1)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(16 * 56 * 28, 5005)
        # self.fc2 = nn.Linear(20020, 10010)
        # self.fc3 = nn.Linear(10010 ,5005)

    def forward(self, x):
        x = x.float()

        #Forward porp
        # (n, 1, 448, 224)
        x = self.conv1(x)
        x = F.relu(x)
        # (n, 4, 448, 224)
        x = self.pool(x)
        # (n, 4, 224, 112)
        x = self.conv2(x)
        x = F.relu(x)
        # (n, 8, 224, 112)
        x = self.pool(x)
        # (n, 8, 112, 56)
        x = self.conv3(x)
        x = F.relu(x)
        # (n, 16, 112, 56)
        x = self.pool(x)
        # (n, 16, 56, 28)
        x = torch.reshape(x, (-1, 16 * 56 * 28))
        # (n, 16 * 56 * 28)
        x = self.fc1(x)
        # (16 * 56 * 28, 5005)


        # x = F.relu(x)
        # (n, 20020)
        # x = self.fc2(x)
        # x = F.relu(x)
        # (n, 10010)
        # x = self.fc3(x)
        # (n, 5005)
        return x
