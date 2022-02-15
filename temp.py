#   temp run file to debug tejas's laptop

#   constants
EPOCHS = 1
BATCH_SIZE = 16
N_EVAL = 1
DATA = "../50whales/sauce"

#   data processing
import torch
from PIL import Image
from PIL import ImageOps
import pandas as pd
import torchvision
import numpy as np


class StartingDataset(torch.utils.data.Dataset):
    """
    Dataset that contains 100000 3x224x224 black images (all zeros).
    """

    def __init__(self, path):
        self.path = path
        self.data = pd.read_csv(DATA + "/train.csv")
        self.data.rename(columns=self.data.iloc[0]).drop(self.data.index[0])
        self.images = self.data.iloc[:, 0]
        self.labels = self.data.iloc[:, 1]
        self.transition = list(set(self.labels))
        self.whales = self.labels.replace(self.transition, list(range(5005)))

    def __getitem__(self, index):
        image = Image.open(DATA + self.path + self.images[index])
        label = self.whales[index]

        image = image.resize((448, 224))
        image = ImageOps.grayscale(image)

        #return torchvision.transforms.functional.pil_to_tensor(image), label
        image = torchvision.transforms.ToTensor()(np.array(image))
        return image, label


    def __len__(self):
        return len(self.labels)

#   neural network
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
        # (n, 8 * 112 * 56)
        x = self.fc1(x)
        # x = F.relu(x)
        # (n, 20020)
        # x = self.fc2(x)
        # x = F.relu(x)
        # (n, 10010)
        # x = self.fc3(x)
        # (n, 5005)
        return x

#   train func + tensorboard
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

def starting_train(train_dataset, val_dataset, model, hyperparameters, n_eval, device):
    """
    Trains and evaluates a model.

    Args:
        train_dataset:   PyTorch dataset containing training data.
        val_dataset:     PyTorch dataset containing validation data.
        model:           PyTorch model to be trained.
        hyperparameters: Dictionary containing hyperparameters.
        n_eval:          Interval at which we evaluate our model.
    """

    # Get keyword arguments
    batch_size, epochs = hyperparameters["batch_size"], hyperparameters["epochs"]

    # Initialize dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True
    )

    # Initalize optimizer (for gradient descent) and loss function
    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss()

    # Move the model to the GPU
    model = model.to(device)

    step = 1

    # tb = SummaryWriter()
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} of {epochs}")

        # Loop over each batch in the dataset
        for batch in tqdm(train_loader):
            # TODO: Backpropagation and gradient descent
            images, labels = batch
            labels = torch.stack(list(labels), dim=0)

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = loss_fn(outputs, labels)
            loss.backward()       # Compute gradients
            optimizer.step()      # Update all the weights with the gradients you just calculated
            optimizer.zero_grad()

            # Periodically evaluate our model + log to Tensorboard
            if step % n_eval == 0:
                model.eval()
                # TODO:
                # Compute training loss and accuracy.
                # Log the results to Tensorboard.

                writer.add_scalar("Loss/train", loss.mean().item(), epoch + 1)

                # TODO:
                # Compute validation loss and accuracy.
                # Log the results to Tensorboard.
                # Don't forget to turn off gradient calculations!
                
                vloss, vaccuracy = evaluate(val_loader, model, loss_fn, device)
                writer.add_scalar("Loss/val", vloss, epoch + 1)
                writer.add_scalar("Accuracy/val", vaccuracy, epoch + 1)
                model.train()

            step += 1

        print('Epoch:', epoch + 1, 'Loss:', loss.item())

    writer.flush()

#   accuracy compute

def compute_accuracy(outputs, labels):
    """
    Computes the accuracy of a model's predictions.

    Example input:
        outputs: [0.7, 0.9, 0.3, 0.2]
        labels:  [1, 1, 0, 1]

    Example output:
        0.75
    """

    n_correct = (outputs == labels).int().sum()
    n_total = len(outputs)
    return n_correct / n_total

#   validation eval function

def evaluate(loader, model, loss_fn, device):
    """
    Computes the loss and accuracy of a model on the validation dataset.
    """
    correct = 0
    total = 0
    loss = 0
    with torch.no_grad(): # IMPORTANT: turn off gradient computations
        for batch in loader:
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)

            # labels == predictions does an elementwise comparison
            # e.g.                labels = [1, 2, 3, 4]
            #                predictions = [1, 4, 3, 3]
            #      labels == predictions = [1, 0, 1, 0]  (where 1 is true, 0 is false)
            # So the number of correct predictions is the sum of (labels == predictions)
            correct += (labels == predictions).int().sum()
            total += len(predictions)
            loss += loss_fn(outputs, labels).mean().item()

    accuracy = correct / total
    
    return loss/len(loader), accuracy

#   main
import torch
from PIL import Image

def main():
    # Get command line arguments
    hyperparameters = {"epochs": EPOCHS, "batch_size": BATCH_SIZE}

    # Add GPU support. This line of code might be helpful.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")

    print("Epochs:", EPOCHS)
    print("Batch size:", BATCH_SIZE)

    # Initalize dataset and model. Then train the model!
    data = StartingDataset("/train/")
    train_size = int(0.7 * len(data))
    test_size = len(data) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(data, [train_size, test_size])
    model = StartingNetwork()
    starting_train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        model=model,
        hyperparameters=hyperparameters,
        n_eval=N_EVAL,
        device = device
    )


if __name__ == "__main__":
    main()