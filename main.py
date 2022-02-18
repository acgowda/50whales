import os

import constants
import torch
#from data.StartingDataset import StartingDataset
from data.TransferDataset import TransferDataset
from data.SiameseDataset import SiameseDataset
#from networks.StartingNetwork import StartingNetwork
#from networks.TransferNetwork import TransferNetwork
from networks.SiameseNetwork import SiameseNetwork
#from train_functions.starting_train import starting_train
from train_functions.siamese_train import train
from PIL import Image


def main():
    # Get command line arguments
    hyperparameters = {"epochs": constants.EPOCHS, "batch_size": constants.BATCH_SIZE}

    # Add GPU support. This line of code might be helpful.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    print("Epochs:", constants.EPOCHS)
    print("Batch size:", constants.BATCH_SIZE)

    # Initalize dataset and model. Then train the model!

    #data = StartingDataset("/train/")
    #data = TransferDataset("/train/")
    data = SiameseDataset("/train/")
    train_size = int(0.7 * len(data))
    test_size = len(data) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(data, [train_size, test_size])
    val_dataset = torch.utils.data.Subset(test_dataset, list(range(1000)))
    # model = StartingNetwork()
    model = SiameseNetwork()
    train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        model=model,
        hyperparameters=hyperparameters,
        n_eval=constants.N_EVAL,
        device = device
    )


if __name__ == "__main__":
    main()
