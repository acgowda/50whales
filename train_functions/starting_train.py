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

                tloss, taccuracy = evaluate(train_loader, model, loss_fn, device)
                writer.add_scalar("Loss/train", tloss, epoch + 1)
                writer.add_scalar("Accuracy/train", taccuracy, epoch + 1)

                # TODO:
                # Compute validation loss and accuracy.
                # Log the results to Tensorboard.
                # Don't forget to turn off gradient calculations!
                
                vloss, vaccuracy= evaluate(val_loader, model, loss_fn, device)
                writer.add_scalar("Loss/val", vloss, epoch + 1)
                writer.add_scalar("Accuracy/val", vaccuracy, epoch + 1)
                model.train()

            step += 1

        print('Epoch:', epoch, 'Loss:', loss.item())

    writer.flush()


async def compute_accuracy(outputs, labels):
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


async def evaluate(loader, model, loss_fn, device):
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
            loss += loss_fn(outputs, labels)

    accuracy = correct / total
    
    return loss, accuracy

