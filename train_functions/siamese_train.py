import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from pytorch_metric_learning import miners, losses, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()
# https://github.com/KevinMusgrave/pytorch-metric-learning/blob/master/examples/notebooks/TripletMarginLossMNIST.ipynb

def train(train_dataset, val_dataset, model, hyperparameters, n_eval, device):
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
    loss_fn = losses.TripletMarginLoss()
    miner = miners.MultiSimilarityMiner()

    accuracy_calculator = AccuracyCalculator(include=("precision_at_1",), k=1)

    # Move the model to the GPU
    model = model.to(device)

    step = 1

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} of {epochs}")

        # Loop over each batch in the dataset
        for batch in tqdm(train_loader):
            # TODO: Backpropagation and gradient descent
            images, labels = batch
            labels = torch.stack(list(labels), dim=0)

            embeddings = model(images)
            hard_pairs = miner(embeddings, labels)

            images = images.to(device)
            labels = labels.to(device)
            #hard_pairs = hard_pairs.to(device)

            loss = loss_fn(embeddings, labels, hard_pairs)
            loss.backward()       # Compute gradients
            optimizer.step()      # Update all the weights with the gradients you just calculated

            optimizer.zero_grad()

            # Periodically evaluate our model + log to Tensorboard
            # if step % n_eval == 0:
            #     model.eval()
            #     # TODO:
            #     # Compute training loss and accuracy.
            #     # Log the results to Tensorboard.

            #     writer.add_scalar("Loss/train", loss.mean().item(), epoch + 1)

            #     # TODO:
            #     # Compute validation loss and accuracy.
            #     # Log the results to Tensorboard.
            #     # Don't forget to turn off gradient calculations!
                
            #     vloss, vaccuracy = evaluate(val_loader, model, loss_fn, device)
            #     writer.add_scalar("Loss/val", vloss, epoch + 1)
            #     writer.add_scalar("Accuracy/val", vaccuracy, epoch + 1)
            #     model.train()

            # step += 1
        
        writer.add_scalar("Loss/train", loss.mean().item(), epoch + 1)
        a = test(train_dataset, val_dataset, model, accuracy_calculator)
        writer.add_scalar("Accuracy/Precision@1", a, epoch + 1)
        
        print('Epoch:', epoch + 1, 'Loss:', loss.item())

    writer.flush()

def test(train_set, test_set, model, accuracy_calculator):
    train_embeddings, train_labels = get_all_embeddings(train_set, model)
    test_embeddings, test_labels = get_all_embeddings(test_set, model)
    train_labels = train_labels.squeeze(1)
    test_labels = test_labels.squeeze(1)
    print("Computing accuracy")
    accuracies = accuracy_calculator.get_accuracy(
        test_embeddings, train_embeddings, test_labels, train_labels, False
    )
    return accuracies["precision_at_1"]

def compute_accuracy(embeddings, labels):
    """
    Computes the accuracy of a model's predictions.

    Example input:
        outputs: [0.7, 0.9, 0.3, 0.2]
        labels:  [1, 1, 0, 1]

    Example output:
        0.75
    """

    n_correct = (embeddings == labels).int().sum()
    n_total = len(embeddings)
    return n_correct / n_total

def get_all_embeddings(dataset, model):
    tester = testers.BaseTester()
    return tester.get_all_embeddings(dataset, model)


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

