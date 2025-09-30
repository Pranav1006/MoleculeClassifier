import torch, os, csv
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor, Lambda
import matplotlib.pyplot as plt



# The training data of digits
training_data = datasets.MNIST(
    root="data", # store it in data
    train=True, # this data is to be trained
    download=True, # download this if we haven't already
    transform=ToTensor(), # transform the data into a tensor for training and linear algebra operations.
)

# The testing data of digits (to test our models accuracy)
test_data = datasets.MNIST(
    root="data", # Once again, store it in data
    train=False, # This set isn't for training, it's for testing
    download=True, # Once again, download it if we don't have it already
    transform=ToTensor() # Once again, transform the data into a tensor
)

# Using DataLoader provides an iterable for the model to train with, shuffling the training data and sending it in batches of 64 images
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True) # The training data iterable
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True) # The testing data iterable

# Define the Neural Network
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10)  # 10 classes
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# an instance of the model
model = NeuralNetwork()

# Loss & Optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training Loop
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction & loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss_val = loss.item()
            current = batch * len(X)
            print(f"loss: {loss_val:>7f}  [{current:>5d}/{size:>5d}]")

# Testing Loop
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    accuracy = correct / size
    print(f"Test Error: \n Accuracy: {(100*accuracy):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# Run Training
epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
torch.save(model, "models/model2.pth")