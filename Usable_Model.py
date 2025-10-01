import torch
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor
from matplotlib import pyplot as plt

# define the same Nueral Network
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


# load the model
model = NeuralNetwork()
model.load_state_dict(torch.load("models/model1.pth"))

# testing input by input
test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

# display the image
figure = plt.figure(figsize=(8,8))
sample_idx = torch.randint(len(test_data), size=(1,)).item()
img, label = test_data[sample_idx]
plt.title(label)
plt.axis("off")
plt.imshow(img.squeeze(), cmap="gray")
plt.show()

# send the image through the network
model.eval()
with torch.no_grad():
    output = model(img.unsqueeze(0))
    print(f"Model predictions:\n {output}")
    print(f"Best value: {torch.argmax(output)}")
    print(f"Correct: {torch.argmax(output) == label}")