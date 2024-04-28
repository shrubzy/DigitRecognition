from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch

# Constants
ROOT = "./MNIST"
NUM_EPOCHS = 11
BATCH_SIZE = 64
LR = 0.02
MOMENTUM = 0.9
SHOW_PLOT = False  # Flag controlling whether the image is displayed
LOG_FREQUENCY = 1.5  # Controls the frequency of thr progress logs. Larger -> more frequent logs


# Class contains the model architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 258)
        self.fc2 = nn.Linear(258, 85)
        self.fc3 = nn.Linear(85, 10)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc3(x))
        return F.softmax(x, dim=1)


def train(epoch_num):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()

        # Calculate error
        output = model(data)
        loss = loss_fn(output, target)

        # Backpropagation
        loss.backward()
        optimizer.step()

        # Log epoch progress
        if batch_idx % int(BATCH_SIZE / LOG_FREQUENCY) == 0:
            print(f"Epoch {epoch_num + 1} {batch_idx/len(train_loader) * 100:.2f}%")


def test():
    model.eval()

    # Initialise loss and correct values
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            # Model the output and update test loss
            output = model(data)
            test_loss += loss_fn(output, target).item()

            # Make prediction and update correct
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    # Log the test loss and test accuracy
    print(f"Test loss: {test_loss/len(test_loader.dataset) * 100:.2f}%")
    print(f"Test accuracy: {correct / len(test_loader.dataset) * 100:.2f}%")


def predict(filepath):
    # Load and transform image
    image = Image.open(filepath)
    image_data = transform(image)

    # Make prediction
    output = model(image_data.unsqueeze(0))
    pred = output.argmax(dim=1, keepdim=True).item()

    # Displays the image if flag is enabled. This will block code execution in the terminal
    if SHOW_PLOT:
        plt.imshow(image_data.squeeze(0), cmap="gray")
        plt.show()

    # Print prediction
    print(f"Classifier: {pred}\n")


def build():
    print("Building Neural Network...")

    for epoch in range(NUM_EPOCHS):
        train(epoch)
        test()

    print("Done!\n")


# Transform and normalise data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)), ])

# Split training and test sets
train_data = datasets.MNIST(root=ROOT, train=True, transform=transform)
test_data = datasets.MNIST(root=ROOT, train=False, transform=transform)

# Data loaders for training and test sets
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)

# Define the neural network, optimizer and loss function
model = Net()
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
loss_fn = nn.CrossEntropyLoss()

if __name__ == '__main__':
    build()
