from pathlib import Path
from read_dataset import MnistDataloader
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Generating the dataset
input_path = Path.cwd().parent / 'data'
training_images_filepath = input_path / 'train-images-idx3-ubyte' /'train-images-idx3-ubyte'
training_labels_filepath = input_path / 'train-labels-idx1-ubyte' /'train-labels-idx1-ubyte'
test_images_filepath = input_path / 't10k-images-idx3-ubyte' /'t10k-images-idx3-ubyte'
test_labels_filepath = input_path / 't10k-labels-idx1-ubyte' /'t10k-labels-idx1-ubyte'

# Load MNIST dataset
mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

# Converting to numpy
x_train = np.array(x_train, dtype=np.float32) / 255.0
x_test  = np.array(x_test,  dtype=np.float32) / 255.0
y_train = np.array(y_train, dtype=np.int64)
y_test  = np.array(y_test,  dtype=np.int64)

# Add channel dim: [N, 28, 28] -> [N, 1, 28, 28]
x_train = np.expand_dims(x_train, 1)
x_test  = np.expand_dims(x_test, 1)

# Convert to tensors
x_train_tensor = torch.tensor(x_train)
y_train_tensor = torch.tensor(y_train)

x_test_tensor = torch.tensor(x_test)
y_test_tensor = torch.tensor(y_test)

# Preparing the dataset
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size = 64, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size = 64, shuffle= True)

# CNN Architecture
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
    
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x))), 2)
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)

        return F.softmax(x)
    
# Training loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 25 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)} / {len(train_loader.dataset)} ({100. * batch_idx / len(train_dataset):.0: .0f}%)]\t{loss.item():.6f}')

def test():
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_fn(output, target).item()
            pred = output.argmax(dim = 1, keepdim = True)
            correct += pred.eq(target.view_as(pred).sum().item())

    test_loss /= len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%\n)')

#Training
for epoch in range(1, 11):
    train(epoch)
    test()