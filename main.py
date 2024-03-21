import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from sklearn.model_selection import train_test_split

# Step 1: Data Loading
dataset = ImageFolder(root='jet_events', transform=ToTensor())

# Step 2: Data Preprocessing
# Split the dataset into training and testing sets
train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Step 3: Model Definition
class DiffusionNetwork(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(DiffusionNetwork, self).__init__()
        # Define the layers of the Diffusion Network
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim)
        )

    def forward(self, x):
        # Encode the input
        encoded = self.encoder(x)
        # Decode the encoded representation
        decoded = self.decoder(encoded)
        return decoded

# Step 4: Training
def train(model, dataloader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for data, _ in dataloader:
            optimizer.zero_grad()
            inputs = data.view(data.size(0), -1)  # Flatten the input images
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(dataloader.dataset)}")

# Step 5: Evaluation
def evaluate(model, dataloader):
    model.eval()
    with torch.no_grad():
        for data, _ in dataloader:
            inputs = data.view(data.size(0), -1)  # Flatten the input images
            reconstructed = model(inputs)
            # Calculate evaluation metric (e.g., mean squared error)
            mse = ((inputs - reconstructed) ** 2).mean().item()
            print(f"Mean Squared Error: {mse}")

# Step 6: Visualization
def visualize_comparison(original, reconstructed):
    plt.figure(figsize=(10, 5))
    for i in range(5):  # Visualize 5 samples
        plt.subplot(2, 5, i + 1)
        plt.imshow(original[i].cpu().numpy().reshape(125, 125, 3))
        plt.title('Original')
        plt.axis('off')
        plt.subplot(2, 5, i + 6)
        plt.imshow(reconstructed[i].cpu().numpy().reshape(125, 125, 3))
        plt.title('Reconstructed')
        plt.axis('off')
    plt.show()

# Define hyperparameters
input_dim = 125 * 125 * 3  # Assuming the input images are 125x125 with 3 channels
latent_dim = 64
num_epochs = 10
learning_rate = 0.001

# Initialize the model, criterion, and optimizer
model = DiffusionNetwork(input_dim, latent_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
train(model, train_loader, criterion, optimizer, num_epochs)

# Evaluate the model
evaluate(model, test_loader)

# Reconstruct and visualize some events
data, _ = next(iter(test_loader))
inputs = data.view(data.size(0), -1)  # Flatten the input images
reconstructed_data = model(inputs)
visualize_comparison(data, reconstructed_data)
