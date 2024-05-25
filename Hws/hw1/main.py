import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os 
import cv2 
import numpy as np
from torchvision import transforms
from PIL import Image, ImageOps 

# Set the path to the image folder
image_folder_path = "./image"

# Get a list of all the image file names in the folder
image_file_names = os.listdir(image_folder_path)

# Initialize an empty list to hold the images
images = []

# Loop through each image file name
for index, image_file_name in enumerate(image_file_names):
    image = Image.open(os.path.join(image_folder_path, image_file_name))
    image = ImageOps.grayscale(image)
    # Append the image to the list of images
    transform = transforms.Compose([
    transforms.Resize((8, 8)),  # Specify the new width and height
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
    ])
    images.append(transform(image))

# Convert the list of images to a numpy array
train = torch.squeeze(torch.stack(images))

# Convert the numpy array to a torch tensor with dtype of float32
test = torch.tensor([0, 1, 2])

# Define the MLP model
class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
            nn.Softmax(dim = 1)
        )

    def forward(self, x):
        return self.layers(x)

# Initialize the model
model = MLP()

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
# Train the model for 100 epochs
losses = []
for epoch in range(100):
    # Zero the gradients
    optimizer.zero_grad()

    # Forward pass
    outputs = model(train)
    loss = criterion(outputs, test)

    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    # Save the loss for plotting
    losses.append(loss.item())

    # Print the loss for every 10 epochs
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# Plot the loss for each epoch
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()






