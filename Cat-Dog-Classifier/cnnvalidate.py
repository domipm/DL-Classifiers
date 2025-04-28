import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from PIL import Image

from torch.utils.data       import DataLoader
from torchvision.transforms import v2

import dataloader
import cnnmodel

# Folder path for validation images
valid_dir = "./catdog_data/validation/"

# Directory for writing all the outputs, graphs, and model parameters
output_dir = "./output/"

# Image size
image_size = (128,)*2

# Normalization params. of train data
mean = (0.4854, 0.4515, 0.4143)
sigma = (0.2289, 0.2236, 0.2242)

# Define transform to perform on testing images (no augmentation)
valid_transform = v2.Compose([
                    v2.Resize(image_size),
                    v2.ToImage(),
                    v2.ToDtype(torch.float32, scale=False),
                    v2.Normalize(mean, sigma),
                ])

# Load the validation dataset
valid_dataset = dataloader.CatsDogsDataset(valid_dir, valid_transform)

# Create dataloader
valid_loader = DataLoader(valid_dataset, batch_size = 64, shuffle = False)

# Initiaize CNN model
model = cnnmodel.CNN(valid_dataset[0][0].shape)
# Load pre-trained weights
model.load_state_dict(torch.load(output_dir + "model_params.pt", weights_only=True))
# Set model to evaluation mode
model.eval()

# Loss Function
criterion = nn.CrossEntropyLoss()

# Loss and accuracy
valid_loss = 0
valid_accuracy = 0

# Itrate through all the batches in validation dataloader
for batch, (image, label) in enumerate(valid_loader):

    # Don't compute gradients
    with torch.no_grad():
        # Compute output from model
        output = model(image)
        # Compute loss
        loss = criterion(output, label)
        # Add current batch loss to train loss
        valid_loss += loss.item()
        # Add current batch accuracy to train accuracy
        valid_accuracy += (output.argmax(1) == label).sum().item()/len(output)
        
# Compute average validation loss and accuracy
valid_loss = valid_loss / len(valid_loader)
valid_accuracy = 100 * valid_accuracy / len(valid_loader)

# Print accuracy and loss obtained
print("-"*64)
print("\nValidation Loss: \t", str(valid_loss))
print("Validation Accuracy: \t", str(valid_accuracy) + "%\n")
print("-"*64)

# Show random sub-batch
org_data = dataloader.CatsDogsDataset(valid_dir, v2.Compose([
    v2.Resize(image_size), v2.ToImage(), v2.ToDtype(torch.float32, scale=True)
    ]))
# How many images to generate
n_images = 5
# Setup plotting
_, ax = plt.subplots(nrows=1, ncols=n_images, figsize=(8,3))
# Plot all images and their predicted label
for n in range(n_images):
    # Choose random images
    rand_index = np.random.randint(0, len(valid_dataset))
    # Load transformed images (to compute model's response)
    image, _ = valid_dataset[rand_index]
    # Load original images (only resizing)
    org_image, _ = org_data[rand_index]
    # Plot config
    ax[n].imshow(org_image.permute(1,2,0))
    ax[n].axis("Off")
    # Compute model's output to these images
    out = torch.softmax(model(image.unsqueeze(0)), dim=1)
    # Probability
    prob = out.max()
    # Label prediction
    pred_lab = out.argmax(1)
    # Convert label integer to class name
    if pred_lab == 0: pred_lab = "Dog"
    elif pred_lab == 1: pred_lab = "Cat"
    # Plot title (class and probability)
    ax[n].set_title(pred_lab + "({:.2f}%)".format(prob*100))
# Plot and save figure
plt.savefig(output_dir + "batchvalid.png", dpi=300, bbox_inches="tight")
plt.show()

# Validate model on custom image
image_org = Image.open("./cat/IMG_20210319_130644.jpg")
# Convert it to a tensor and resize
# Define transform to perform on testing images (no augmentation)
image_transform = v2.Compose([
                    v2.Resize(image_size),
                    v2.ToImage(),
                    v2.ToDtype(torch.float32, scale=False),
                ])
# Apply transform
image = image_transform(image_org)
# Compute the model output
output = model(image.unsqueeze(0))
# Probability
prob = out.max()
# Label prediction
pred_lab = output.argmax(1)
# Convert label integer to class name
if pred_lab == 0: pred_lab = "Dog"
elif pred_lab == 1: pred_lab = "Cat"
# Plot image
plt.imshow(image_org)
plt.title(pred_lab + " ({:.2f}%)".format(prob))
plt.axis("Off")
plt.savefig(output_dir + "cat.png", dpi=300, bbox_inches="tight")
plt.show()