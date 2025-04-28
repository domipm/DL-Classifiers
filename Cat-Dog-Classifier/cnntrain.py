import random
import os
import alive_progress
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import torch
import torch.nn as nn
import torch.optim as optim

from PIL import Image
from datetime import datetime

from torch.utils.data import DataLoader
from torchvision.transforms import v2

import dataloader   # Custom dataloader
import cnnmodel     # Convolutional neural network model

# Folder paths for testing and training images
train_dir = "./catdog_data/train/"
test_dir = "./catdog_data/test/"

# Directory for writing all the outputs, graphs, and model parameters
output_dir = "./output/"

# Seed for random numbers (based on current time, change to set fixed seed)
rand_seed = datetime.now().timestamp()
# Set seed for image selector
random.seed(rand_seed)
# Set seed for pytorch
torch.manual_seed(rand_seed)

# Transform parameters
image_size = (128,)*2

# Batchsize to load from data
batchsize = 64

# Training Hyperparameters
epochs = 20                 
learning_rate = 0.005   
weigh_decay = 0.0005       

# Compute mean and standard deviation for training dataset
print("\nNormalizing...", end="\r")
# Initialize to zero
mean = sigma = 0
# For each image in subfolders
for image in [ file for class_type in [f for f in os.listdir(train_dir) if not f.startswith('.')] for file in os.listdir(os.path.join(train_dir, class_type)) ]:
    # Get path to image
    image_path = os.path.join(train_dir, image.split(".")[0] + "s", image)
    # Open image and transform to tensor
    img = v2.Compose([v2.ToImage(), v2.ToDtype(dtype=torch.float32, scale=True)])(Image.open(image_path))
    # Compute the values
    mean += torch.mean(img, dim=(1,2))
    sigma += torch.std(img, dim=(1,2))
# Normalize with total dataset length
train_len = len([ file for class_type in [f for f in os.listdir(train_dir) if not f.startswith('.')] for file in os.listdir(os.path.join(train_dir, class_type)) ])
mean = mean / train_len
sigma = sigma / train_len
print(mean, sigma)
# Print information
print("Normalizing complete!\n")

train_transform = v2.Compose([
    v2.Resize(image_size),
    #v2.Normalize((970.7758, 920.9586, 828.5756), (457.7354, 447.1332, 448.4109)),
    #v2.RandAugment(num_ops=5, magnitude=10)
    v2.RandomRotation(degrees=45),
    v2.RandomGrayscale(p=0.25),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomInvert(p=0.25),
    v2.RandomPerspective(distortion_scale=0.25, p=0.15),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=False),
    v2.Normalize(mean, sigma),
])

# Define transform to perform on testing images (no augmentation)
test_transform = v2.Compose([
                    v2.Resize(image_size),
                    v2.ToImage(),
                    v2.ToDtype(torch.float32, scale=False),
                    v2.Normalize(mean, sigma),
])

# Initialize datasets
train_dataset = dataloader.CatsDogsDataset(train_dir, train_transform)
test_dataset = dataloader.CatsDogsDataset(test_dir, test_transform)

# Load datasets into pytorch
train_loader = DataLoader(train_dataset, batch_size = batchsize, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size = batchsize, shuffle = False)

# Initiaize CNN model
model = cnnmodel.CNN(train_dataset[0][0].shape)

# Loss Function
criterion = nn.CrossEntropyLoss()
# Optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weigh_decay)
# Learning rate scheduler
scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=0.001, total_iters=15)

# Arrays for storing loss evolution over epochs
train_loss_epoch = []
test_loss_epoch = []
# Arrays for storing accuracy over epochs
train_accuracy_epoch = []
test_accuracy_epoch = []

# Run over multiple epochs
for epoch in range(epochs):

    # Reset train and test losses
    train_loss = 0.0
    test_loss = 0.0
    # Reset train and test accuracy
    train_accuracy = 0.0
    test_accuracy = 0.0

    # Set model to train mode
    model.train()

    # Bar for training progress visualization
    with alive_progress.alive_bar(total=len(train_loader),
                   max_cols=64,
                   title="Epoch {}/{}".format(epoch+1, epochs),
                   bar="classic", 
                   spinner=None, 
                   monitor="Batch {count}/{total}", 
                   elapsed="[{elapsed}]",
                   elapsed_end="[{elapsed}]",
                   stats=None) as bar:
        
        # Run over all batches
        for batch, (image, label) in enumerate(train_loader):
            # Reset gradients
            optimizer.zero_grad()
            # Output from model
            output = model(image)
            # Compute loss
            loss = criterion(output, label)
            # Backpropagation
            loss.backward()
            # Perform one step of optimizer
            optimizer.step()
            # Add current batch loss to train loss
            train_loss += loss.item()
            # Add current batch accuracy to train accuracy
            train_accuracy += (output.argmax(dim=1) == label).sum().item()/len(output)
            # Update progress bar
            bar()

    # Compute average training loss for current epoch
    avg_train_loss = train_loss / len(train_loader)
    train_loss_epoch.append(avg_train_loss)  
    # Compute averate training accuracy for current epoch
    avg_train_accuracy = 100 * train_accuracy / len(train_loader)
    train_accuracy_epoch.append(avg_train_accuracy)

    # Print out average training loss for each epoch 
    print('- Avg. Train Loss: {:.6f}\t Avg. Train Accuracy {:.6f}'.format(avg_train_loss, avg_train_accuracy)) 

    # Set model to evaluation mode
    model.eval()

    # Ensure no gradient is computed
    with torch.no_grad():

        for batch, (image, label) in enumerate(test_loader):
            # Output from model
            output = model(image)
            # Compute loss
            loss = criterion(output, label)
            # Add current batch loss to test loss
            test_loss += loss.item()
            # Add current batch accuracy to test accuracy
            test_accuracy += (output.argmax(dim=1) == label).sum().item()/len(output)

    # Compute average test loss for current epoch
    avg_test_loss = test_loss / len(test_loader)
    test_loss_epoch.append(avg_test_loss) 
    # Compute averate training accuracy for current epoch
    avg_test_accuracy = 100 * test_accuracy / len(test_loader)
    test_accuracy_epoch.append(avg_test_accuracy)

    # Print out average testing loss for each epoch 
    print('- Avg. Test  Loss: {:.6f}\t Avg. Test  Accuracy {:.6f}'.format(avg_test_loss, avg_test_accuracy), end="\n\n") 

    scheduler.step()

print("-"*64)

# After training the model, save the parameters
torch.save(model.state_dict(), output_dir + "model_params.pt")

# Write the training and testing loss and accuracy to file !
with open(output_dir + "output_logs.txt", "w") as f:
    for epoch in range(epochs):
        f.write("{}\t{}\t{}\t{}\n".format(train_loss_epoch[epoch], test_loss_epoch[epoch], train_accuracy_epoch[epoch], test_accuracy_epoch[epoch]))
f.close()

# Plot training and testing loss over time
fig, ax = plt.subplots()

ax.plot(np.arange(epochs), train_loss_epoch, label="Train loss")
ax.plot(np.arange(epochs), test_loss_epoch, label="Test loss")

ax.set_title("Train/Test Loss")
ax.set_xlabel("Epochs")
ax.set_ylabel("Average epoch loss")
ax.xaxis.set_major_locator(MaxNLocator(integer=True))

plt.legend()
plt.savefig(output_dir + "loss.png", dpi=300)
plt.show()
plt.close()

# Plot training and testing accuracy over time
fig, ax = plt.subplots()

ax.plot(np.arange(epochs), train_accuracy_epoch, label="Train accuracy")
ax.plot(np.arange(epochs), test_accuracy_epoch, label="Test accuracy")

ax.set_title("Train/Test Accuracy")
ax.set_xlabel("Epochs")
ax.set_ylabel("Average epoch accuracy [%]")
ax.xaxis.set_major_locator(MaxNLocator(integer=True))

plt.legend()
plt.savefig(output_dir + "accuracy.png", dpi=300)
plt.show()
plt.close()