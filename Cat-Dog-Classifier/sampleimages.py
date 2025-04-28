import random
import numpy as np
import matplotlib.pyplot as plt

import torch

from torch.utils.data import DataLoader

from torchvision.utils import make_grid
from torchvision.transforms import v2

from datetime import datetime

import dataloader
import cnnmodel

# Folder paths for testing and training images
data_dir = "./catdog_data/train/"

# Path where to save visualizations
save_path = "./samples/"

# Seed for random numbers (based on current time, change to set fixed seed)
rand_seed = datetime.now().timestamp()
# Set seed for image selector
random.seed(rand_seed)
# Set seed for pytorch
torch.manual_seed(rand_seed)

# Transform parameters
image_size = (128,)*2
p_hflip = 0.5
p_grayscale = 0.25
p_invert = 0.15
degrees = 35

# Define transform to perform on images (augmentation)
image_transform = v2.Compose([
                    v2.ToImage(),                               # Convert to image object
                    v2.ToDtype(torch.float32, scale=True),      # Convert to tensor
                    v2.Resize(image_size),                      # Resize images to same size
                    #v2.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),  # Normalize images
                    #v2.RandomHorizontalFlip(p_hflip),
                    #v2.GaussianNoise(mean=np.random.uniform(0,0.05),
                    #                 sigma=np.random.uniform(0,0.05)),
                    #v2.ColorJitter(),
                    #v2.RandomRotation(degrees),
                    #v2.RandomGrayscale(p_grayscale),
                    #v2.RandomAdjustSharpness(sharpness_factor=np.random.uniform(0,2)),
                    #v2.RandomInvert(p_invert),
                    #v2.RandAugment(num_ops=3, magnitude=5)
                ])

# Initialize datasets
dataset = dataloader.CatsDogsDataset(data_dir, image_transform)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

# Show random sample batch from dataloader
for img, _ in loader:
    _, ax = plt.subplots()
    ax.set_title("Sample training batch")
    ax.axis("Off")
    ax.imshow(make_grid(img, nrow=4).permute(1,2,0), vmin=0, vmax=1)
    break
plt.savefig(save_path + "sample_batch.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()