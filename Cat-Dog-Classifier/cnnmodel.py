import torchsummary
import torch.nn as nn

class CNN(nn.Module):

    # Initialization function with definitions for all layers
    def __init__(self, in_shape = None):

        # Initialize parent class
        super().__init__()

        # Define all layer to be used
        self.network = nn.Sequential(

                    # First Convolutional "Sequence"
                    nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),  
                    nn.BatchNorm2d(num_features=32),                                                 
                    nn.ReLU(),                                                                       
                    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),  
                    nn.BatchNorm2d(num_features=64),                                                
                    nn.ReLU(), 
                    nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                    # Second Convolutional "Sequence"
                    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1), 
                    nn.BatchNorm2d(num_features=128), 
                    nn.ReLU(),
                    nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(num_features=256),
                    nn.ReLU(),                                                                                                                                                   
                    nn.MaxPool2d(kernel_size=2, stride=2, padding=0), 
                    # Third Convolutional "Sequence"
                    nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1), 
                    nn.BatchNorm2d(num_features=512), 
                    nn.ReLU(),
                    nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(num_features=512),
                    nn.ReLU(),                                                                                                                                                   
                    nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                    # Flatten to one-dimensional tensor
                    nn.Flatten(start_dim=1),           
                    # First Linear "Sequence"                                              
                    nn.LazyLinear(out_features=128),
                    nn.BatchNorm1d(num_features=128),                                                   
                    nn.ReLU(),                                                                      
                    nn.Dropout(p=0.25), 
                    # Second Linear "Sequence"                                                             
                    nn.LazyLinear(out_features=64),
                    nn.BatchNorm1d(num_features=64), 
                    nn.ReLU(),
                    nn.Dropout(p=0.15),
                    # Output Linear Layer
                    nn.LazyLinear(out_features=2)  

        )

        # Print model summary if given input shape
        if in_shape != None:
            torchsummary.summary(self, in_shape)

        return
    
    # Forward pass of the networks, return output from last dense layer 
    def forward(self, x):

        return self.network(x)