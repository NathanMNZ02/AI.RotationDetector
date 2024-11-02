import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvolutionalNetwork(nn.Module):
    def __init__(self):
        super(ConvolutionalNetwork, self).__init__()
        
        # Input: 224x224x1
        self.conv_layer1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        # Output: 224x224x16
        #self.pool_layer1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Output: 112x112x16
        
        # Input: 112x112x16
        self.conv_layer2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        # Output: 112x112x32
        #self.pool_layer2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Output: 56x56x32
        
        # Input: 56x56x64
        self.conv_layer3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        # Output: 56x56x64
        self.pool_layer3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Output: 28x28x64
        
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(28 * 28 * 64, 128)
        
        # Orientamento: 7 output (uno per ogni orientamento possibile)
        self.orientation_head = nn.Linear(128, 7) 
        
    def forward(self, x):
        x = F.relu(self.conv_layer1(x))
        #x = self.pool_layer1(x)
        
        x = F.relu(self.conv_layer2(x))
        #x = self.pool_layer2(x)
        
        x = F.relu(self.conv_layer3(x))
        x = self.pool_layer3(x)

        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        
        orientation_output = self.orientation_head(x) 
        
        return orientation_output
