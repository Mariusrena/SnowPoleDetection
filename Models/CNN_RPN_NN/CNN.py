import torch.nn as nn
import torch.nn.functional as F

""" Standard Imports """
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from Models.Hyperparameters import NUM_CNN_OUTPUT_CHANNELS

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layer_1 = nn.Conv2d(3, 64, 5, 1) # In channels, out channels, kernel size, stride (Bias is true by default)
        self.conv_layer_2 = nn.Conv2d(64, 128, 5, 1)
        self.conv_layer_3 = nn.Conv2d(128, NUM_CNN_OUTPUT_CHANNELS, 5, 1)
        #self.conv_layer_4 = nn.Conv2d(256, 512, 5, 1)
        #self.conv_layer_5 = nn.Conv2d(512, NUM_CNN_OUTPUT_CHANNELS, 5, 1)
        
        self.max_pooling = nn.MaxPool2d(2,2) # Kernel size, stride <- Effectively halves number of features in each channel
        
    def forward(self, x):
        x = self.max_pooling(F.silu(self.conv_layer_1(x)))
        x = self.max_pooling(F.silu(self.conv_layer_2(x)))
        x = self.max_pooling(F.silu(self.conv_layer_3(x)))
        #x = self.max_pooling(F.silu(self.conv_layer_4(x)))
        #x = self.max_pooling(F.silu(self.conv_layer_5(x)))
        return x
    
# Used to print number of parameters in this architecture    
if __name__ == "__main__":
    net = ConvNet()
    total_params = sum(p.numel() for p in net.parameters())
    print(total_params)
