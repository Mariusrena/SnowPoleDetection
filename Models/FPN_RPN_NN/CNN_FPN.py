import torch.nn as nn
import torch.nn.functional as F

""" Standard Imports """
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from FPN_RPN_NN.Hyperparameters import NUM_CNN_OUTPUT_CHANNELS

class ConvFPN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layer_1 = nn.Conv2d(3, 64, 5, 1) # In channels, out channels, kernel size, stride (Bias is true by default)
        self.conv_layer_2 = nn.Conv2d(64, 128, 5, 1)
        self.conv_layer_3 = nn.Conv2d(128, 256, 5, 1)
        self.conv_layer_4 = nn.Conv2d(256, 512, 5, 1)
        
        self.max_pooling = nn.MaxPool2d(2,2) # Kernel size, stride <- Effectively halves number of features in each channel

        # Lateral 1x1 Convs to match top-down channels
        self.lat2 = nn.Conv2d(128, 256, 1)
        self.lat3 = nn.Conv2d(256, 256, 1)
        self.lat4 = nn.Conv2d(512, 256, 1)

        # Output 3x3 convs to smooth after fusion
        self.out3 = nn.Conv2d(256, NUM_CNN_OUTPUT_CHANNELS, 3, padding=1)
        self.out4 = nn.Conv2d(256, NUM_CNN_OUTPUT_CHANNELS, 3, padding=1)
        self.out5 = nn.Conv2d(256, NUM_CNN_OUTPUT_CHANNELS, 3, padding=1)
        
    def forward(self, x):
        x1 = self.max_pooling(F.silu(self.conv_layer_1(x)))
        x2 = self.max_pooling(F.silu(self.conv_layer_2(x1)))
        x3 = self.max_pooling(F.silu(self.conv_layer_3(x2)))
        x4 = self.max_pooling(F.silu(self.conv_layer_4(x3)))

        # Top-down FPN pathway
        p4 = self.lat4(x4)                      # Reduce channels to 256
        p3 = self.lat3(x3) + F.interpolate(p4, size=x3.shape[-2:], mode='nearest')
        p2 = self.lat2(x2) + F.interpolate(p3, size=x2.shape[-2:], mode='nearest')

        # Smooth output feature maps
        p2 = self.out3(p2)
        p3 = self.out4(p3)
        p4 = self.out5(p4)

        return {"0": p2, "1": p3, "2": p4}
    
# Used to print number of parameters in this architecture    
if __name__ == "__main__":
    net = ConvFPN()
    total_params = sum(p.numel() for p in net.parameters())
    print(total_params)
