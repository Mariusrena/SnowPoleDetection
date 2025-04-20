import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

""" Standard Imports """
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from FPN_RPN_NN.Hyperparameters import NUM_CNN_OUTPUT_CHANNELS

""" 
    Conv blocks and Residual blocks as defined in Darknet53 used in YOLOv3.
    Improves upon previous simple convnet by:
        - Switching out Maxpool with ConvLayer of stride 2.
          Resulting in a learned feature reduction rather than a static one. 
        - Uses 1x1 layers to reduce number of channels.
          This makes the network smaller, while retaining most of the effectiveness. 
"""
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_batch_norm=True, **kwargs): 
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not use_batch_norm, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity()

    def forward(self, x):
        return F.silu(self.bn(self.conv(x)))

class ResidualBlock(nn.Module):
    def __init__(self, channels, use_residual=True, num_repeats=1):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_repeats):
            self.layers += [
                nn.Sequential(
                    ConvBlock(channels, channels // 2, kernel_size=1),
                    ConvBlock(channels // 2, channels, kernel_size=3, padding=1),
                )
            ]

        self.use_residual = use_residual
        self.num_repeats = num_repeats

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x) if self.use_residual else layer(x)
        return x

class ConvFPN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layer1 = ConvBlock(3, 64, True, kernel_size=3, stride=1)
        self.res_block1 = ResidualBlock(64, True, 3)
        self.down_sample1 = ConvBlock(64, 128, kernel_size=3, stride=2) # Replaces Maxpool
        self.res_block2 = ResidualBlock(128, True, 5)
        self.down_sample2 = ConvBlock(128, 256, kernel_size=3, stride=2) # Replaces Maxpool
        self.res_block3 = ResidualBlock(256, True, 5)
        self.down_sample3 = ConvBlock(256, NUM_CNN_OUTPUT_CHANNELS, kernel_size=3, stride=2) # Replaces Maxpool

        # Lateral 1x1 Convs to match top-down channels
        self.lat2 = nn.Conv2d(128, NUM_CNN_OUTPUT_CHANNELS, 1, padding=0)
        self.lat3 = nn.Conv2d(256, NUM_CNN_OUTPUT_CHANNELS, 1, padding=0)
        self.lat4 = nn.Conv2d(NUM_CNN_OUTPUT_CHANNELS, NUM_CNN_OUTPUT_CHANNELS, 1, padding=0)

        # Output 3x3 convs to smooth after fusion
        self.out3 = nn.Conv2d(NUM_CNN_OUTPUT_CHANNELS, NUM_CNN_OUTPUT_CHANNELS, 3, padding=1)
        self.out4 = nn.Conv2d(NUM_CNN_OUTPUT_CHANNELS, NUM_CNN_OUTPUT_CHANNELS, 3, padding=1)
        self.out5 = nn.Conv2d(NUM_CNN_OUTPUT_CHANNELS, NUM_CNN_OUTPUT_CHANNELS, 3, padding=1)

        #self._initialize_weights()

    def _initialize_weights(self):
        print("Initializing weights...")
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Kaiming Normal initialization for weights
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # Initialize bias to zero if it exists
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv_layer1(x)
        x1 = self.res_block1(x)
        x2 = self.down_sample1(x1) 
        x3 = self.res_block2(x2)
        x4 = self.down_sample2(x3)
        x5 = self.res_block3(x4) 
        x6 = self.down_sample3(x5)

        # Top-down FPN pathway
        p4 = self.lat4(x6)                
        p3 = self.lat3(x4) + F.interpolate(p4, size=x4.shape[-2:], mode='nearest')
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
