import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops

class ROI_NN(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.shared_input_layer = nn.Linear(in_features, 256)
        self.bn_shared = nn.BatchNorm1d(256)

        self.reg_layer1 = nn.Linear(256, 256)
        self.bn_reg1 = nn.BatchNorm1d(256)
        self.reg_layer2 = nn.Linear(256, 4)
        
        self.prob_layer1 = nn.Linear(256, 256)
        self.bn_prob1 = nn.BatchNorm1d(256)
        self.prob_layer2 = nn.Linear(256, 1)

        self.dropout = nn.Dropout(p = 0.2)

    def forward(self, x):
        x = self.shared_input_layer(x)
        x = self.bn_shared(x)
        x = F.silu(x)
        x = self.dropout(x)
        
        x1 = self.reg_layer1(x)
        x1 = self.bn_reg1(x1)
        x1 = F.silu(x1)
        x1 = self.dropout(x1)
        bbox = self.reg_layer2(x1)

        x2 = self.prob_layer1(x)
        x2 = self.bn_prob1(x2)
        x2 = F.silu(x2)
        x2 = self.dropout(x2)
        prob = self.prob_layer2(x2)
        prob = torch.clamp(prob, min=-20.0, max=20.0)

        return bbox, prob
    
# Used to print number of parameters in this architecture    
if __name__ == "__main__":
    net = ROI_NN(25088)
    total_params = sum(p.numel() for p in net.parameters())
    print(total_params)
