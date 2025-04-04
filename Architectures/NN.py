import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops

class ROI_NN(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.reg_layer1 = nn.Linear(in_features, 256)
        self.reg_layer2 = nn.Linear(256, 256)
        self.reg_layer3 = nn.Linear(256, 4)
        
        self.prob_layer1 = nn.Linear(in_features, 256)
        self.prob_layer2 = nn.Linear(256, 256)
        self.prob_layer3 = nn.Linear(256, 1)

    def forward(self, x):
        
        x1 = F.silu(self.reg_layer1(x))
        x1 = F.silu(self.reg_layer2(x1))
        bbox = self.reg_layer3(x1)
        
        x2 = F.silu(self.prob_layer1(x))
        x2 = F.silu(self.prob_layer2(x2)) 
        prob = self.prob_layer3(x2)

        return bbox, prob
    
# Used to print number of parameters in this architecture    
if __name__ == "__main__":
    net = ROI_NN(2000)
    total_params = sum(p.numel() for p in net.parameters())
    print(total_params)
