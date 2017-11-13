import torch
import torch.nn as nn
import torch.nn.functional as F

class FullyConnected(nn.Module):
    
    def __init__(self, dimensions, classes):
        
        self.dimensions = dimensions
        self.classes = classes
        
        super(FullyConnected, self).__init__()
        
        self.fc1 = nn.Linear(dimensions, 256, bias=True)
        self.bn1 = nn.BatchNorm1d(256, affine=True)
        
        self.fc2 = nn.Linear(256, 128, bias=True)
        self.bn2 = nn.BatchNorm1d(128, affine=True)
        
        self.fc3 = nn.Linear(128, 64, bias=True)
        self.bn3 = nn.BatchNorm1d(64, affine=True)
        
        self.fc4 = nn.Linear(64, 64, bias=True)
        self.bn4 = nn.BatchNorm1d(64, affine=True)
        
        self.fc_out = nn.Linear(64, classes, bias=True)
        
    def forward(self, x):
        
        x = x.view(-1, self.dimensions)
        
        x = self.bn1(self.fc1(x))
        x = F.relu(x)
        
        x = self.bn2(self.fc2(x))
        x = F.relu(x)
        
        x = self.bn3(self.fc3(x))
        x = F.relu(x)
        
        x = self.bn4(self.fc4(x))
        x = F.relu(x)
        
        out = self.fc_out(x)
        
        return out