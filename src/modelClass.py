import torch 
import torch.nn as nn
import torch.nn.functional as F

class Model (nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(8, 4),
            torch.nn.BatchNorm1d(4),
            torch.nn.ReLU(),
            torch.nn.Linear(4, 2),
            torch.nn.BatchNorm1d(2),

            torch.nn.Softmax(dim=0)
        )

    def forward(self, x):
        return self.model(x)


        
