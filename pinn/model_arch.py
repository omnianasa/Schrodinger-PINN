import torch.nn as nn
import torch
from pinn.activation import Sin

class DoubleWallModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layer = nn.Linear(3, 256)
        self.l1 = nn.Linear(256, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 256)
        self.output_layer = nn.Linear(256, 2)
        #sin is better than relu based on the waves we deal with 
        self.sin = Sin()

    def forward(self, x, t, H):
        #t*20 provides enough motion for k0=5 with scale x = 2
        z = torch.cat([x * 2.0, t * 20.0, H], dim=1)
        x1 = self.sin(self.input_layer(z))
        x2 = self.sin(self.l1(x1)) + x1
        x3 = self.sin(self.l2(x2)) + x2
        x4 = self.sin(self.l2(x3)) + x3
        return self.output_layer(x4)