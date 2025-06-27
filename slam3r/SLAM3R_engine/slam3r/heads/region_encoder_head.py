import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

# Spatial Region Encoder with 3D Convolution
# This is used to compress the spatial information of the input image into a single vector
class SpatialRegionEncoder_3DConv(nn.Module):
    def __init__(self, ):
        super().__init__()
        
# Spatial Region Encoder with LSTM
# This is used to compress the spatial information of the input image into a single vector
class SpatialRegionEncoder_LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x)

# Spatial Region Encoder with LSTM and Attention
class SpatialRegionEncoder_LSTMAttn(nn.Module):
    def __init__(self, ):
        super().__init__()

# Spatial Region Encoder with Transformer and RoPE
class SpatialRegionEncoder_Transformer(nn.Module):
    def __init__(self, ):
        super().__init__()


class SpatialRegionEncoder_Hybrid(nn.Module):
    def __init__(self, ):
        super().__init__()

# Frame2Region
# Used compares frames to the compressedspatial region 
class Frame2Region(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net(x)
