import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from t_net import TNet, TransformNet

class PointNetCls(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.feature_transform = TransformNet()
        
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)

    def forward(self, x):
        x, matrix3x3, matrix64x64 = self.feature_transform(x)
        x = F.relu(self.bn1(self.fc1(x)))
        
        return x, matrix3x3, matrix64x64

if __name__ == "__main__":
    # Create a sample input (batch_size=2, channels=3, points=5)
    sample_input = torch.randn(2, 3, 5)
    
    # Test PointNetCls
    model = PointNetCls()
    output, mat3, mat64 = model(sample_input)
    
    print("\nPointNetCls outputs:")
    print("Feature representation shape:", output.shape)      # Should be [2, 512]
    print("Matrix3x3 shape:", mat3.shape)                   # Should be [2, 3, 3]
    print("Matrix64x64 shape:", mat64.shape)                # Should be [2, 64, 64]
