import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .t_net import TNet, TransformNet

class PointNetCls(nn.Module):
    def __init__(self, num_classes=40):  # Changed default to ModelNet40
        super().__init__()
        
        self.feature_transform = TransformNet()
        
        # MLP (1024 -> 512 -> 256 -> k)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x, return_global_features=True):
        print(f'input into pointnet = > {x.shape}')
        # Get global features and transformation matrices
        global_features, matrix3x3, matrix64x64 = self.feature_transform(x)
        
        # MLP classification
        x = F.relu(self.bn1(self.fc1(global_features)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)  # Output k scores
        
        if return_global_features:
            return F.log_softmax(x, dim=1), matrix3x3, matrix64x64, global_features
        return F.log_softmax(x, dim=1), matrix3x3, matrix64x64

# if __name__ == "__main__":
#     # Create a sample input (batch_size=2, channels=3, points=5)
#     sample_input = torch.randn(2, 3, 5)
    
#     # Test PointNetCls
#     model = PointNetCls()
#     output, mat3, mat64 = model(sample_input)
    
#     print("\nPointNetCls outputs:")
#     print("Feature representation shape:", output.shape)      # Should be [2, 512]
#     print("Matrix3x3 shape:", mat3.shape)                   # Should be [2, 3, 3]
#     print("Matrix64x64 shape:", mat64.shape)                # Should be [2, 64, 64]
