import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TNet(nn.Module):
    def __init__(self, k=3):
        super(TNet, self).__init__()
        self.k = k
        
        # Shared MLP layers
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(1, k))
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(1, 1))
        self.conv3 = nn.Conv2d(128, 1024, kernel_size=(1, 1))
        
        # Fully connected layers
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        
        # Batch normalization layers
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        
        # Initialize the final FC layer with zeros
        nn.init.zeros_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Shared MLP with batch norm and ReLU
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Global max pooling
        x = x.view(batch_size, 1024, -1)
        x = torch.max(x, 2)[0]
        
        # Fully connected layers
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        
        # Add identity matrix to the output
        identity = torch.eye(self.k, requires_grad=True).repeat(batch_size, 1, 1)
        if x.is_cuda:
            identity = identity.cuda()
        x = self.fc3(x).view(-1, self.k, self.k) + identity
        
        return x

class TransformNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_transform = TNet(k=3)
        self.feature_transform = TNet(k=64)

        self.conv1 = nn.Conv1d(3, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.conv3 = nn.Conv1d(128, 1024, kernel_size=1)

        self.bn1 = nn.BatchNorm1d(64)   
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

    def forward(self, x):
        # Reshape input for TNet (batch_size, 1, points, channels)
        x_transform = x.unsqueeze(1).transpose(2, 3)
        matrix3x3 = self.input_transform(x_transform)
        
        # batch matrix multiplication
        x = torch.bmm(torch.transpose(x, 1, 2), matrix3x3).transpose(1, 2)

        x = F.relu(self.bn1(self.conv1(x)))

        # Reshape for feature transform
        x_transform = x.unsqueeze(1).transpose(2, 3)
        matrix64x64 = self.feature_transform(x_transform)
        x = torch.bmm(torch.transpose(x, 1, 2), matrix64x64).transpose(1, 2)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = nn.MaxPool1d(x.size(-1))(x)
        output = nn.Flatten(1)(x)

        return output, matrix3x3, matrix64x64


if __name__ == "__main__":
    # Create a sample input (batch_size=2, channels=3, points=5)
    sample_input = torch.randn(2, 3, 5)
    # Reshape for TNet input (batch_size, 1, points, channels)
    tnet_input = sample_input.unsqueeze(1).transpose(2, 3)
    
    # Test TNet
    tnet = TNet(k=3)
    tnet_output = tnet(tnet_input)
    print("TNet output shape:", tnet_output.shape)  # Should be [2, 3, 3]
    
    # Test TransformNet
    transform_net = TransformNet()
    features, mat3, mat64 = transform_net(sample_input)
    print("\nTransformNet outputs:")
    print("Features shape:", features.shape)        # Should be [2, 1024]
    print("Matrix3x3 shape:", mat3.shape)          # Should be [2, 3, 3]
    print("Matrix64x64 shape:", mat64.shape)       # Should be [2, 64, 64]

