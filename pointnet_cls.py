import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from t_net import TNet

class PointNetCls(nn.Module):
    def __init__(self, feature_dim=1024):
        super(PointNetCls, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, 1)
        self.conv2 = nn.Conv2d(64, 64, 1)
        self.conv3 = nn.Conv2d(64, 64, 1)
        self.conv4 = nn.Conv2d(64, 128, 1)
        self.conv5 = nn.Conv2d(128, 1024, 1)
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(1024)
        
        self.fc1 = nn.Linear(1024, feature_dim)
        
        self.bn6 = nn.BatchNorm1d(feature_dim)
        
        self.t_net = TNet()

    def forward(self, x, is_training=True):
        batch_size = x.size()[0]
        num_points = x.size()[2]
        
        # Input transform
        trans = self.t_net.forward_input_transform(x)
        x = torch.bmm(x, trans)
        
        # MLP 1
        x = x.transpose(2, 1).unsqueeze(-1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        
        # Feature transform
        trans_feat = self.t_net.forward_feature_transform(x)
        x = x.squeeze(-1)
        x = torch.bmm(x.transpose(2, 1), trans_feat)
        x = x.transpose(2, 1).unsqueeze(-1)
        
        # MLP 2
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        
        # Max pooling
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        
        # Get features
        x = F.relu(self.bn6(self.fc1(x)))
        
        return x, trans_feat

if __name__ == '__main__':
    model = PointNetCls(feature_dim=512)
    xyz = torch.randn(32, 3, 1024)
    features, _ = model(xyz)
    print(f"Feature shape: {features.size()}")