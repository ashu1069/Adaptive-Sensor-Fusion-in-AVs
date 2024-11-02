import torch
import torch.nn as nn
import numpy as np
import sys
import os

class TNet(nn.Module):
    def __init__(self):
        super(TNet, self).__init__()
        
        # Input transform layers
        self.input_transform = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=[1,3], stride=[1,1]),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(64, 128, kernel_size=[1,1], stride=[1,1]),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.Conv2d(128, 1024, kernel_size=[1,1], stride=[1,1]),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
        )
        
        self.input_mlp = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        
        self.input_transform_layer = nn.Linear(256, 9)  # 3*3 matrix
        
        # Feature transform layers
        self.feature_transform = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=[1,1], stride=[1,1]),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(64, 128, kernel_size=[1,1], stride=[1,1]),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.Conv2d(128, 1024, kernel_size=[1,1], stride=[1,1]),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
        )
        
        self.feature_mlp = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        
        self.feature_transform_layer = nn.Linear(256, 64*64)  # K*K matrix
    
    def forward_input_transform(self, point_cloud):
        batch_size = point_cloud.size(0)
        input_image = point_cloud.unsqueeze(2)
        
        net = self.input_transform(input_image.transpose(1,3))
        net = torch.max(net, 2, keepdim=True)[0]
        net = net.view(batch_size, -1)
        net = self.input_mlp(net)
        
        transform = self.input_transform_layer(net)
        init_transform = torch.eye(3, device=point_cloud.device).view(1, 9).repeat(batch_size, 1)
        transform = transform.view(batch_size, 3, 3)
        transform += init_transform.view(batch_size, 3, 3)
        
        return transform
    
    def forward_feature_transform(self, inputs):
        batch_size = inputs.size(0)
        
        net = self.feature_transform(inputs.transpose(1,3))
        net = torch.max(net, 2, keepdim=True)[0]
        net = net.view(batch_size, -1)
        net = self.feature_mlp(net)
        
        transform = self.feature_transform_layer(net)
        init_transform = torch.eye(64, device=inputs.device).view(1, 64*64).repeat(batch_size, 1)
        transform = transform.view(batch_size, 64, 64)
        transform += init_transform.view(batch_size, 64, 64)
        
        return transform