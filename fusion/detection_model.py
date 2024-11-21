import torch
import torch.nn as nn
from fusion import FusionModule
from detection_head import DetectionHead

class DetectionModel(nn.Module):
    def __init__(self, img_channels=2048, lidar_channels=1024, num_classes=9, num_anchors=9):
        super().__init__()
        
        # Fusion module
        self.fusion_module = FusionModule(img_channels, lidar_channels)
        
        # Detection head
        self.detection_head = DetectionHead(img_channels, num_classes, num_anchors)
        
    def forward(self, img_feats, lidar_feats):
        # Fuse features
        fused_features = self.fusion_module(img_feats, lidar_feats)
        
        # Get predictions
        bbox_pred, cls_pred = self.detection_head(fused_features)
        
        return bbox_pred, cls_pred