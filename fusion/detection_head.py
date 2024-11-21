import torch
import torch.nn as nn

class DetectionHead(nn.Module):
    def __init__(self, in_channels, num_classes, num_anchors=9):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        # Shared convolutional layers for feature refinement
        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        # Bounding box regression head
        self.bbox_head = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, num_anchors * 4, kernel_size=1)  # 4 values per anchor (dx,dy,dw,dh)
        )
        
        # Classification head
        self.cls_head = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, num_anchors * num_classes, kernel_size=1)
        )

    def forward(self, x):
        # Shared features
        shared_features = self.shared_conv(x)
        
        # Bounding box predictions
        # Output shape: (B, num_anchors*4, H, W)
        bbox_pred = self.bbox_head(shared_features)
        
        # Class predictions
        # Output shape: (B, num_anchors*num_classes, H, W)
        cls_pred = self.cls_head(shared_features)
        
        # Reshape predictions to (B, H, W, num_anchors*4) and (B, H, W, num_anchors*num_classes)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1)
        cls_pred = cls_pred.permute(0, 2, 3, 1)
        
        return bbox_pred, cls_pred