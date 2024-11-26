import torch
import torch.nn as nn
from fusion import FusionModule
from detection_head import DetectionHead

class DetectionModel(nn.Module):
    def __init__(self, img_channels=2048, lidar_channels=512, num_classes=9):
        super().__init__()
        
        # Fusion module
        self.fusion_module = FusionModule(
            img_channels=img_channels, 
            lidar_channels=lidar_channels,
        )
        
        # Detection head takes fused features as input
        self.detection_head = DetectionHead(
            in_channels=img_channels,  
            num_classes=num_classes
        )
        
    def forward(self, img_feats, lidar_feats):
        # Fuse features
        fused_feats, attention_weights = self.fusion_module(img_feats, lidar_feats)
        
        # Get predictions from detection head
        bbox_pred, cls_pred = self.detection_head(fused_feats)
        
        # Select top K predictions based on classification confidence
        K = 8  # Match the number of objects in your example
        B = bbox_pred.size(0)
        
        # Get top K predictions for each sample in batch
        scores, class_ids = torch.max(cls_pred, dim=2)  # Get max class score and ID
        topk_scores, topk_idx = torch.topk(scores, k=K, dim=1)  # (B, K)
        
        # Gather corresponding boxes and classes
        final_boxes = torch.gather(bbox_pred, 1, topk_idx.unsqueeze(-1).expand(-1, -1, 4))  # (B, K, 4)
        final_classes = torch.gather(class_ids, 1, topk_idx)  # (B, K)
        
        return final_boxes, final_classes, attention_weights

    def get_attention_weights(self):
        """Returns the attention weights from the last forward pass"""
        return self.fusion_module.attention.get_attention_weights()