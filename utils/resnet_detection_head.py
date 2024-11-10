import os
import sys
# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.adaptive_fusion import AdaptiveFusion
from pointnet.pointnet_cls import PointNetCls
from resnet.resnet import ImageFeatureExtractor

class DetectionHead(nn.Module):
    def __init__(self, num_classes=3, confidence_threshold=0.5):
        super(DetectionHead, self).__init__()
        
        # Initialize feature extractors
        self.pointnet = PointNetCls()
        self.resnet = ImageFeatureExtractor()
        
        # Initialize adaptive fusion module
        self.fusion_module = AdaptiveFusion(
            lidar_dim=512,      # From PointNetCls
            image_dim=2048,     # From ResNet50
            output_dim=512      # Fused feature dimension
        )
        
        # Detection head layers
        self.detection_layers = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Output layers for classification and bounding box regression
        self.classification = nn.Linear(128, num_classes)
        self.bbox_regression = nn.Linear(128, 4)  # (x, y, w, h)
        
        self.confidence_threshold = confidence_threshold
        
    def forward(self, point_cloud=None, image=None):
        """
        Forward pass of the detection head
        Args:
            point_cloud: Point cloud input (B, 3, N)
            image: Image input (B, C, H, W)
        Returns:
            class_scores: Classification logits (B, num_classes)
            bbox_pred: Bounding box predictions (B, 4)
        """
        # Extract features from each modality
        lidar_features = None
        image_features = None
        
        if point_cloud is not None:
            lidar_features, _, _ = self.pointnet(point_cloud)
            
        if image is not None:
            image_features = self.resnet(image)
        
        # Get fused features using adaptive fusion
        fused_features = self.fusion_module(
            lidar_features=lidar_features,
            image_features=image_features
        )
        
        # Pass through detection layers
        shared_features = self.detection_layers(fused_features)
        
        # Get classification scores and bbox predictions
        class_scores = self.classification(shared_features)
        bbox_pred = self.bbox_regression(shared_features)
        
        return class_scores, bbox_pred
    
    def predict(self, point_cloud=None, image=None):
        """
        Prediction method with confidence thresholding
        """
        class_scores, bbox_pred = self.forward(point_cloud, image)
        
        # Apply softmax to get probabilities
        class_probs = F.softmax(class_scores, dim=1)
        
        # Get predicted class and confidence
        confidence, predicted_class = torch.max(class_probs, dim=1)
        
        # Apply confidence threshold
        mask = confidence >= self.confidence_threshold
        
        return {
            'class_probs': class_probs[mask],
            'predicted_class': predicted_class[mask],
            'confidence': confidence[mask],
            'bbox_pred': bbox_pred[mask]
        }


# Example usage
if __name__ == "__main__":
    # Create sample inputs
    batch_size = 4
    point_cloud = torch.randn(batch_size, 3, 1024)  # (B, 3, N) point cloud
    image = torch.randn(batch_size, 3, 224, 224)    # (B, C, H, W) image
    
    # Initialize detection head
    detection_head = DetectionHead(num_classes=3)
    
    # Test forward pass
    class_scores, bbox_pred = detection_head(point_cloud, image)
    print("\nDetection Head outputs:")
    print(f"Class scores shape: {class_scores.shape}")  # Should be [4, 3]
    print(f"Bbox predictions shape: {bbox_pred.shape}") # Should be [4, 4]
    
    # Test with single modality
    print("\nTesting single modality:")
    class_scores_lidar, bbox_pred_lidar = detection_head(point_cloud=point_cloud)
    print(f"LiDAR-only class scores shape: {class_scores_lidar.shape}")
    
    class_scores_image, bbox_pred_image = detection_head(image=image)
    print(f"Image-only class scores shape: {class_scores_image.shape}")
