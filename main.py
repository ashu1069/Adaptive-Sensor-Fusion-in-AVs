from utils.resnet_detection_head import DetectionHead
import torch

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