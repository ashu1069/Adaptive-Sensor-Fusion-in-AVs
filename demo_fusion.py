import torch
from pointnet.pointnet_cls import PointNetCls
from resnet.resnet import ImageFeatureExtractor, get_transform
from utils.adaptive_fusion import AdaptiveFusion
from PIL import Image
import numpy as np

def demo_fusion(point_cloud, image_path):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize all models
    pointnet = PointNetCls().to(device)
    resnet = ImageFeatureExtractor().to(device)
    fusion_module = AdaptiveFusion().to(device)
    
    # Set models to eval mode
    pointnet.eval()
    resnet.eval()
    fusion_module.eval()
    
    # Process point cloud
    point_cloud = torch.FloatTensor(point_cloud).to(device)
    if len(point_cloud.shape) == 2:  # If point cloud is [N, 3]
        point_cloud = point_cloud.unsqueeze(0)  # Add batch dimension
    point_cloud = point_cloud.transpose(1, 2)  # Change to [B, 3, N] format
    
    # Process image
    transform = get_transform()
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    
    # Extract features
    with torch.no_grad():
        # Get LiDAR features from PointNet
        lidar_features, _, _ = pointnet(point_cloud)
        
        # Get image features from ResNet
        image_features = resnet(image)
        
        # Perform fusion
        fused_features = fusion_module(lidar_features, image_features)
    
    return fused_features

if __name__ == "__main__":
    # Create sample inputs
    # Sample point cloud: 1024 points with xyz coordinates
    sample_points = np.random.randn(1024, 3)
    
    # Sample image path (replace with your actual image path)
    image_path = "samples/image_left/000044.png"
    
    # Perform fusion
    fused_features = demo_fusion(sample_points, image_path)
    
    print("\nFeature dimensions:")
    print(f"Fused features shape: {fused_features.shape}")  # Should be [1, 512] 