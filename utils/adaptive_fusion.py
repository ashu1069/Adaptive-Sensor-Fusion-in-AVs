import torch
import torch.nn as nn
from sklearn.decomposition import PCA
import numpy as np

class AdaptiveFusion(nn.Module):
    def __init__(self, lidar_dim=512, image_dim=2048, output_dim=512):
        super(AdaptiveFusion, self).__init__()
        
        self.output_dim = output_dim
        self.lidar_dim = lidar_dim
        self.image_dim = image_dim
        
        # Remove PCA and replace with linear projection
        self.image_projection = nn.Linear(image_dim, output_dim)
        
        # Weighted fusion parameters (trainable)
        self.weight_gate = nn.Sequential(
            nn.Linear(lidar_dim + output_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 2),
            nn.Softmax(dim=1)
        )
        
        # Gating mechanism
        self.fusion_gate = nn.Sequential(
            nn.Linear(lidar_dim + output_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.Sigmoid()
        )
        
        # Modality-specific adaptation layers
        self.lidar_adapter = nn.Linear(lidar_dim, output_dim)
        self.image_adapter = nn.Linear(output_dim, output_dim)
    
    def _process_image_features(self, image_features):
        # Direct linear projection instead of PCA
        projected_features = self.image_projection(image_features)
        return self.image_adapter(projected_features)
    
    def forward(self, lidar_features=None, image_features=None):
        """
        Handles three cases:
        1. Both modalities present: adaptive fusion
        2. Only LiDAR present: use LiDAR features
        3. Only image present: use image features
        """
        
        # Check which modalities are available
        has_lidar = lidar_features is not None
        has_image = image_features is not None
        
        # Case 1: Only LiDAR available
        if has_lidar and not has_image:
            return self.lidar_adapter(lidar_features)
        
        # Case 2: Only image available
        if has_image and not has_lidar:
            return self._process_image_features(image_features)
        
        # Case 3: Both modalities available - do adaptive fusion
        if has_lidar and has_image:
            adapted_lidar = self.lidar_adapter(lidar_features)
            adapted_image = self._process_image_features(image_features)
            
            # Concatenate features for weight calculation
            combined_features = torch.cat([adapted_lidar, adapted_image], dim=1)
            
            # Calculate adaptive weights
            weights = self.weight_gate(combined_features)
            lidar_weight = weights[:, 0].unsqueeze(1)
            image_weight = weights[:, 1].unsqueeze(1)
            
            # Print weights
            print(f"\nAdaptive Weights:")
            print(f"LiDAR weights: {lidar_weight.mean().item():.4f}")
            print(f"Image weights: {image_weight.mean().item():.4f}")
            
            # Weighted combination
            weighted_fusion = (lidar_weight * adapted_lidar + 
                             image_weight * adapted_image)
            
            # Calculate and apply gating values
            gate_values = self.fusion_gate(combined_features)
            print(f"Gate values mean: {gate_values.mean().item():.4f}")
            
            gated_fusion = weighted_fusion * gate_values
            
            return gated_fusion
        
        # Case 4: No modalities available
        raise ValueError("At least one modality (LiDAR or image) must be provided")

# Example usage
if __name__ == "__main__":
    # Create sample inputs
    batch_size = 4
    lidar_features = torch.randn(batch_size, 512)  # From PointNet
    image_features = torch.randn(batch_size, 2048)  # From ResNet
    
    # Initialize fusion module
    fusion_module = AdaptiveFusion()
    
    # Test all cases
    print("\nTesting all fusion cases:")
    
    # Case 1: Both modalities
    fused_features = fusion_module(lidar_features, image_features)
    print(f"Both modalities - output shape: {fused_features.shape}")
    
    # Case 2: Only LiDAR
    lidar_only = fusion_module(lidar_features=lidar_features)
    print(f"LiDAR only - output shape: {lidar_only.shape}")
    
    # Case 3: Only image
    image_only = fusion_module(image_features=image_features)
    print(f"Image only - output shape: {image_only.shape}")
