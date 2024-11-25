import torch
import torch.nn as nn

class CrossModalAttention(nn.Module):
    def __init__(self, img_channels, lidar_channels):
        super().__init__()
        self.img_channels = img_channels
        self.lidar_channels = lidar_channels
        
        # Project image features to query space
        self.query_proj = nn.Conv2d(img_channels, lidar_channels, 1)
        
        # Project lidar features for key and value
        self.key_proj = nn.Linear(lidar_channels, lidar_channels)
        self.value_proj = nn.Linear(lidar_channels, lidar_channels)
        
        self.scale = torch.sqrt(torch.FloatTensor([lidar_channels]))
        
        # Add attention weights storage
        self.last_attention_weights = None

    def forward(self, img_feats, lidar_feats):
        batch_size, _, H, W = img_feats.shape
        
        # Handle missing LiDAR features
        if lidar_feats is None:
            self.last_attention_weights = None
            # Return zero-initialized attended features
            return torch.zeros(batch_size, self.lidar_channels, H, W, device=img_feats.device)
            
        # Project image features to create queries
        # Shape: (B, C, H, W) -> (B, H*W, C)
        queries = self.query_proj(img_feats)
        queries = queries.view(batch_size, self.lidar_channels, -1).permute(0, 2, 1)
        
        # Project lidar features to create keys and values
        # Shape: (B, C) -> (B, 1, C)
        keys = self.key_proj(lidar_feats).unsqueeze(1)
        values = self.value_proj(lidar_feats).unsqueeze(1)
        
        # Compute attention scores
        # (B, H*W, C) @ (B, C, 1) -> (B, H*W, 1)
        attention = torch.bmm(queries, keys.transpose(1, 2)) / self.scale.to(queries.device)
        attention = torch.softmax(attention, dim=1)
        
        # Store attention weights
        self.last_attention_weights = attention.view(batch_size, H, W, 1)
        
        # Apply attention to values
        # (B, H*W, 1) @ (B, 1, C) -> (B, H*W, C)
        out = torch.bmm(attention, values)
        
        # Reshape back to spatial dimensions
        # (B, H*W, C) -> (B, C, H, W)
        out = out.permute(0, 2, 1).view(batch_size, self.lidar_channels, H, W)
        
        return out

    def get_attention_weights(self):
        """Returns the attention weights from the last forward pass."""
        return self.last_attention_weights

class FusionModule(nn.Module):
    def __init__(self, img_channels=2048, lidar_channels=512):
        super().__init__()
        
        self.img_channels = img_channels
        self.lidar_channels = lidar_channels
        # Cross-modal attention
        self.attention = CrossModalAttention(img_channels, lidar_channels)
        
        # Final fusion through concatenation and projection
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(img_channels + lidar_channels, img_channels, 1),
            nn.BatchNorm2d(img_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, img_feats, lidar_feats):
        """
        Args:
            img_feats: ResNet features (B, C1, H, W) or None
            lidar_feats: PointNet features (B, C2) or None
        Returns:
            Fused features (B, C1, H, W)
        """
        # Handle missing modalities
        if img_feats is None and lidar_feats is None:
            raise ValueError("At least one modality (image or LiDAR) must be present")
            
        if img_feats is None:
            # If only LiDAR features are present, project them to image space
            B = lidar_feats.shape[0]
            img_feats = torch.zeros(B, self.img_channels, 7, 7, device=lidar_feats.device)
            
        if lidar_feats is None:
            # If only image features are present, skip attention and return image features
            return img_feats
        # Regular fusion when both modalities are present
        attended_lidar = self.attention(img_feats, lidar_feats)
        fused_feats = torch.cat([img_feats, attended_lidar], dim=1)
        output = self.fusion_conv(fused_feats)
        
        # Return both output and attention weights
        return output, self.attention.get_attention_weights()

def test_fusion():
    batch_size = 4
    img_channels = 2048
    lidar_channels = 1024
    H, W = 7, 7
    
    fusion_module = FusionModule(img_channels, lidar_channels)
    
    # Test case 1: Both modalities present
    img_feats = torch.randn(batch_size, img_channels, H, W)
    lidar_feats = torch.randn(batch_size, lidar_channels)
    output, attention_weights = fusion_module(img_feats, lidar_feats)
    print(f"Both modalities - output shape: {output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    print(f"Attention weights mean: {attention_weights.mean().item():.4f}")
    
    # Test case 2: Only image features
    output_img_only = fusion_module(img_feats, None)
    # Since no attention is performed, output is just the image features
    print(f"Image only - output shape: {output_img_only.shape}")
    
    # Test case 3: Only LiDAR features
    output_lidar_only, _ = fusion_module(None, lidar_feats)  # Unpack tuple
    print(f"LiDAR only - output shape: {output_lidar_only.shape}")

if __name__ == "__main__":
    test_fusion()
