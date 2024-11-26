import torch
import torch.nn as nn
from kitti_dataloader import get_dataloader, initialize_models

class CrossModalAttention(nn.Module):
    def __init__(self, img_channels=2048, lidar_channels=512):
        super().__init__()
        self.img_channels = img_channels
        self.lidar_channels = lidar_channels
        
        # Project image features to query space
        self.query_proj = nn.Conv2d(img_channels, lidar_channels, 1)
        
        # Project lidar features for key and value
        self.key_proj = nn.Linear(512, lidar_channels)
        self.value_proj = nn.Linear(512, lidar_channels)
        
        self.scale = torch.sqrt(torch.FloatTensor([lidar_channels]))
        
        self.last_attention_weights = None

    def forward(self, img_feats, lidar_feats):
        """
        Args:
            img_feats: (B, 2048, H, W)
            lidar_feats: (B, 512)
        Returns:
            out: (B, 512, H, W)
        """
        batch_size = img_feats.size(0)
        
        # Handle missing LiDAR features
        if lidar_feats is None:
            self.last_attention_weights = None
            return torch.zeros_like(img_feats)
            
        # Project image features to create queries
        queries = self.query_proj(img_feats)  # (B, 512, H, W)
        H, W = queries.shape[2:]
        queries = queries.view(batch_size, self.lidar_channels, -1)  # (B, 512, H*W)
        queries = queries.permute(0, 2, 1)  # (B, H*W, 512)
        
        # Project lidar features to create keys and values
        keys = self.key_proj(lidar_feats)      # (B, 512) -> (B, 512)
        values = self.value_proj(lidar_feats)  # (B, 512) -> (B, 512)
        
        # Reshape for attention
        keys = keys.unsqueeze(1)      # (B, 1, 512)
        values = values.unsqueeze(1)  # (B, 1, 512)
        
        # Compute attention scores
        attention = torch.bmm(queries, keys.transpose(1, 2))  # (B, H*W, 1)
        attention = attention / self.scale.to(queries.device)
        attention = torch.softmax(attention, dim=1)
        
        # Store attention weights
        self.last_attention_weights = attention.view(batch_size, H, W, 1)
        
        # Apply attention to values
        out = torch.bmm(attention, values)  # (B, H*W, 512)
        
        # Reshape back to spatial dimensions
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
            nn.BatchNorm2d(img_channels, track_running_stats=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, img_feats, lidar_feats):
        """
        Args:
            img_feats: ResNet features (B, C1, H, W) or None
            lidar_feats: PointNet features (B, C2) or None
        Returns:
            fused_feats: (B, img_channels, H, W)
            attention_weights: (B, H, W, 1) or None
        """
        # Handle missing modalities
        if img_feats is None and lidar_feats is None:
            raise ValueError("At least one modality (image or LiDAR) must be present")
            
        if img_feats is None:
            # If only LiDAR features are present, project them to image space
            B = lidar_feats.shape[0]
            img_feats = torch.zeros(B, self.img_channels, 7, 7, device=lidar_feats.device)
            
        if lidar_feats is None:
            fused_feats = img_feats
            attention_weights = None
        else:
            attended_lidar = self.attention(img_feats, lidar_feats)
            
            # Handle single-sample batches
            if img_feats.size(0) == 1:
                img_feats = img_feats.repeat(2, 1, 1, 1)
                attended_lidar = attended_lidar.repeat(2, 1, 1, 1)
                
            fused_feats = torch.cat([img_feats, attended_lidar], dim=1)
            fused_feats = self.fusion_conv(fused_feats)
            
            # If we repeated the batch, take only the first sample
            if img_feats.size(0) == 2 and lidar_feats.size(0) == 1:
                fused_feats = fused_feats[:1]
                
            attention_weights = self.attention.get_attention_weights()
        
        return fused_feats, attention_weights

def test_fusion():
    # Use the same dimensions as our actual data
    batch_size = 4
    img_channels = 2048    # ResNet50 output channels
    lidar_channels = 512   # PointNet output channels (changed from 1024)
    H, W = 7, 7           # Feature map size
    
    fusion_module = FusionModule(
        img_channels=img_channels, 
        lidar_channels=lidar_channels
    )
    
    # Test case 1: Both modalities present
    img_feats = torch.randn(batch_size, img_channels, H, W)    # (4, 2048, 7, 7)
    lidar_feats = torch.randn(batch_size, lidar_channels)      # (4, 512)
    fused_feats, attention_weights = fusion_module(img_feats, lidar_feats)
    
    print(f"Input shapes:")
    print(f"Image features: {img_feats.shape}")
    print(f"LiDAR features: {lidar_feats.shape}")
    print(f"\nOutput shapes:")
    print(f"Fused features: {fused_feats.shape}")
    if attention_weights is not None:
        print(f"Attention weights: {attention_weights.shape}")  # Should be [4, 7, 7, 1]
    
    # Test case 2: Only image features
    fused_feats, attention_weights = fusion_module(img_feats, None)
    print(f"\nImage only - fused features shape: {fused_feats.shape}")
    if attention_weights is not None:
        print(f"Image only - attention weights shape: {attention_weights.shape}")
    
    # Test case 3: Only LiDAR features
    fused_feats, attention_weights = fusion_module(None, lidar_feats)
    print(f"\nLiDAR only - fused features shape: {fused_feats.shape}")
    if attention_weights is not None:
        print(f"LiDAR only - attention weights shape: {attention_weights.shape}")

def test_fusion_with_dataloader():
    # Initialize backbones and dataloader
    resnet_backbone, pointnet_backbone = initialize_models()
    
    dataloader = get_dataloader(
        root_path="CSCI_files/dev_datakit",
        split='training',
        batch_size=4,
        shuffle=True,
        num_workers=4,
        img_backbone=resnet_backbone,
        lidar_backbone=pointnet_backbone
    )
    
    # Initialize fusion module with matching channels
    fusion_module = FusionModule(
        img_channels=2048,  # ResNet50 output channels
        lidar_channels=512  # PointNet output channels (changed from 1024)
    )
    
    # Process one batch
    batch = next(iter(dataloader.dataloader))
    img_feats = batch['image_features']
    lidar_feats = batch['lidar_features']
    
    print(f"\nDataloader shapes:")
    print(f"Image features: {img_feats.shape}")
    print(f"LiDAR features: {lidar_feats.shape}")
    
    # Run fusion
    fused_feats, attention_weights = fusion_module(img_feats, lidar_feats)
    
    print(f"\nFusion output shapes:")
    print(f"Fused features: {fused_feats.shape}")
    if attention_weights is not None:
        print(f"Attention weights: {attention_weights.shape}")  # Should be [4, 7, 7, 1]

if __name__ == "__main__":
    test_fusion()
    test_fusion_with_dataloader()
