#File:detr/attentions.py

import torch
import torch.nn as nn

class ContextGuidedFusion(nn.Module):
    def __init__(self, img_channels, lidar_channels, hidden_dim):
        super().__init__()

        # Modality-specific feature projection
        self.img_proj = nn.Conv2d(img_channels, hidden_dim, kernel_size=1)
        self.lidar_proj = nn.Conv2d(lidar_channels, hidden_dim, kernel_size=1)

        # Cross-attention layers
        self.attn_img_to_lidar = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.attn_lidar_to_img = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)

        # Output projection
        self.fusion_proj = nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=1)

        # Normalization and activation
        self.norm = nn.LayerNorm(hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, img_feat, lidar_feat):
        # Project to common hidden_dim space
        img_feat_proj = self.img_proj(img_feat)  # (B, C, H, W)
        lidar_feat_proj = self.lidar_proj(lidar_feat)  # (B, C, H, W)

        # Flatten spatial dimensions for attention
        B, C, H, W = img_feat_proj.shape
        img_feat_flat = img_feat_proj.flatten(2).permute(0, 2, 1)
        lidar_feat_flat = lidar_feat_proj.flatten(2).permute(0, 2, 1)

        # Apply cross-attention
        img_attn, _ = self.attn_img_to_lidar(img_feat_flat, lidar_feat_flat, lidar_feat_flat)
        lidar_attn, _ = self.attn_lidar_to_img(lidar_feat_flat, img_feat_flat, img_feat_flat)

        # Reshape back to (B, C, H, W)
        img_attn = img_attn.permute(0, 2, 1).reshape(B, C, H, W)

        lidar_H, lidar_W = lidar_feat_proj.shape[2:]
        lidar_attn = lidar_attn.permute(0, 2, 1).reshape(B, C, lidar_H, lidar_W)
        

        # Interpolate lidar features to match image spatial dimensions
        lidar_attn = nn.functional.interpolate(lidar_attn, size=(H, W), mode='bilinear', align_corners=False)

        # Fuse features
        fused_features = torch.cat([img_attn, lidar_attn], dim=1)
        fused_features = self.fusion_proj(fused_features)

        return self.relu(self.norm(fused_features))

if __name__ == "__main__":
    img_feat = torch.randn(1, 2048, 39, 12)
    lidar_feat = torch.randn(1, 1024,1,1)
    fusion = ContextGuidedFusion(img_feat.shape[1], lidar_feat.shape[1], 256)
    fused_feat = fusion(img_feat, lidar_feat)
    print(fused_feat.shape)