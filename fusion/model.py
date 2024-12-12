import torch
import torch.nn as nn
import torchvision.ops as ops
import math
from loss import DetectionLoss
from utils import post_process_detections

class ElementWiseFusion(nn.Module):
    def __init__(self, lidar_dim=1024, image_dim=1024, fusion_type='adaptive'):
        super().__init__()
        self.fusion_type = fusion_type
        self.img_proj = nn.Conv2d(image_dim, lidar_dim, 1)
        
        # Channel attention with spatial dimensions
        self.channel_attention = nn.Sequential(
            nn.Conv2d(lidar_dim, lidar_dim // 4, 1),
            nn.ReLU(),
            nn.Conv2d(lidar_dim // 4, lidar_dim, 1),
            nn.Sigmoid()
        )
        
        # Modified gate network to work with spatial dimensions
        self.gate_network = nn.Sequential(
            nn.Conv2d(lidar_dim * 2, 512, 1),
            nn.ReLU(),
            nn.Conv2d(512, 2, 1),
            nn.Softmax(dim=1)
        )
        
    def forward(self, lidar_features, image_features):
        # Remove extra dimension if present
        B, C, H, W = image_features.shape
        
        # Get spatial dimensions from image features
        lidar_features = lidar_features.view(B, -1, 1, 1)
        lidar_features = lidar_features.expand(-1, -1, H, W)
        
        # Project image features
        img_feat = self.img_proj(image_features)
        
        # Apply channel attention
        channel_weights = self.channel_attention(lidar_features)
        lidar_features = lidar_features * channel_weights
        
        if self.fusion_type == 'adaptive':
            # Concatenate along channel dimension
            combined = torch.cat([lidar_features, img_feat], dim=1)
            weights = self.gate_network(combined)
            
            add_fusion = lidar_features + img_feat
            mul_fusion = lidar_features * img_feat
            
            return weights[:, 0:1] * add_fusion + weights[:, 1:] * mul_fusion
        elif self.fusion_type == 'add':
            return lidar_features + img_feat
        else:  # multiply
            return lidar_features * img_feat
        
class CrossAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        
    def forward(self, x1, x2):
        q = self.query(x1)
        k = self.key(x2)
        v = self.value(x2)
        
        attention = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        attention = torch.softmax(attention, dim=-1)
        
        return torch.matmul(attention, v)

class FusionFPN(nn.Module):
    def __init__(self, in_channels=[64, 128, 256]):
        super().__init__()
        self.in_channels = in_channels
        
        # Cross-modal attention for each scale
        self.fusion_layers = nn.ModuleList([
            ElementWiseFusion(lidar_dim=1024, image_dim=c)
            for c in in_channels
        ])
        
        # FPN layers
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(c, 256, 1) for c in in_channels
        ])
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(256, 256, 3, padding=1) for _ in in_channels
        ])
        
        # Add cross-attention
        self.cross_attention = nn.ModuleList([
            CrossAttention(1024) for _ in in_channels
        ])
        
    def forward(self, lidar_features, image_features):
        # Apply cross-modal attention at each scale
        attented_features = []
        for attention, img_feat in zip(self.fusion_layers, image_features):
            att_feat = attention(lidar_features, img_feat)
            attented_features.append(att_feat)
            
        # Build FPN top-down pathway
        laterals = [
            lateral_conv(feat)
            for lateral_conv, feat in zip(self.lateral_convs, image_features)
        ]
        
        # Top-down pathway
        fpn_features = [laterals[-1]]
        for i in range(len(laterals)-2, -1, -1):
            top = torch.nn.functional.interpolate(
                fpn_features[-1], size=laterals[i].shape[-2:], mode='nearest'
            )
            fpn_features.append(laterals[i] + top)
        fpn_features = fpn_features[::-1]
        
        # Apply final convolutions
        outputs = [
            fpn_conv(feat)
            for fpn_conv, feat in zip(self.fpn_convs, fpn_features)
        ]
        
        return outputs, attented_features

class DetectionHead(nn.Module):
    def __init__(self, in_channels=256, num_classes=8):
        super().__init__()
        self.num_classes = num_classes
        # Each cell predicts: 4 box coords + 1 objectness + num_classes
        self.num_outputs = 5 + num_classes
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, self.num_outputs, 1)
        )
        
    def forward(self, x):
        return self.conv(x)

class MultiModalDetector(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
        self.fusion_fpn = FusionFPN()
        self.detection_heads = nn.ModuleList([
            DetectionHead(256, num_classes) for _ in range(3)
        ])
        self.detection_loss = DetectionLoss(num_classes)
        
    def forward(self, batch):
        # Unpack the batch data
        lidar_features = batch['lidar_features']  # Shape: (B, 1024)
        image_features = batch['image_features']  # List of 3 tensors with different spatial dimensions [B, C, H, W]
        targets = batch.get('targets', None)  # List of target tensors for each image in batch

        fpn_features, attended_features = self.fusion_fpn(lidar_features, image_features)
        # Get predictions from each scale
        detections = []
        for feat, head in zip(fpn_features, self.detection_heads):
            pred = head(feat)
            B, _, H, W = pred.shape
            pred = pred.permute(0, 2, 3, 1).contiguous()
            
            # Create grid cells
            grid_y, grid_x = torch.meshgrid(
                torch.arange(H, device=pred.device),
                torch.arange(W, device=pred.device),
                indexing='ij'
            )
            grid = torch.stack((grid_x, grid_y), 2).float()
            
            # Process predictions
            box_xy = torch.sigmoid(pred[..., :2])  # Center coordinates
            box_wh = torch.exp(pred[..., 2:4])     # Width and height
            obj_conf = torch.sigmoid(pred[..., 4:5])  # Objectness score
            class_probs = torch.sigmoid(pred[..., 5:])  # Class probabilities
            
            # Convert predictions to absolute coordinates
            box_xy = (box_xy + grid) / torch.tensor([W, H], device=pred.device)
            box_wh = box_wh / torch.tensor([W, H], device=pred.device)
            
            # Get the class with highest probability and its index
            class_scores, class_idx = class_probs.max(dim=-1, keepdim=True)  # [B, H, W, 1]
            
            # Combine predictions in target format: [class_idx, x, y, w, h, confidence]
            detection = torch.cat((
                class_idx,             # Class index
                box_xy,                # Normalized center coordinates (x,y)
                box_wh,                # Normalized dimensions (w,h)
                obj_conf * class_scores  # Final confidence (objectness * class confidence)
            ), -1)
            
            # Reshape to [B, H*W, 6]
            detection = detection.view(B, H*W, 6)
            detections.append(detection)
        
        if self.training and targets is not None:
            loss = self.detection_loss(detections, targets)
            return {
                'loss': loss,
                'detections': detections
            }
        else:
            processed_detections = post_process_detections(detections)
            return {
                'loss': None,
                'detections': processed_detections
            }
'''
def test_element_wise_fusion():
    # Create sample inputs
    batch_size = 2
    lidar_dim = 1024
    image_dim = 256
    height, width = 20, 20

    # Create random tensors
    lidar_features = torch.randn(batch_size, lidar_dim)  # [B, C]
    image_features = torch.randn(batch_size, image_dim, height, width)  # [B, C, H, W]

    # Test all fusion types
    fusion_types = ['adaptive', 'add', 'multiply']
    
    for fusion_type in fusion_types:
        print(f"\nTesting {fusion_type} fusion:")
        model = ElementWiseFusion(lidar_dim=lidar_dim, image_dim=image_dim, fusion_type=fusion_type)
        output = model(lidar_features, image_features)
        
        print(f"Input lidar features shape: {lidar_features.shape}")
        print(f"Input image features shape: {image_features.shape}")
        print(f"Output features shape: {output.shape}")

def test_fusion_fpn():
    # Create sample inputs
    batch_size = 2
    lidar_dim = 1024
    image_dims = [64, 128, 256]  # Channels for different scales
    
    # Create random lidar features
    lidar_features = torch.randn(batch_size, lidar_dim)  # [B, C]
    
    # Create multi-scale image features
    # Assuming each subsequent level has half the spatial dimensions
    image_features = [
        torch.randn(batch_size, dim, 80//(2**i), 80//(2**i))  
        for i, dim in enumerate(image_dims)
    ]
    
    # Initialize model
    model = FusionFPN(in_channels=image_dims)
    
    # Forward pass
    fpn_features, attended_features = model(lidar_features, image_features)
    
    print("\nTesting FusionFPN:")
    print(f"Input lidar features shape: {lidar_features.shape}")
    print("\nInput image features shapes:")
    for i, feat in enumerate(image_features):
        print(f"Level {i}: {feat.shape}")
    
    print("\nOutput FPN features shapes:")
    for i, feat in enumerate(fpn_features):
        print(f"Level {i}: {feat.shape}")
        
    print("\nOutput attended features shapes:")
    for i, feat in enumerate(attended_features):
        print(f"Level {i}: {feat.shape}")

def test_detection_head():
    # Create sample inputs
    batch_size = 2
    in_channels = 256
    num_classes = 8
    feature_size = 80  # Spatial dimensions of the feature map
    
    # Create random feature map
    features = torch.randn(batch_size, in_channels, feature_size, feature_size)
    
    # Initialize model
    model = DetectionHead(in_channels=in_channels, num_classes=num_classes)
    
    # Forward pass
    output = model(features)
    
    print("\nTesting DetectionHead:")
    print(f"Input features shape: {features.shape}")
    print(f"Output shape: {output.shape}")
    
    # Verify output dimensions
    expected_channels = 5 + num_classes  # 4 box coords + 1 objectness + num_classes
    assert output.shape == (batch_size, expected_channels, feature_size, feature_size), \
        f"Expected shape {(batch_size, expected_channels, feature_size, feature_size)}, got {output.shape}"
    print("Output shape verification passed!")

def test_multimodal_detector():
    # Create sample inputs
    batch_size = 2
    lidar_dim = 1024
    image_dims = [64, 128, 256]  # Channels for different scales
    num_classes = 8
    
    # Create sample batch dictionary
    batch = {
        'lidar_features': torch.randn(batch_size, lidar_dim),  # [B, C]
        'image_features': [
            torch.randn(batch_size, dim, 80//(2**i), 80//(2**i))  
            for i, dim in enumerate(image_dims)
        ],
        'targets': [
            torch.tensor([
                [0, 0.5, 0.5, 0.2, 0.2, 0],  # Object assigned to scale 0
                [1, 0.3, 0.7, 0.1, 0.1, 1],  # Object assigned to scale 1
            ]).float(),
            torch.tensor([
                [2, 0.4, 0.6, 0.15, 0.15, 2],  # Object assigned to scale 2
            ]).float()
        ]
    }
    
    # Initialize model
    model = MultiModalDetector(num_classes=num_classes)
    
    print("\nTesting MultiModalDetector:")
    print(f"Input lidar features shape: {batch['lidar_features'].shape}")
    print("\nInput image features shapes:")
    for i, feat in enumerate(batch['image_features']):
        print(f"Level {i}: {feat.shape}")
    
    # Test training mode
    print("\nTesting training mode:")
    model.train()
    train_output = model(batch)
    print("Training outputs:")
    print(f"Loss: {train_output['loss']}")
    print(f"Loss components: {train_output['loss_components']}")
    print("\nDetection shapes:")
    for i, det in enumerate(train_output['detections']):
        print(f"Level {i}: {det.shape}")  # Should be [B, H*W, 6]
    
    # Test inference mode
    print("\nTesting inference mode:")
    model.eval()
    batch.pop('targets')  # Remove targets for inference
    with torch.no_grad():
        inference_output = model(batch)
    print("Inference outputs:")
    print(f"Number of processed detections: {len(inference_output['detections'])}")
    for i, dets in enumerate(inference_output['detections']):
        if len(dets) > 0:  # If any detections exist
            print(f"Level {i} detections shape: {dets.shape}")

if __name__ == '__main__':
    test_element_wise_fusion()
    test_fusion_fpn()
    test_detection_head()
    test_multimodal_detector()
'''