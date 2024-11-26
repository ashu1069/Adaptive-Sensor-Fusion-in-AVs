import torch
import torch.nn as nn

class DetectionHead(nn.Module):
    def __init__(self, in_channels=2048, num_classes=9):  # 9 classes from KITTI
        super().__init__()
        
        # Shared feature extraction
        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Box regression head
        self.box_head = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 4)  # 4 values for bbox (x1, y1, x2, y2)
        )
        
        # Classification head
        self.cls_head = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )

    def forward(self, x, attention_weights=None):
        """
        Args:
            x: Fused features (B, 2048, 7, 7)
            attention_weights: Optional attention weights (B, 7, 7, 1)
        Returns:
            dict containing:
                boxes: (B, 4) - normalized coordinates (x1, y1, x2, y2)
                scores: (B, num_classes) - class logits
        """
        # Apply shared features
        features = self.shared_conv(x)  # (B, 256, 7, 7)
        
        # If attention weights provided, apply them
        if attention_weights is not None:
            # Reshape attention weights to (B, 1, 7, 7)
            attention = attention_weights.permute(0, 3, 1, 2)
            features = features * attention
            
        # Get predictions
        boxes = self.box_head(features)  # (B, 4)
        scores = self.cls_head(features)  # (B, num_classes)
        
        # Normalize box coordinates to [0, 1] using sigmoid
        boxes = torch.sigmoid(boxes)
        
        return {
            'boxes': boxes,
            'scores': scores
        }

def detection_loss(predictions, targets, cls_weight=1.0, box_weight=1.0):
    batch_size = predictions['scores'].shape[0]
    
    # For each batch item, take the first target
    batch_classes = torch.stack([target['classes'][0] for target in targets])
    
    # Get and normalize target boxes to [0, 1]
    batch_boxes = torch.stack([target['boxes_2d'][0] for target in targets])
    batch_boxes = batch_boxes.float()  # Ensure float type
    
    # Normalize target boxes if they're not already in [0, 1]
    # Assuming original coordinates are in pixels, e.g., [0, image_width] and [0, image_height]
    # You'll need to adjust these values based on your actual image dimensions
    image_width = 1242.0  # KITTI image width
    image_height = 375.0  # KITTI image height
    
    # Normalize x coordinates
    batch_boxes[:, [0, 2]] = batch_boxes[:, [0, 2]] / image_width
    # Normalize y coordinates
    batch_boxes[:, [1, 3]] = batch_boxes[:, [1, 3]] / image_height
    
    # Clamp values to ensure they're in [0, 1]
    batch_boxes = torch.clamp(batch_boxes, 0, 1)
    
    # Calculate classification loss
    cls_loss = nn.CrossEntropyLoss()(
        predictions['scores'],
        batch_classes
    )
    
    # Calculate bounding box regression loss
    box_loss = nn.SmoothL1Loss()(
        predictions['boxes'],
        batch_boxes
    )
    
    # Adjust box_weight to balance the losses
    box_weight = 0.1  # Reduced from 1.0 to 0.1 to balance the loss values
    
    # Combine losses
    total_loss = cls_weight * cls_loss + box_weight * box_loss
    
    return {
        'total_loss': total_loss,
        'cls_loss': cls_loss,
        'box_loss': box_loss
    }
