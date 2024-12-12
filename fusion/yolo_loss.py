import torch
import torch.nn as nn
import torch.nn.functional as F

class YOLOv8Loss(nn.Module):
    def __init__(self, num_classes, balance_weights=[0.5, 1.0, 2.0]):
        """
        Multi-scale loss for YOLOv8 anchor-free detection
        
        Args:
            num_classes (int): Number of object classes
            balance_weights (list): Weights for different scale losses
        """
        super().__init__()
        self.num_classes = num_classes
        self.balance_weights = balance_weights
        
        # Focal loss parameters
        self.alpha = 0.25
        self.gamma = 2.0
        
    def bbox_iou(self, pred_boxes, true_boxes, eps=1e-7):
        """
        Calculate IoU between predicted and ground truth boxes
        
        Args:
            pred_boxes (torch.Tensor): Predicted bounding boxes [N, 4]
            true_boxes (torch.Tensor): Ground truth boxes [M, 4]
        
        Returns:
            torch.Tensor: IoU matrix
        """
        # Convert [x,y,w,h] to [x1,y1,x2,y2]
        pred_boxes = self._xywh2xyxy(pred_boxes)
        true_boxes = self._xywh2xyxy(true_boxes)
        
        # Calculate intersection
        b1_x1, b1_y1, b1_x2, b1_y2 = pred_boxes[:, 0], pred_boxes[:, 1], pred_boxes[:, 2], pred_boxes[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = true_boxes[:, 0], true_boxes[:, 1], true_boxes[:, 2], true_boxes[:, 3]
        
        inter_rect_x1 = torch.max(b1_x1[:, None], b2_x1)
        inter_rect_y1 = torch.max(b1_y1[:, None], b2_y1)
        inter_rect_x2 = torch.min(b1_x2[:, None], b2_x2)
        inter_rect_y2 = torch.min(b1_y2[:, None], b2_y2)
        
        inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, min=0) * \
                     torch.clamp(inter_rect_y2 - inter_rect_y1, min=0)
        
        pred_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        true_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
        
        union_area = pred_area[:, None] + true_area - inter_area
        iou = inter_area / (union_area + eps)
        
        return iou
    
    def _xywh2xyxy(self, x):
        """Convert [x,y,w,h] to [x1,y1,x2,y2]"""
        y = x.clone()
        y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
        y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
        y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
        y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
        return y
    
    def focal_classification_loss(self, pred_cls, true_cls):
        """
        Focal loss for class predictions
        
        Args:
            pred_cls (torch.Tensor): Predicted class probabilities
            true_cls (torch.Tensor): True class indices
        
        Returns:
            torch.Tensor: Focal classification loss
        """
        # One-hot encode true classes
        true_cls_one_hot = F.one_hot(true_cls.long(), self.num_classes).float()
        
        # Compute focal loss
        pt = torch.where(true_cls_one_hot == 1, pred_cls, 1 - pred_cls)
        focal_loss = -self.alpha * (1 - pt)**self.gamma * torch.log(pt + 1e-7)
        
        return focal_loss.mean()
    
    def bbox_loss(self, pred_boxes, true_boxes):
        """
        Complete bounding box regression loss
        
        Args:
            pred_boxes (torch.Tensor): Predicted bounding boxes
            true_boxes (torch.Tensor): Ground truth boxes
        
        Returns:
            torch.Tensor: Bounding box regression loss
        """
        # IoU Loss
        iou = self.bbox_iou(pred_boxes, true_boxes)
        iou_loss = 1 - iou.mean()
        
        # CIoU or DIoU could be implemented here for more advanced loss
        
        # Smooth L1 Loss for box coordinates
        smooth_l1_loss = F.smooth_l1_loss(pred_boxes, true_boxes)
        
        return iou_loss + smooth_l1_loss
    
    def forward(self, predictions, targets):
        """
        Multi-scale loss computation
        """
        total_loss = 0
        
        for scale_idx, pred in enumerate(predictions):
            # Get batch size and grid size from predictions
            batch_size, num_anchors, _ = pred.shape
            
            # Reshape predictions to match target format
            pred_cls = pred[..., :self.num_classes]
            pred_boxes = pred[..., self.num_classes:self.num_classes+4]
            
            # Create a mask for valid targets
            valid_mask = targets[..., -1] > 0
            valid_targets = targets[valid_mask]
            
            if len(valid_targets) > 0:
                # Match predictions to targets using IoU
                pred_boxes_reshaped = pred_boxes.view(-1, 4)  # Flatten predictions
                target_boxes = valid_targets[..., 1:5]
                ious = self.bbox_iou(pred_boxes_reshaped, target_boxes)  # Calculate IoU matrix
                
                # Assign each target to the prediction with highest IoU
                best_pred_idx = ious.max(dim=0)[1]  # Get indices of best predictions
                
                # Get corresponding predictions
                valid_pred_cls = pred_cls.view(-1, self.num_classes)[best_pred_idx]
                valid_pred_boxes = pred_boxes_reshaped[best_pred_idx]
                
                # Classification loss
                cls_loss = self.focal_classification_loss(valid_pred_cls, valid_targets[..., 0])
                
                # Bounding box loss
                bbox_loss = self.bbox_loss(valid_pred_boxes, target_boxes)
            else:
                cls_loss = torch.tensor(0.0, device=pred.device)
                bbox_loss = torch.tensor(0.0, device=pred.device)
            
            # Confidence loss (optional, depending on your implementation)
            conf_pred = pred[..., -1].view(-1)  # Flatten predictions
            conf_target = torch.zeros_like(conf_pred)  # Initialize all to background
            
            if len(valid_targets) > 0:
                # Set confidence=1 for the best matching predictions
                conf_target[best_pred_idx] = 1
            
            conf_loss = F.binary_cross_entropy_with_logits(conf_pred, conf_target)
            
            # Compute weighted loss for this scale
            scale_loss = (
                cls_loss * self.balance_weights[scale_idx] + 
                bbox_loss * self.balance_weights[scale_idx] + 
                conf_loss * self.balance_weights[scale_idx]
            )
            
            total_loss += scale_loss
        
        return total_loss / len(predictions)

# # Example usage
# num_classes = 8  # COCO dataset
# loss_fn = YOLOv8Loss(num_classes)

# # Simulated predictions and targets with valid class indices
# predictions = [
#     torch.randn(2, 1600, num_classes + 5),  # 40x40 scale: classes + 4 bbox coords + 1 conf
#     torch.randn(2, 1600, num_classes + 5),  # 40x40 scale
#     torch.randn(2, 400, num_classes + 5)    # 20x20 scale
# ]

# # Generate targets with valid class indices (between 0 and num_classes-1)
# targets = torch.zeros(2, 10, 6)  # batch_size, num_objects, (class_idx + 4 bbox coords + conf)
# targets[..., 0] = torch.randint(0, num_classes, (2, 10))  # Valid class indices
# targets[..., 1:] = torch.rand(2, 10, 5)  # Random bbox coordinates and confidence

# loss = loss_fn(predictions, targets)
# print(loss)