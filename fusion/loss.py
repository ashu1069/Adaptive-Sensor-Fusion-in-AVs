import torch
import torch.nn as nn

class DetectionLoss(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        # Loss functions
        self.bce = nn.BCEWithLogitsLoss(reduction='sum')
        self.cls_loss = nn.CrossEntropyLoss(reduction='sum')
        self.bbox_loss = nn.MSELoss(reduction='sum')
        
        # Loss weights
        self.lambda_box = 0.05
        self.lambda_obj = 1.0
        self.lambda_cls = 0.5

    def forward(self, predictions, targets):
        """
        Args:
            predictions (List[Tensor]): List of prediction tensors for each scale
                Shape of each tensor: [B, H*W, 6] where 6 = [class_idx, x, y, w, h, confidence]
            targets (List[Tensor]): List of target tensors for each scale
                Shape of each tensor: [M, 6] where 6 = [batch_idx, class_id, x, y, w, h]
        Returns:
            Tuple[Tensor, Tensor]: (total_loss, individual_losses)
        """
        device = predictions[0].device
        # Initialize losses as scalars on the correct device
        lbox = torch.tensor(0., device=device, requires_grad=True)
        lobj = torch.tensor(0., device=device, requires_grad=True)
        lcls = torch.tensor(0., device=device, requires_grad=True)
        
        total_targets = 0

        # Process each scale
        for scale_idx, (pred, target) in enumerate(zip(predictions, targets)):
            if len(pred) == 0 or len(target) == 0:
                continue

            batch_size = pred.shape[0]
            
            # Process each batch
            for b in range(batch_size):
                pred_batch = pred[b]  # [H*W, 6]
                
                # Get targets for this batch
                if len(target.shape) == 3:  # [B, N, 6]
                    target_batch = target[b]
                else:  # [M, 6]
                    target_batch = target[target[:, 0] == b]  # Filter by batch index
                
                if len(target_batch) == 0:
                    continue
                    
                total_targets += len(target_batch)
                
                # Calculate IoU between predictions and targets
                pred_boxes = pred_batch[:, 1:5]  # [H*W, 4] (x,y,w,h)
                target_boxes = target_batch[:, 2:6]  # [M, 4] (x,y,w,h)
                
                ious = box_iou(pred_boxes, target_boxes)  # [H*W, M]
                
                # Get best matching prediction for each target
                best_ious, best_idx = ious.max(dim=0)  # [M]
                
                # Box regression loss for matched pairs
                matched_preds = pred_batch[best_idx]  # [M, 6]
                # Add losses directly without assignment
                lbox = lbox + self.bbox_loss(matched_preds[:, 1:5], target_boxes)
                
                # Objectness loss
                obj_targets = torch.zeros_like(pred_batch[:, 5])
                max_iou_per_pred, _ = ious.max(dim=1)  # [H*W]
                obj_targets[max_iou_per_pred > 0.5] = 1.0
                lobj = lobj + self.bce(pred_batch[:, 5], obj_targets)
                
                # Classification loss
                cls_targets = target_batch[:, 1]  # class indices
                lcls = lcls + self.cls_loss(matched_preds[:, 0], cls_targets)

        # Normalize and combine losses
        if total_targets > 0:
            lbox = (lbox / total_targets) * self.lambda_box
            lobj = (lobj / total_targets) * self.lambda_obj
            lcls = (lcls / total_targets) * self.lambda_cls
            loss = lbox + lobj + lcls
        else:
            # Handle case with no targets
            loss = lbox + lobj + lcls  # Will be zero but maintains grad

        return loss


def box_iou(boxes1, boxes2, eps=1e-5):
    """Calculate IoU between two sets of boxes"""
    # Ensure both inputs are on the same device
    device = boxes1.device
    boxes1 = boxes1.to(device)
    boxes2 = boxes2.to(device)
    
    # Convert to x1, y1, x2, y2 format
    b1_x1 = boxes1[:, 0] - boxes1[:, 2] / 2
    b1_y1 = boxes1[:, 1] - boxes1[:, 3] / 2
    b1_x2 = boxes1[:, 0] + boxes1[:, 2] / 2
    b1_y2 = boxes1[:, 1] + boxes1[:, 3] / 2
    
    b2_x1 = boxes2[:, 0] - boxes2[:, 2] / 2
    b2_y1 = boxes2[:, 1] - boxes2[:, 3] / 2
    b2_x2 = boxes2[:, 0] + boxes2[:, 2] / 2
    b2_y2 = boxes2[:, 1] + boxes2[:, 3] / 2
    
    # Get intersection area
    inter_x1 = torch.max(b1_x1[:, None], b2_x1)
    inter_y1 = torch.max(b1_y1[:, None], b2_y1)
    inter_x2 = torch.min(b1_x2[:, None], b2_x2)
    inter_y2 = torch.min(b1_y2[:, None], b2_y2)
    
    inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
    inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
    inter_area = inter_w * inter_h
    
    # Calculate union area
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union_area = b1_area[:, None] + b2_area - inter_area
    
    return inter_area / (union_area + eps)

# def test_detection_loss():
#     """
#     Test function to verify DetectionLoss with various input shapes
#     """
#     # Initialize loss function
#     num_classes = 8
#     loss_fn = DetectionLoss(num_classes=num_classes)
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     loss_fn = loss_fn.to(device)

#     # Test cases with different shapes
#     test_cases = [
#         # Single scale, single batch
#         {
#             'pred': [torch.randn(1, 13*13, 6).to(device)],  # [B, H*W, 6] where 6 = [class_id, x, y, w, h, obj]
#             'target': [torch.randn(5, 6).to(device)],  # [M, 6] where 6 = [class_id, x, y, w, h, scale_idx]
#             'name': "Single scale, single batch"
#         },
#         # Multiple scales, single batch
#         {
#             'pred': [
#                 torch.randn(1, 13*13, 6).to(device),  # 13x13 grid
#                 torch.randn(1, 26*26, 6).to(device),  # 26x26 grid
#             ],
#             'target': [
#                 torch.randn(3, 6).to(device),  # 3 objects
#                 torch.randn(4, 6).to(device),  # 4 objects
#             ],
#             'name': "Multiple scales, single batch"
#         },
#         # Multiple scales, multiple batches
#         {
#             'pred': [
#                 torch.randn(2, 13*13, 6).to(device),  # Batch size 2
#                 torch.randn(2, 26*26, 6).to(device),
#             ],
#             'target': [
#                 torch.randn(6, 6).to(device),  # 6 total objects
#                 torch.randn(8, 6).to(device),  # 8 total objects
#             ],
#             'name': "Multiple scales, multiple batches"
#         }
#     ]

#     print("\nRunning DetectionLoss tests...")
#     for test_case in test_cases:
#         try:
#             print(f"\nTesting: {test_case['name']}")
#             total_loss, loss_dict = loss_fn(test_case['pred'], test_case['target'])
            
#             print(f"Total loss: {total_loss.item():.4f}")
#             print("Individual losses:")
#             for k, v in loss_dict.items():
#                 if isinstance(v, torch.Tensor):
#                     print(f"  {k}: {v.item():.4f}")
#                 else:
#                     print(f"  {k}: {v:.4f}")
#             print("✓ Test passed")
            
#         except Exception as e:
#             print(f"✗ Test failed: {str(e)}")

# if __name__ == "__main__":
#     test_detection_loss()
