import torch
import torch.nn as nn
import torch.optim as optim
from detection_model import DetectionModel
from kitti_dataloader import get_dataloader, initialize_models
import multiprocessing
from tqdm import tqdm

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)

def train_model(model, train_loader, criterion_cls, criterion_bbox, optimizer, device, num_epochs=10):
    model.train()
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        
        for batch_idx, batch in tqdm(enumerate(train_loader.dataloader), total=len(train_loader.dataloader), desc=f'Epoch {epoch+1}/{num_epochs}'):
            # Move data to device
            img_feats = batch['image_features'].to(device)
            lidar_feats = batch['lidar_features'].to(device)
            targets = batch['targets']
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            bbox_pred, cls_pred = model(img_feats, lidar_feats)
            # print(f'Predictions shapes - bbox: {bbox_pred.shape}, class: {cls_pred.shape}')
            
            # Initialize batch losses
            batch_cls_loss = 0.0
            batch_bbox_loss = 0.0
            total_objects = 0
            
            # Process each sample in the batch
            for sample_idx, target in enumerate(targets):
                # Get predictions for this sample
                sample_bbox_pred = bbox_pred[sample_idx]  # Shape: [H, W, 4]
                sample_cls_pred = cls_pred[sample_idx]    # Shape: [H, W, num_classes]
                
                # Get ground truth
                gt_boxes = target['boxes_3d'].to(device)    # Shape: [num_objects, 4]
                gt_classes = target['classes'].to(device)   # Shape: [num_objects]
                num_objects = target['num_objects']
                
                # Reshape predictions to match targets
                # Flatten spatial dimensions
                flat_bbox_pred = sample_bbox_pred.reshape(-1, 4)  # [H*W, 4]
                flat_cls_pred = sample_cls_pred.reshape(-1, sample_cls_pred.size(-1))  # [H*W, num_classes]
                
                # For each ground truth box, find the best matching predicted box
                for obj_idx in range(num_objects):
                    gt_box = gt_boxes[obj_idx]
                    gt_class = gt_classes[obj_idx]
                    
                    # Compute IoU between this gt_box and all predicted boxes
                    ious = compute_iou(gt_box.unsqueeze(0), flat_bbox_pred)  # [1, H*W]
                    
                    # Get the best matching prediction
                    best_pred_idx = torch.argmax(ious)
                    
                    # Only compute loss if IoU is above threshold
                    # print(ious[0, best_pred_idx])
                    if ious[0, best_pred_idx] > -1:
                        # Classification loss for best matching prediction
                        cls_loss = criterion_cls(
                            flat_cls_pred[best_pred_idx].unsqueeze(0),
                            gt_class.unsqueeze(0)
                        )
                        
                        # Bounding box regression loss
                        bbox_loss = criterion_bbox(
                            flat_bbox_pred[best_pred_idx].unsqueeze(0),
                            gt_box.unsqueeze(0)
                        )
                        
                        batch_cls_loss += cls_loss
                        batch_bbox_loss += bbox_loss
                        total_objects += 1
            
            # Compute average losses for the batch
            if total_objects > 0:
                batch_cls_loss /= total_objects
                batch_bbox_loss /= total_objects
                total_loss = batch_cls_loss + batch_bbox_loss
                
                # Backward pass and optimize
                total_loss.backward()
                optimizer.step()
                
                running_loss += total_loss.item()
                
                if batch_idx % 10 == 0:
                    print(f'Epoch [{epoch+1}/{num_epochs}], '
                          f'Batch [{batch_idx+1}/{len(train_loader.dataloader)}], '
                          f'Loss: {total_loss.item():.4f} '
                          f'(Cls: {batch_cls_loss.item():.4f}, '
                          f'Box: {batch_bbox_loss.item():.4f})')
        
        epoch_loss = running_loss / len(train_loader.dataloader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {epoch_loss:.4f}')
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss,
        }
        torch.save(checkpoint, f'checkpoints/model_epoch_{epoch+1}.pth')
        
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(checkpoint, 'checkpoints/best_model.pth')

def compute_iou(box1, box2):
    """
    Compute IoU between box1 and box2
    box1: [1, 4] tensor (x1, y1, x2, y2)
    box2: [N, 4] tensor (x1, y1, x2, y2)
    Returns: [1, N] tensor of IoU values
    """
    # Get coordinates
    x1 = torch.max(box1[:, 0].unsqueeze(1), box2[:, 0].unsqueeze(0))
    y1 = torch.max(box1[:, 1].unsqueeze(1), box2[:, 1].unsqueeze(0))
    x2 = torch.min(box1[:, 2].unsqueeze(1), box2[:, 2].unsqueeze(0))
    y2 = torch.min(box1[:, 3].unsqueeze(1), box2[:, 3].unsqueeze(0))
    
    # Calculate areas
    intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    box1_area = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    box2_area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    union = box1_area.unsqueeze(1) + box2_area.unsqueeze(0) - intersection
    
    return intersection / (union + 1e-6)  # Add small epsilon to avoid division by zero

def main():
    # Initialize backbone models and get device
    resnet_backbone, pointnet_backbone, device = initialize_models()
    print(f"Using device: {device}")
    
    # Create data loader
    train_loader = get_dataloader(
        root_path="/home/sm2678/csci_739_term_project/CSCI739/data/",
        split='training',
        batch_size=4,
        shuffle=True,
        num_workers=4,
        img_backbone=resnet_backbone,
        lidar_backbone=pointnet_backbone
    )
    
    # Initialize model and move to device
    model = DetectionModel(
        img_channels=2048,
        lidar_channels=512,
        num_classes=9,
        num_anchors=9
    ).to(device)
    
    # Define loss functions and move to device
    criterion_cls = nn.CrossEntropyLoss().to(device)
    criterion_bbox = nn.SmoothL1Loss().to(device)
    
    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    train_model(
        model=model,
        train_loader=train_loader,
        criterion_cls=criterion_cls,
        criterion_bbox=criterion_bbox,
        optimizer=optimizer,
        device=device,
        num_epochs=10
    )

if __name__ == "__main__":
    # Set multiprocessing start method
    multiprocessing.set_start_method('spawn', force=True)
    
    # Set default CUDA device
    if torch.cuda.is_available():
        torch.cuda.set_device(4)  # Use cuda:4 consistently
    
    main()