import torch
import torch.nn as nn
import torch.optim as optim
from detection_model import DetectionModel
from kitti_dataloader import get_dataloader, initialize_models

def train_model(model, train_loader, criterion_cls, criterion_bbox, optimizer, device, num_epochs=10):
    model.train()
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader.dataloader):
            # Move data to device
            img_feats = batch['image_features'].to(device)
            lidar_feats = batch['lidar_features'].to(device)
            targets = batch['targets']
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            bbox_pred, cls_pred = model(img_feats, lidar_feats)
            
            # Calculate losses
            cls_loss = criterion_cls(cls_pred, targets['classes'])
            bbox_loss = criterion_bbox(bbox_pred, targets['boxes_2d'])
            total_loss = cls_loss + bbox_loss
            
            # Backward pass and optimize
            total_loss.backward()
            optimizer.step()
            
            running_loss += total_loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], '
                      f'Batch [{batch_idx+1}/{len(train_loader.dataloader)}], '
                      f'Loss: {total_loss.item():.4f}')
        
        epoch_loss = running_loss / len(train_loader.dataloader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {epoch_loss:.4f}')

def main():
    # Initialize device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize backbone models
    resnet_backbone, pointnet_backbone = initialize_models()
    
    # Create data loader
    train_loader = get_dataloader(
        root_path="path/to/kitti",
        split='training',
        batch_size=4,
        img_backbone=resnet_backbone,
        lidar_backbone=pointnet_backbone
    )
    
    # Initialize model
    model = DetectionModel(
        img_channels=2048,  # ResNet50 output channels
        lidar_channels=1024,  # PointNet output channels
        num_classes=9,  # KITTI classes
        num_anchors=9
    ).to(device)
    
    # Define loss functions
    criterion_cls = nn.CrossEntropyLoss()
    criterion_bbox = nn.SmoothL1Loss()
    
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
    main()