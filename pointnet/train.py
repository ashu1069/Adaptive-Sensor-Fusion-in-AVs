import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pointnet_cls import PointNetCls
from dataset import PointCloudDataset
import os
import sys
import mlflow


kitti_labels_to_num = {
    'Car': 0,
    'Van': 1,
    'Truck': 2,
    'Pedestrian': 3,
    'Person_sitting': 4,
    'Cyclist': 5,
    'Tram': 6,
    'Misc': 7,
    'DontCare': 8
}

def get_label_from_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        labels = []
        for line in lines:
            lbl = line.strip().split(' ')[0]
            labels.append(kitti_labels_to_num[lbl])
        # return the most common label
        return max(set(labels), key=labels.count)

def train_pointnet(train_points, train_labels, val_points=None, val_labels=None, 
                  num_epochs=250, batch_size=32, learning_rate=0.001, num_classes=40):
    # Initialize MLflow run
    mlflow.start_run()
    mlflow.log_params({
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "num_classes": num_classes
    })

    # Create model save directory
    save_dir = "model_weights"
    os.makedirs(save_dir, exist_ok=True)

    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PointNetCls(num_classes=num_classes).to(device)
    
    # Create datasets and dataloaders
    train_dataset = PointCloudDataset(train_points, train_labels, mode = 'train')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    if val_points is not None:
        val_dataset = PointCloudDataset(val_points, val_labels, mode = 'val')
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Loss and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    # Training loop
    best_val_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_points, batch_labels in train_loader:
            batch_points = batch_points.to(device)
            batch_labels = batch_labels.to(device)
            # print(batch_labels.size(), batch_points.size())
            # Forward pass
            logits, mat3, mat64 = model(batch_points)
            
            # Calculate classification loss
            cls_loss = criterion(logits, batch_labels)
            
            # Feature transform regularization
            identity3 = torch.eye(3).to(device)
            identity64 = torch.eye(64).to(device)
            mat3_diff = torch.matmul(mat3, mat3.transpose(1, 2)) - identity3
            mat64_diff = torch.matmul(mat64, mat64.transpose(1, 2)) - identity64
            reg_loss = (torch.mean(torch.norm(mat3_diff, dim=(1, 2))) + 
                       torch.mean(torch.norm(mat64_diff, dim=(1, 2))))
            
            loss = cls_loss + 0.001 * reg_loss
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            total_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()
        
        epoch_loss = total_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Training Loss: {epoch_loss:.4f}, Training Acc: {epoch_acc:.2f}%')

        # Log metrics to MLflow
        mlflow.log_metrics({
            "train_loss": epoch_loss,
            "train_accuracy": epoch_acc
        }, step=epoch)
        
        # Step the learning rate scheduler
        scheduler.step()
        
        # Validation
        if val_points is not None:
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_points, batch_labels in val_loader:
                    batch_points = batch_points.to(device)
                    batch_labels = batch_labels.to(device)
                    
                    # Forward pass
                    logits, mat3, mat64 = model(batch_points)
                    
                    # Calculate loss
                    cls_loss = criterion(logits, batch_labels)
                    identity3 = torch.eye(3).to(device)
                    identity64 = torch.eye(64).to(device)
                    mat3_diff = torch.matmul(mat3, mat3.transpose(1, 2)) - identity3
                    mat64_diff = torch.matmul(mat64, mat64.transpose(1, 2)) - identity64
                    reg_loss = (torch.mean(torch.norm(mat3_diff, dim=(1, 2))) + 
                               torch.mean(torch.norm(mat64_diff, dim=(1, 2))))
                    loss = cls_loss + 0.001 * reg_loss
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(logits.data, 1)
                    val_total += batch_labels.size(0)
                    val_correct += (predicted == batch_labels).sum().item()
            
            val_loss = val_loss / len(val_loader)
            val_acc = 100 * val_correct / val_total
            print(f'Validation Loss: {val_loss:.4f}, Validation Acc: {val_acc:.2f}%')

            # Log validation metrics
            mlflow.log_metrics({
                "val_loss": val_loss,
                "val_accuracy": val_acc
            }, step=epoch)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                model_path = os.path.join(save_dir, 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                }, model_path)
                mlflow.log_artifact(model_path)
        
        print('-' * 60)

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc if val_points is not None else None,
            }, checkpoint_path)
    
    mlflow.end_run()
    return model

def test_pointnet(model, test_points, test_labels, batch_size=32):
    """
    Test the trained PointNet model
    """
    device = next(model.parameters()).device
    model.eval()
    
    test_dataset = PointCloudDataset(test_points, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_points, batch_labels in test_loader:
            batch_points = batch_points.to(device)
            batch_labels = batch_labels.to(device)
            
            logits, _, _ = model(batch_points)
            preds = torch.exp(logits).max(1)[1]
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    test_acc = 100 * (all_preds == all_labels).mean()
    
    print(f'Test Accuracy: {test_acc:.2f}%')
    return test_acc

# Example usage:
if __name__ == "__main__":
    import numpy as np
    
    # Create dummy data for demonstration
    num_points = 1024
    num_classes = 9
    num_samples = 100
    
    # Get all .bin files
    data_dir = "/home/sm2678/csci_739_term_project/CSCI739/data/"
    calib_file = "CSCI739/samples/calib.yaml"  # Optional calibration file
    
    # Get all .bin files
    # get all file names in velodyne/training/velodyne
    
    velo_data_dir = os.path.join(data_dir, 'velodyne', 'training', 'velodyne')
    point_cloud_files = [f.split('.')[0] for f in os.listdir(velo_data_dir) if f.endswith('.bin')]
    point_cloud_files.sort()
    # print(point_cloud_files)
    
    # get all file names in labels/training/label_2
    label_data_dir = os.path.join(data_dir, 'labels', 'training', 'label_2')
    label_files = [f.split('.')[0] for f in os.listdir(label_data_dir) if f.endswith('.txt')]
    label_files.sort()
    labels = []
    for file in label_files:
        lbls = get_label_from_file(os.path.join(label_data_dir, file + '.txt'))
        labels.append(lbls)
    
    # get all file named from velodyne/validation/velodyne and labels/validation/label_2
    val_velo_data_dir = os.path.join(data_dir, 'velodyne', 'validation', 'velodyne')
    val_point_cloud_files = [f.split('.')[0] for f in os.listdir(val_velo_data_dir) if f.endswith('.bin')]
    val_point_cloud_files.sort()
    
    val_label_data_dir = os.path.join(data_dir, 'labels', 'validation', 'label_2')
    val_label_files = [f.split('.')[0] for f in os.listdir(val_label_data_dir) if f.endswith('.txt')]
    val_label_files.sort()
    
    val_labels = []
    for file in val_label_files:
        lbls = get_label_from_file(os.path.join(val_label_data_dir, file + '.txt'))
        val_labels.append(lbls)
    
    # # Split into train/val
    # train_size = int(0.8 * len(point_cloud_files))
    # train_points = point_cloud_files[:train_size]
    # train_labels = labels[:train_size]
    # val_points = point_cloud_files[train_size:]
    # val_labels = labels[train_size:]
    # Train the model
    model = train_pointnet(
        train_points=point_cloud_files,
        train_labels=labels,
        val_points=val_point_cloud_files,
        val_labels=val_labels,
        num_epochs=200,
        batch_size=32,
        learning_rate=0.001,
        num_classes=num_classes
    )
    
    # Test the model
    test_acc = test_pointnet(model, val_point_cloud_files, val_labels)