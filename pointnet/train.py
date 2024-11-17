import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from .pointnet_cls import PointNetCls
from .dataset import PointCloudDataset

def train_pointnet(train_points, train_labels, val_points=None, val_labels=None, 
                  num_epochs=250, batch_size=32, learning_rate=0.001, num_classes=40):
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PointNetCls(num_classes=num_classes).to(device)
    
    # Create datasets and dataloaders
    train_dataset = PointCloudDataset(train_points, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    if val_points is not None:
        val_dataset = PointCloudDataset(val_points, val_labels)
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
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                }, 'best_model.pth')
        
        print('-' * 60)
    
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
    num_classes = 3
    num_samples = 100
    
    # Generate random point clouds and labels
    train_points = [np.random.randn(num_points, 3) for _ in range(num_samples)]
    train_labels = np.random.randint(0, num_classes, num_samples)
    
    val_points = [np.random.randn(num_points, 3) for _ in range(num_samples//5)]
    val_labels = np.random.randint(0, num_classes, num_samples//5)
    
    # Train the model
    model = train_pointnet(
        train_points=train_points,
        train_labels=train_labels,
        val_points=val_points,
        val_labels=val_labels,
        num_epochs=10,
        batch_size=32,
        learning_rate=0.001,
        num_classes=num_classes
    )
    
    # Test the model
    test_acc = test_pointnet(model, val_points, val_labels)