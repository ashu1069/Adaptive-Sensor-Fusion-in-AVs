import numpy as np
import math
import random
import os
import torch
from args import parse_args
from dataset import get_dataloader
from pointnet_cls import PointNet
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

def pointnetloss(outputs, labels, m3x3, m64x64, alpha = 0.0001):
    criterion = torch.nn.NLLLoss()
    bs=outputs.size(0)
    id3x3 = torch.eye(3, requires_grad=True).repeat(bs,1,1)
    id64x64 = torch.eye(64, requires_grad=True).repeat(bs,1,1)
    if outputs.is_cuda:
        id3x3=id3x3.cuda()
        id64x64=id64x64.cuda()
    diff3x3 = id3x3-torch.bmm(m3x3,m3x3.transpose(1,2))
    diff64x64 = id64x64-torch.bmm(m64x64,m64x64.transpose(1,2))
    return criterion(outputs, labels) + alpha * (torch.norm(diff3x3)+torch.norm(diff64x64)) / float(bs)



def train(args):
    # Set seeds for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize model with correct number of classes
    pointnet = PointNet(classes=args.num_classes)
    pointnet.to(device)
    optimizer = torch.optim.Adam(pointnet.parameters(), lr=args.lr)
    
    # Get data loaders using the dataset.py implementation
    train_loader = get_dataloader(
        root_path=args.root_dir,
        split='training',
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )
    
    valid_loader = get_dataloader(
        root_path=args.root_dir,
        split='testing',
        batch_size=args.batch_size*2,
        shuffle=False,
        num_workers=0
    )
    
    print('Train dataset size: ', len(train_loader.dataloader.dataset))
    print('Valid dataset size: ', len(valid_loader.dataloader.dataset))
    print('Number of classes: ', args.num_classes)
    
    # Create checkpoints directory
    os.makedirs(args.save_model_path, exist_ok=True)
    
    print('Start training')
    for epoch in range(args.epochs):
        pointnet.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader.dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            # Note: inputs are already in the correct format (B, 3, N)
            outputs, m3x3, m64x64 = pointnet(inputs)
            loss = pointnetloss(outputs, labels, m3x3, m64x64)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 10 == 9:
                print('[Epoch: %d, Batch: %4d / %4d], loss: %.3f' %
                    (epoch + 1, i + 1, len(train_loader.dataloader), running_loss / 10))
                running_loss = 0.0
        
        # Validation with detailed metrics
        pointnet.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in valid_loader.dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs, _, _ = pointnet(inputs)
                _, predicted = torch.max(outputs.data, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Convert lists to numpy arrays
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # Calculate metrics
        val_acc = 100. * np.mean(all_preds == all_labels)
        precision_per_class = precision_score(all_labels, all_preds, average=None)
        recall_per_class = recall_score(all_labels, all_preds, average=None)
        f1_per_class = f1_score(all_labels, all_preds, average=None)
        
        # Calculate macro averages
        macro_precision = precision_score(all_labels, all_preds, average='macro')
        macro_recall = recall_score(all_labels, all_preds, average='macro')
        macro_f1 = f1_score(all_labels, all_preds, average='macro')
        
        # Print metrics
        print(f'\nEpoch {epoch + 1} Validation Metrics:')
        print(f'Overall Accuracy: {val_acc:.2f}%')
        print(f'Macro Precision: {macro_precision:.4f}')
        print(f'Macro Recall: {macro_recall:.4f}')
        print(f'Macro F1: {macro_f1:.4f}')
        
        print('\nPer-class metrics:')
        for i in range(args.num_classes):
            print(f'Class {i}:')
            print(f'  Precision: {precision_per_class[i]:.4f}')
            print(f'  Recall: {recall_per_class[i]:.4f}')
            print(f'  F1-score: {f1_per_class[i]:.4f}')
        
        # Save metrics in checkpoint
        checkpoint = os.path.join(args.save_model_path, f'model_epoch_{epoch}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': pointnet.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'metrics': {
                'precision_per_class': precision_per_class,
                'recall_per_class': recall_per_class,
                'f1_per_class': f1_per_class,
                'macro_precision': macro_precision,
                'macro_recall': macro_recall,
                'macro_f1': macro_f1
            }
        }, checkpoint)
        print('Model and metrics saved to', checkpoint)

if __name__ == '__main__':
    args = parse_args()
    train(args)