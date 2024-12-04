import torch
import os

def read_checkpoint_metrics(checkpoint_path):
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path)
    
    # Print basic metrics
    print(f"\nResults for epoch {checkpoint['epoch']}:")
    print(f"Validation Accuracy: {checkpoint['val_acc']:.2f}%")
    
    # Print detailed metrics
    metrics = checkpoint['metrics']
    print(f"\nMacro Averages:")
    print(f"Precision: {metrics['macro_precision']:.4f}")
    print(f"Recall: {metrics['macro_recall']:.4f}")
    print(f"F1-Score: {metrics['macro_f1']:.4f}")
    
    # Print per-class metrics
    print("\nPer-class Metrics:")
    num_classes = len(metrics['precision_per_class'])
    for i in range(num_classes):
        print(f"\nClass {i}:")
        print(f"  Precision: {metrics['precision_per_class'][i]:.4f}")
        print(f"  Recall: {metrics['recall_per_class'][i]:.4f}")
        print(f"  F1-score: {metrics['f1_per_class'][i]:.4f}")

# Example usage
checkpoint_path = "/home/stu12/s11/ak1825/CSCI739/pointnet/checkpoints/model_epoch_14.pth"  # Replace with your checkpoint path
read_checkpoint_metrics(checkpoint_path)