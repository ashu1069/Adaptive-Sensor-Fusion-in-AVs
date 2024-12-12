import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Multi-Modal 3D Object Detection Training')
    
    # Data paths
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to KITTI dataset')
    parser.add_argument('--yolo_path', type=str, required=True,
                        help='Path to YOLO model weights')
    parser.add_argument('--pointnet_path', type=str, required=True,
                        help='Path to PointNet model weights')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Directory to save outputs')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Initial learning rate')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--num_classes', type=int, default=8,
                        help='Number of object classes')
    
    # Model parameters
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    # Add evaluation-specific arguments
    # parser.add_argument('--model_path', type=str, required=True,
    #                    help='Path to the best model checkpoint')
    
    return parser.parse_args()
