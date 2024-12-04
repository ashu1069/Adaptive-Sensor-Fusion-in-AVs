import sys
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='PointNet training for KITTI dataset')

    # Dataset settings
    parser.add_argument('--root_dir', default='../CSCI_files/training/', type=str,
                        help='root directory of KITTI dataset')
    parser.add_argument('--split', default='training', type=str, choices=['training', 'testing'],
                        help='dataset split to use')
    
    # Training settings
    parser.add_argument('--batch_size', default=32, type=int,
                        help='training batch size')
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='learning rate')
    parser.add_argument('--epochs', default=15, type=int,
                        help='number of training epochs')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='number of data loading workers')
    parser.add_argument('--num_classes', default=8, type=int,
                        help='number of classes in KITTI dataset')
    
    # Model settings
    parser.add_argument('--num_points', default=512, type=int,
                        help='number of points in point cloud')
    parser.add_argument('--save_model_path', default='./checkpoints/', type=str,
                        help='checkpoints directory')

    args = parser.parse_args()
    
    assert args.root_dir is not None, "Please specify the root directory of KITTI dataset"
    
    print(' '.join(sys.argv))
    print(args)

    return args