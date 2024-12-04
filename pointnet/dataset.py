import os
import sys
from typing import Dict
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from torchvision.models import resnet50

# Add parent directory to Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)


class PointCloudDataset(Dataset):
    '''
    Handles both LiDAR and camera data
    Processes KITTI format labels
    Applies transformations to images (224x224)
    Preprocesses point clouds (2048 points)
    Maps KITTI classes to indices
    '''
    def __init__(self, root_path, lidar_path, camera_path, calib_path, label_path,
                 mode='train', img_backbone=None, lidar_backbone=None):
        """
        Args:
            root_path: Path to KITTI dataset (dev_datakit)
            lidar_path: Path to lidar data
            camera_path: Path to camera data
            calib_path: Path to calibration data
            label_path: Path to label data
            mode: One of ['training', 'testing']
        """
        self.root_path = root_path
        self.lidar_path = lidar_path.format(mode)
        self.camera_path = camera_path.format(mode)
        self.calib_path = calib_path.format(mode)
        self.label_path = label_path.format(mode)
        self.img_backbone = img_backbone
        self.lidar_backbone = lidar_backbone
        
        # Image preprocessing for ResNet50
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
        # Point cloud preprocessing for PointNet
        self.num_points = 512  # Adjust based on your PointNet configuration
        
        # Get all frame IDs
        self.frame_ids = [x.split('.')[0] for x in os.listdir(os.path.join(
            root_path, self.lidar_path))]
            
        # Complete KITTI class mapping
        self.class_map = {
            'Car': 0,
            'Van': 1,
            'Truck': 2,
            'Pedestrian': 3,
            'Person_sitting': 4,
            'Cyclist': 5,
            'Tram': 6,
            'DontCare': 7
        }

    def __len__(self):
        return len(self.frame_ids)

    def _read_labels(self, label_path):
        """Read KITTI label file and return 2D bounding box targets"""
        boxes = []
        classes = []
        
        with open(label_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                cls_name = parts[0]
                
                # Get class index
                cls_idx = self.class_map.get(cls_name, self.class_map['DontCare'])
                
                # Parse 2D box parameters [left, top, right, bottom]
                # KITTI format: left(4), top(5), right(6), bottom(7)
                box = np.array([float(x) for x in parts[4:8]], dtype=np.float32)
                
                boxes.append(box)
                classes.append(cls_idx)
        
        return np.array(boxes), np.array(classes)

    def preprocess_pointcloud(self, point_cloud):
        """Preprocess point cloud for your PointNet implementation"""
        # Convert to correct format (N, 3) - using only x,y,z coordinates
        point_cloud = point_cloud[:, :3]
        
        # Randomly sample points if too many
        if point_cloud.shape[0] > self.num_points:
            choice = np.random.choice(point_cloud.shape[0], self.num_points, replace=False)
            point_cloud = point_cloud[choice, :]
        else:
            # Use interpolation for insufficient points
            num_missing = self.num_points - point_cloud.shape[0]
            # Generate indices for interpolation
            idx = np.linspace(0, point_cloud.shape[0] - 1, num_missing)
            idx = np.floor(idx).astype(int)
            # Interpolate between consecutive points
            interpolated = (point_cloud[idx] + np.roll(point_cloud[idx], -1, axis=0)) / 2
            point_cloud = np.vstack((point_cloud, interpolated))
        
        # Convert to tensor and transpose for PointNet input (3, N)
        point_cloud = torch.FloatTensor(point_cloud)
        point_cloud = point_cloud.transpose(0, 1).contiguous()
        return point_cloud

    def __getitem__(self, idx):
        frame_id = self.frame_ids[idx]
        
        # Load point cloud
        lidar_file = os.path.join(self.root_path, self.lidar_path, f"{frame_id}.bin")
        label_file = os.path.join(self.root_path, self.label_path, f"{frame_id}.txt")

        # Load and preprocess point cloud
        point_cloud = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)
        point_cloud = self.preprocess_pointcloud(point_cloud)
        
        # Load labels and get only classes
        _, classes = self._read_labels(label_file)
        
        return point_cloud, torch.tensor(classes[0] if len(classes) > 0 else 0, dtype=torch.long)

class LazyDataLoader:
    def __init__(self, dataset: Dataset, batch_size: int = 4, 
                 shuffle: bool = True, num_workers: int = 4,
                 worker_init_fn = None):
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self.collate_fn,
            worker_init_fn=worker_init_fn
        )

    def collate_fn(self, batch):
        """
        Simplified collate function for point clouds and classes
        """
        point_clouds = torch.stack([item[0] for item in batch])
        classes = torch.stack([item[1] for item in batch])
        
        return point_clouds, classes
        
def get_dataloader(root_path: str,
                  split: str = 'train',
                  batch_size: int = 4,
                  shuffle: bool = True,
                  num_workers: int = 4,
                  img_backbone = None,
                  lidar_backbone = None,
                  worker_init_fn = None) -> LazyDataLoader:
    """
    Create a dataloader for a specific split
    
    Args:
        root_path: Path to KITTI dataset (should point to the training folder)
        split: One of ['training', 'testing']
        batch_size: Batch size for the dataloader
        shuffle: Whether to shuffle the dataset
        num_workers: Number of workers for data loading
        img_backbone: Pre-trained image backbone model
        lidar_backbone: Pre-trained lidar backbone model
    """
    # Updated paths to point directly to subdirectories
    paths = {
        'lidar': 'velodyne',  # Remove format string
        'camera': 'image_2',
        'calib': 'calib',
        'label': 'label_2'
    }
    
    dataset = PointCloudDataset(
        root_path=root_path,
        lidar_path=paths['lidar'],
        camera_path=paths['camera'],
        calib_path=paths['calib'],
        label_path=paths['label'],
        mode=split,
        img_backbone=img_backbone,
        lidar_backbone=lidar_backbone
    )
    
    dataloader = LazyDataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn
    )
    
    return dataloader
