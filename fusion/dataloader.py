import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os
from utils import MultiModalFeatureExtractor
from torchvision import transforms

class KITTIDataset(Dataset):
    def __init__(self, root_dir, split='training', yolo_path=None, pointnet_path=None):
        self.root_dir = os.path.join(root_dir, split)
        self.split = split
        
        # Verify paths exist
        if not os.path.exists(self.root_dir):
            raise FileNotFoundError(f"Dataset directory not found: {self.root_dir}")
        if not all(os.path.exists(os.path.join(self.root_dir, d)) for d in ['image_2', 'velodyne', 'label_2']):
            raise FileNotFoundError("Missing required dataset directories (image_2, velodyne, or label_2)")
        
        # Initialize feature extractor with error handling
        try:
            self.feature_extractor = MultiModalFeatureExtractor(yolo_path, pointnet_path)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize feature extractor: {str(e)}")
        
        # Cache directory listings
        image_dir = os.path.join(self.root_dir, 'image_2')
        self.samples = [os.path.splitext(f)[0] for f in sorted(os.listdir(image_dir)) 
                       if f.endswith(('.png', '.jpg'))]
        
        if not self.samples:
            raise RuntimeError(f"No valid images found in {image_dir}")
        
        # Update transform with error handling for corrupted images
        self.transform = transforms.Compose([
            transforms.Resize((640, 640)),
            transforms.ToTensor()
        ])
        
        # Add KITTI classes mapping
        self.classes = {
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
        return len(self.samples)
    
    def _load_point_cloud(self, lidar_path):
        """Load and preprocess LiDAR point cloud"""
        if not os.path.exists(lidar_path):
            raise FileNotFoundError(f"Point cloud file not found: {lidar_path}")
            
        try:
            scan = np.fromfile(lidar_path, dtype=np.float32)
            scan = scan.reshape((-1, 4))  # x, y, z, intensity
            points = scan[:, :3]  # Only use x, y, z
            
            # Handle empty point clouds
            if points.shape[0] == 0:
                raise ValueError(f"Empty point cloud found: {lidar_path}")
            
            # Randomly sample 1024 points
            if points.shape[0] > 1024:
                choice = np.random.choice(points.shape[0], 1024, replace=False)
                points = points[choice]
            else:
                # Pad with repeated points if less than 1024
                pad_choice = np.random.choice(points.shape[0], 1024 - points.shape[0])
                points = np.concatenate([points, points[pad_choice]], axis=0)
                
            return points
            
        except Exception as e:
            raise RuntimeError(f"Failed to load point cloud {lidar_path}: {str(e)}")
    
    def _process_label(self, label_path, img_size):
        """Process KITTI label file into target format"""
        targets = []
        
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split(' ')
                if parts[0] in self.classes:
                    # Extract bbox coordinates
                    bbox = [float(x) for x in parts[4:8]]  # [left, top, right, bottom]
                    
                    # Normalize coordinates by image dimensions
                    x_center = (bbox[0] + bbox[2]) / 2
                    y_center = (bbox[1] + bbox[3]) / 2
                    width = bbox[2] - bbox[0]
                    height = bbox[3] - bbox[1]
                    
                    # Determine scale_idx based on box size
                    box_size = max(width, height)
                    if box_size > 0.15:
                        scale_idx = 0  # Large objects
                    elif box_size > 0.05:
                        scale_idx = 1  # Medium objects
                    else:
                        scale_idx = 2  # Small objects
                    
                    targets.append([
                        self.classes[parts[0]],  # class_idx
                        x_center, y_center,      # normalized center coordinates
                        width, height,           # normalized dimensions
                        scale_idx                # scale index
                    ])
        
        return torch.tensor(targets, dtype=torch.float32)
    
    def __getitem__(self, idx):
        try:
            sample_id = self.samples[idx]
            
            # Load and preprocess image with error handling
            img_path = os.path.join(self.root_dir, 'image_2', f'{sample_id}.png')
            try:
                image = Image.open(img_path).convert('RGB')
                image = self.transform(image)
                if len(image.shape) == 3:
                    image = image.unsqueeze(0)
                assert image.shape == (1, 3, 640, 640), f"Unexpected image shape: {image.shape}"
            except Exception as e:
                raise RuntimeError(f"Failed to load/process image {img_path}: {str(e)}")
            
            # Load point cloud
            lidar_path = os.path.join(self.root_dir, 'velodyne', f'{sample_id}.bin')
            point_cloud = self._load_point_cloud(lidar_path)
            
            # Extract features
            try:
                features = self.feature_extractor.fuse_features(image, point_cloud)
            except Exception as e:
                raise RuntimeError(f"Feature extraction failed for sample {sample_id}: {str(e)}")
            
            # Load and process labels
            label_path = os.path.join(self.root_dir, 'label_2', f'{sample_id}.txt')
            if not os.path.exists(label_path):
                raise FileNotFoundError(f"Label file not found: {label_path}")
                
            orig_image = np.array(Image.open(img_path))
            targets = self._process_label(label_path, orig_image.shape[:2])
            
            # Validate data
            self._validate_data(features['global_features'], 
                              features['spatial_features'], 
                              targets, 
                              sample_id)
            
            return {
                'lidar_features': features['global_features'],
                'image_features': features['spatial_features'],
                'targets': targets,
                'image_id': sample_id
            }
            
        except Exception as e:
            print(f"Error processing sample {idx}: {str(e)}")
            # Return a default or skip this sample
            return self.__getitem__((idx + 1) % len(self))
    
    def _validate_data(self, lidar_features, image_features, targets, sample_id):
        """Validate data before returning"""
        if torch.isnan(lidar_features).any():
            raise ValueError(f"NaN values found in lidar features for {sample_id}")
        
        for i, feat in enumerate(image_features):
            if torch.isnan(feat).any():
                raise ValueError(f"NaN values found in image features {i} for {sample_id}")
        
        if torch.isinf(lidar_features).any():
            raise ValueError(f"Infinite values found in lidar features for {sample_id}")
        
        for i, feat in enumerate(image_features):
            if torch.isinf(feat).any():
                raise ValueError(f"Infinite values found in image features {i} for {sample_id}")
        
        if targets.min() < 0:
            raise ValueError(f"Negative target values found for {sample_id}")

def collate_fn(batch):
    """Custom collate function to handle batched data"""

    # for item in batch:
    #     for i in range(len(item['image_features'])):
    #         print(f"image_features shape: {item['image_features'][i].shape}")
    #     print('========================')
    
    return {
        'lidar_features': torch.stack([item['lidar_features'] for item in batch]),
        'image_features': [
            torch.stack([item['image_features'][i] for item in batch])
            for i in range(len(batch[0]['image_features']))
        ],
        'targets': [item['targets'] for item in batch],
        'image_id': [item['image_id'] for item in batch]
    }

'''
# Usage example:
dataset = KITTIDataset(
    root_dir='/home/stu12/s11/ak1825/CSCI_files',
    split='training',
    yolo_path='/home/stu12/s11/ak1825/idai710/Project/yolov8n.pt',
    pointnet_path='/home/stu12/s11/ak1825/CSCI_files/pointnet_checkpoints/model_epoch_14.pth'
)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=collate_fn
)

# Print one batch
batch = next(iter(dataloader))
print("Batch contents:")
print(f"Lidar features shape: {batch['lidar_features'].shape}")
print(f"Image features shapes: {[feat.shape for feat in batch['image_features']]}")
print(f"Targets: {[t.shape for t in batch['targets']]}")
print(f"Image IDs: {batch['image_id']}")
'''