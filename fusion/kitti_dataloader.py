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

# Now we can import from pointnet
from pointnet.pointnet_cls import PointNetCls

class PointCloudDataset(Dataset):
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
        # Get device from backbone if available, otherwise use cuda:4
        self.device = (next(img_backbone.parameters()).device 
                      if img_backbone is not None 
                      else torch.device("cuda:4" if torch.cuda.is_available() else "cpu"))
        # Image preprocessing for ResNet50
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
        # Point cloud preprocessing for PointNet
        self.num_points = 2048  # Adjust based on your PointNet configuration
        
        # Get all frame IDs
        self.frame_ids = [x.split('.')[0] for x in os.listdir(os.path.join(
            root_path, self.lidar_path))]
        # print(self.frame_ids)
            
        # Complete KITTI class mapping
        self.class_map = {
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

    def __len__(self):
        return len(self.frame_ids)

    def _read_labels(self, label_path):
        """Read KITTI label file and return targets"""
        boxes = []
        classes = []
        
        with open(label_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                cls_name = parts[0]
                
                # Get class index (all classes are now included)
                cls_idx = self.class_map.get(cls_name, self.class_map['DontCare'])
                
                # Parse box parameters [x,y,z,l,w,h,rot]
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
            # Pad with zeros if too few points
            zeros = np.zeros((self.num_points - point_cloud.shape[0], 3))
            point_cloud = np.vstack((point_cloud, zeros))
        
        # Convert to tensor and transpose for PointNet input (3, N)
        point_cloud = torch.FloatTensor(point_cloud)
        point_cloud = point_cloud.transpose(0, 1).contiguous()
        return point_cloud

    def __getitem__(self, idx):
        frame_id = self.frame_ids[idx]
        
        # Construct paths
        lidar_file = os.path.join(self.root_path, self.lidar_path, f"{frame_id}.bin")
        image_file = os.path.join(self.root_path, self.camera_path, f"{frame_id}.png")
        label_file = os.path.join(self.root_path, self.label_path, f"{frame_id}.txt")

        # Load and preprocess image
        image = Image.open(image_file).convert('RGB')
        image = self.image_transform(image)
        
        # Extract image features
        if self.img_backbone is not None:
            with torch.no_grad():
                image = image.unsqueeze(0).to(self.device)
                image_features = self.img_backbone(image)
                image_features = image_features.squeeze(0)
        else:
            image_features = image.to(self.device)

        # Load and preprocess point cloud
        point_cloud = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)
        point_cloud = self.preprocess_pointcloud(point_cloud)
        
        # Extract point cloud features
        if self.lidar_backbone is not None:
            with torch.no_grad():
                point_cloud = point_cloud.to(self.device)
                lidar_features, _, _ = self.lidar_backbone(point_cloud.unsqueeze(0))
                lidar_features = lidar_features.squeeze()
        else:
            lidar_features = point_cloud.to(self.device)

        # Load labels
        boxes, classes = self._read_labels(label_file)
        
        # Prepare targets
        targets = {
            'boxes_3d': torch.tensor(boxes, dtype=torch.float32),
            'classes': torch.tensor(classes, dtype=torch.long),
            'num_objects': len(boxes)
        }

        # Return features and targets
        return {
            'image_features': image_features,
            'lidar_features': lidar_features,
            'targets': targets
        }

class LazyDataLoader:
    def __init__(self, dataset: Dataset, batch_size: int = 4, 
                 shuffle: bool = True, num_workers: int = 4):
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self.collate_fn
        )

    def collate_fn(self, batch):
        """
        Collate function that handles the batch processing
        """
        batch_dict = {
            'frame_ids': [],
            'image_features': [],
            'lidar_features': [],
            'targets': []
        }
        # print(batch[0])
        for sample in batch:
            # batch_dict['frame_ids'].append(sample['frame_id'])
            # Image features should be (C, H, W)
            batch_dict['image_features'].append(sample['image_features'])
            # Lidar features should be (C,)
            batch_dict['lidar_features'].append(sample['lidar_features'])
            batch_dict['targets'].append(sample['targets'])

        # Stack to get (B, C, H, W) for image features
        batch_dict['image_features'] = torch.stack(batch_dict['image_features'])
        # Stack to get (B, C) for lidar features
        batch_dict['lidar_features'] = torch.stack(batch_dict['lidar_features'])

        return batch_dict
        
def get_dataloader(root_path: str,
                  split: str = 'train',
                  batch_size: int = 4,
                  shuffle: bool = True,
                  num_workers: int = 4,
                  img_backbone = None,
                  lidar_backbone = None) -> LazyDataLoader:
    """
    Create a dataloader for a specific split
    
    Args:
        root_path: Path to KITTI dataset (dev_datakit)
        split: One of ['training', 'testing']  # Updated valid values
        batch_size: Batch size for the dataloader
        shuffle: Whether to shuffle the dataset
        num_workers: Number of workers for data loading
        img_backbone: Pre-trained image backbone model
        lidar_backbone: Pre-trained lidar backbone model
    """
    paths = {
        'lidar': os.path.join('velodyne', '{}', 'velodyne'),
        'camera': os.path.join('left_images', '{}', 'image_2'),
        'calib': os.path.join('calibration', '{}', 'calib'),
        'label': os.path.join('labels', '{}', 'label_2') 
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
        num_workers=num_workers
    )
    
    return dataloader

def initialize_models(pointnet_weights_path=None):
    """Initialize both backbone models"""
    # Set device consistently
    device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")  # Using cuda:4 consistently
    
    # Initialize ResNet50
    resnet = resnet50(weights='IMAGENET1K_V1')
    resnet = torch.nn.Sequential(*list(resnet.children())[:-2]) 
    resnet.eval()
    resnet = resnet.to(device)

    # Initialize PointNet
    pointnet = PointNetCls()
    if pointnet_weights_path:
        checkpoint = torch.load(pointnet_weights_path, map_location=device)
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                k = k[7:]
            if k not in ['epoch', 'optimizer_state_dict', 'val_acc']:
                new_state_dict[k] = v
            
        try:
            pointnet.load_state_dict(new_state_dict, strict=False)
            print("PointNet weights loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load some PointNet weights: {str(e)}")
            
    pointnet.eval()
    pointnet = pointnet.to(device)

    return resnet, pointnet, device

def main():
    """Example usage with both backbones"""
    root_path = "CSCI_files/dev_datakit"
    pointnet_weights = "/home/sk4858/CSCI739/model_weights/checkpoint_epoch_200.pth"
    
    # Initialize models
    resnet_backbone, pointnet_backbone, device = initialize_models(pointnet_weights)
    
    # Device is now handled in initialize_models()
    
    # Updated split name to match your directory structure
    train_loader = get_dataloader(
        root_path=root_path,
        split='training',
        batch_size=4,
        shuffle=True,
        num_workers=4,
        img_backbone=resnet_backbone,
        lidar_backbone=pointnet_backbone
    )
    
    # Test the dataloader
    batch = next(iter(train_loader.dataloader))
    print("Batch contents:")
    print(f"Image features shape: {batch['image_features'].shape}")
    print(f"LiDAR features shape: {batch['lidar_features'].shape}")
    print(f"Number of targets: {len(batch['targets'])}")

if __name__ == "__main__":
    main()
        
        