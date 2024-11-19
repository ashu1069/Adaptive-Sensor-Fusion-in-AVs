
import os
from typing import Dict
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pointnet.dataset import read_bin_point_cloud, read_calib_file
from vit_utils import label_to_id
import sys
import open3d as o3d

class PointCloudDataset(Dataset):
    def __init__(self, root_path, lidar_path, camera_path, calib_path, 
                 label_path, num_points=1024, mode='train',
                 return_image=False, return_calib=False, return_labels=True, 
                 image_transform=None):
        """
        Modified to store only file paths and configuration
        """
        self.root_path = root_path
        self.lidar_path = lidar_path.format(mode)
        self.camera_path = camera_path.format(mode)
        self.calib_path = calib_path.format(mode)
        self.label_path = label_path.format(mode)
        self.point_cloud_files = [x.split('.')[0] for x in os.listdir(os.path.join(
            root_path, self.lidar_path))]
        self.num_points = num_points
        self.mode = mode
        self.return_image = return_image
        self.return_calib = return_calib
        self.return_labels = return_labels
        self.image_transform = image_transform

    def __len__(self):
        return len(self.point_cloud_files)

    def __getitem__(self, idx):
        # Return only the file paths and configuration
        sample_id = self.point_cloud_files[idx]
        
        data = {
            'sample_id': sample_id,
            'lidar_path': os.path.join(self.root_path, self.lidar_path, f"{sample_id}.bin"),
            'config': {
                'num_points': self.num_points,
                'return_image': self.return_image,
                'return_calib': self.return_calib,
                'return_labels': self.return_labels,
            }
        }
        
        if self.return_image:
            data['image_path'] = os.path.join(self.root_path, self.camera_path, 
                                              f"{sample_id}.png")
            
        if self.return_calib:
            data['calib_path'] = os.path.join(self.root_path, self.calib_path, 
                                              f"{sample_id}.txt")
            
        if self.return_labels:
            data['label_path'] = os.path.join(self.root_path, self.label_path, 
                                              f"{sample_id}.txt")
            
        return data

class LazyDataLoader:
    def __init__(self, 
                 dataset:Dataset, 
                 batch_size:int =4 , 
                 shuffle:bool = True,
                 num_workers:int = 4,
                 image_size:tuple = (224, 224),
                 label_data_loc = list(range(15))):
        """
        Custom data loader that implements lazy loading
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.image_size = image_size
        self.label_data_loc = label_data_loc # indices of labels to extract
        self.dataloader = DataLoader(
            dataset, 
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self.collate_fn
        )

    def collate_fn(self, batch):
        """
        Custom collate function that loads the actual data
        """
        processed_batch = {
            'point_clouds': [],
            'images': [],
            'calibrations': [],
            'labels': [],
            'sample_ids': []
        }
        i = -1
        for sample in batch:
            i+=1
            sample_id = sample['sample_id']
            processed_batch['sample_ids'].append(sample_id)
            
            # Load calibration
            if sample['config']['return_calib']:
                calib = read_calib_file(sample['calib_path'])
                processed_batch['calibrations'].append(calib)
            else:
                calib = None
            
            # Load point cloud
            point_cloud = read_bin_point_cloud(
                file_path=sample['lidar_path'],
                calib_data=calib
            )
            processed_batch['point_clouds'].append(
                self._voxelize_grid(point_cloud))            
            # Load image if required
            if sample['config']['return_image']:
                image = self._load_image(sample['image_path'])
                # print(image, sample['image_path'])
                processed_batch['images'].append(image)
                
            # Load labels if required
            if sample['config']['return_labels']:
                label = self._load_label(sample['label_path'])
                label = [torch.tensor(x) for x in label]
                processed_batch['labels'].append(label)
            # print(processed_batch)
        # Convert lists to tensors
        processed_batch['point_clouds'] = torch.stack(processed_batch['point_clouds'])
        if processed_batch['images']:
            processed_batch['images'] = torch.stack(processed_batch['images'])
        if processed_batch['labels']:
            processed_batch['labels'] = (processed_batch['labels'])

        return processed_batch
    
    def _load_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        if self.dataset.image_transform:
            image = self.dataset.image_transform(image)
        else:
            image = transforms.ToTensor()(image)
        image = transforms.Resize(self.image_size)(image)
        return image
    
    def _voxelize_grid(self, points:np.array, voxel_size=0.1, num_points=50_000):
        # Convert NumPy array to Open3D PointCloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # Apply voxelization
        voxelized_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        voxelized_points = np.asarray(voxelized_pcd.points)

        # Resample to ensure the desired number of points
        if num_points is not None:
            if len(voxelized_points) > num_points:
                indices = np.random.choice(len(voxelized_points), num_points, replace=False)
                voxelized_points = voxelized_points[indices]
            elif len(voxelized_points) < num_points:
                additional_indices = np.random.choice(len(voxelized_points), num_points - len(voxelized_points), replace=True)
                voxelized_points = np.vstack((voxelized_points, voxelized_points[additional_indices]))

        # Convert to PyTorch tensor
        torch_tensor = torch.tensor(voxelized_points, dtype=torch.float32)
        # final_num_points = torch_tensor.shape[0]
        return torch_tensor
        
    def _load_label(self, label_path):
        '''
        Data reader for label file, data description:
            0-   Object type with dtype int
            1-   Truncation level (0-1)
            2-   Occlusion state (0,1,2,3)
            3-   Alpha angle
            4-   2D bbox left
            5-   2D bbox top
            6-   2D bbox right 
            7-   2D bbox bottom
            8-   Height
            9-   Width
            10-  Length
            11-  X location
            12-  Y location
            13-  Z location
            14-  Rotation Y
        '''
        labels = []
        with open(label_path, 'r') as f:
            label_content = f.readlines()
        
        for lable_line in label_content:
            parts = lable_line.strip().split()
            label = []
            for loc in self.label_data_loc:
                if loc== 0:
                    label.append(label_to_id(parts[loc]))
                elif loc ==2:
                    label.append(int(parts[loc]))
                else:
                    label.append(float(parts[loc]))
        
            # Add score if it exists (only in results files)
            if len(parts) > 15:
                label['score'] = float(parts[15])
            labels.append(label)
        return labels

# # **************** Sample usage **************** 
# if __name__ == "__main__":
#     # Example of how to use the dataset and dataloader
#     data_dir = "/home/sm2678/csci_739_term_project/CSCI739/data"
#     camera_dir = "left_images/{}/image_2"
#     lidar_dir = "velodyne/{}/velodyne/"
#     calib_dir = "calibration/{}/calib"
#     label_dir = "labels/{}/label_2"

#     mode = 'training'  # or 'validation' or 'test'

    
    
#     dataset = PointCloudDataset(
#         data_dir,lidar_dir, camera_dir, calib_dir, label_dir, 1024, "training",
#         return_image=True, return_calib=True, return_labels=True
#     )
    
#     lazy_loader = LazyDataLoader(
#         dataset=dataset,
#         batch_size=4,
#         shuffle=True,
#         num_workers=1
#     )
#     i=0
#     for batch in lazy_loader.dataloader:
#         i+=1
#         print(i)
#         # Each batch will contain:
#         point_clouds = batch['point_clouds']      # Shape: (batch_size, num_points, 3)
#         images = batch['images']                  # Shape: (batch_size, C, H, W)
#         labels = batch['labels']                  # List of label dictionaries
#         calibrations = batch['calibrations']      # List of calibration dictionaries
#         sample_ids = batch['sample_ids']          # List of sample IDs

#         # Print shapes and contents
#         print(f"Point clouds shape: {point_clouds.shape}")
#         print(f"Images shape: {images.shape}")
#         print(f"Number of labels: {len(labels)}")
#         print(f"Sample IDs: {sample_ids}")
        
        