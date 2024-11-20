import torch
from torch.utils.data import Dataset
import numpy as np
import os
import yaml

def read_calib_file(calib_path):
    """
    Read calibration file
    Args:
        calib_path: path to calibration yaml file
    Returns:
        calib_data: dictionary containing calibration matrices
    """
    with open(calib_path, 'r') as f:
        calib_data = yaml.safe_load(f)
    return calib_data

def read_bin_point_cloud(file_path, calib_data=None):
    """
    Read point cloud from binary file and optionally apply calibration
    Args:
        file_path: path to .bin file
        calib_data: optional calibration data dictionary
    Returns:
        points: numpy array of shape (N, 3)
    """
    root_path = '' # add path to data
    # Read binary point cloud
    points = np.fromfile(file_path, dtype=np.float32)
    
    # Calculate number of points
    num_points = len(points) // 4  # Each point has x,y,z,intensity
    
    # Reshape to (N, 4) - x,y,z,intensity
    points = points.reshape((num_points, 4))
    
    # Extract xyz coordinates
    points = points[:, :3]  # Only take x,y,z, dropping intensity
    
    # Apply calibration if provided
    if calib_data is not None:
        velo_to_cam = np.array(calib_data.get('velo_to_cam', np.eye(4)))
        points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))
        points = (velo_to_cam @ points_homogeneous.T).T[:, :3]
    
    return points

class PointCloudDataset(Dataset):
    def __init__(self, point_clouds, labels, num_points=1024, calib_file=None, mode = 'train'):
        """
        Args:
            point_clouds: list of paths to .bin files or numpy arrays of shape (N, 3)
            labels: list of class labels
            num_points: number of points to sample from each point cloud
            calib_file: optional path to calibration file
        """
        self.point_clouds = point_clouds
        self.labels = labels
        self.num_points = num_points
        
        """
        TODO:
        - Add path as a parameter
        """
        self.root_path = "/home/sm2678/csci_739_term_project/CSCI739/data/"
        self.mode = mode
        
        # Load calibration if provided
        if calib_file and os.path.exists(calib_file):
            self.calib_data = read_calib_file(calib_file)
        else:
            self.calib_data = None

    def __len__(self):
        return len(self.point_clouds)

    def __getitem__(self, idx):
        # Load point cloud
        if isinstance(self.point_clouds[idx], str):
            # Load from .bin file
            # print(os.path.join(self.root_path, self.point_clouds[idx]+'.bin'))
            """
            TODO:
            - Handle path in __init__
            """
            if self.mode == 'train':
                point_cloud = read_bin_point_cloud(os.path.join(
                    self.root_path, 'velodyne', 'training', 'velodyne', self.point_clouds[idx]+'.bin'), self.calib_data)
            else:
                point_cloud = read_bin_point_cloud(os.path.join(
                    self.root_path, 'velodyne', 'validation', 'velodyne', self.point_clouds[idx]+'.bin'), self.calib_data)
        else:
            # Already a numpy array
            point_cloud = self.point_clouds[idx]

        # Randomly sample points if necessary
        if point_cloud.shape[0] > self.num_points:
            indices = np.random.choice(point_cloud.shape[0], self.num_points, replace=False)
            point_cloud = point_cloud[indices]
        elif point_cloud.shape[0] < self.num_points:
            indices = np.random.choice(point_cloud.shape[0], self.num_points, replace=True)
            point_cloud = point_cloud[indices]

        # Normalize point cloud to unit sphere
        centroid = np.mean(point_cloud, axis=0)
        point_cloud = point_cloud - centroid
        furthest_distance = np.max(np.sqrt(np.sum(point_cloud ** 2, axis=1)))
        point_cloud = point_cloud / furthest_distance

        # Convert to tensor and transpose for PointNet (N,3) -> (3,N)
        point_cloud = torch.FloatTensor(point_cloud.T)
        label = torch.LongTensor([self.labels[idx]])[0]

        return point_cloud, label

# Example usage:
if __name__ == "__main__":
    # Example of how to use the dataset
    data_dir = "CSCI739/samples/velodyne"
    calib_file = "CSCI739/samples/calib.yaml"  # Optional calibration file
    
    # Get all .bin files
    point_cloud_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.bin')]
    labels = [0] * len(point_cloud_files)  # Replace with actual labels
    
    # Create dataset with optional calibration
    dataset = PointCloudDataset(point_cloud_files, labels, calib_file=calib_file)
    
    # Test the dataset
    point_cloud, label = dataset[0]
    print(f"Point cloud shape: {point_cloud.shape}")  # Should be (3, 1024)
    print(f"Label: {label}")
    
    # Verify point cloud statistics
    print(f"Mean: {point_cloud.mean():.3f}")  # Should be close to 0
    print(f"Std: {point_cloud.std():.3f}")   # Should be reasonable