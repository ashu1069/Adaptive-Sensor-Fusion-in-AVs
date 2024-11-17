import os
import sys
import numpy as np
import struct

# -----------------
# Point Cloud/Volume Conversion Utilities
# -----------------

def point_cloud_to_volume_batch(point_clouds: np.ndarray, voxel_size: float, radius: float = 1.0, flatten: bool = True) -> np.ndarray:
    """
    Convert a point cloud to a voxel volume
    Input:
        point_cloud: np.ndarray, shape B*(N, 3)
        voxel_size: float, size of each voxel
        radius: float, radius of the sphere to extract points from
        flatten: bool, whether to flatten the volume
    Output:
        B*(vol_size, vol_size, vol_size)
    """
    vol_list = []
    for b in range(point_clouds.shape[0]):
        vol = point_cloud_to_volume(np.squeeze(point_clouds[b,:,:]), voxel_size, radius)
        if flatten:
            vol_list.append(vol.flatten())
        else:
            vol_list.append(np.expand_dims(np.expand_dims(vol, -1), 0))
    if flatten:
        return np.vstack(vol_list)
    else:
        return np.concatenate(vol_list, 0)
    
def point_cloud_to_volume(point_clouds: np.ndarray, voxel_size: float, radius: float = 1.0) -> np.ndarray:
    """
    Convert a point cloud to a voxel volume
    Input:
        point_cloud: np.ndarray, shape (N, 3)
        voxel_size: float, size of each voxel
        radius: float, radius of the sphere to extract points from
    Output:
        (voxel_size, voxel_size, voxel_size)
    """
    vol = np.zeros((voxel_size, voxel_size, voxel_size))
    voxel = 2*radius/float(voxel_size)
    locations = (point_clouds + radius)/voxel
    locations = locations.astype(np.int32)
    
    # Add bounds checking
    mask = np.all((locations >= 0) & (locations < voxel_size), axis=1)
    locations = locations[mask]
    
    vol[locations[:,0], locations[:,1], locations[:,2]] = 1.0
    return vol

# a = np.zeros((16,1024,3))
# b = point_cloud_to_volume_batch(a, 12, 1.0, False)
# print(b.shape)

def volume_to_point_cloud(volume: np.ndarray) -> np.ndarray:
    """
    Convert a voxel volume to a point cloud
    Input:
        volume: np.ndarray, shape (vol_size, vol_size, vol_size)
    Output:
        point_cloud: np.ndarray, shape (N, 3)
    """
    voxel_size = volume.shape[0]
    assert(volume.shape == (voxel_size, voxel_size, voxel_size))

    points = []
    for a in range(voxel_size):
        for b in range(voxel_size):
            for c in range(voxel_size):
                if volume[a,b,c] == 1.0:
                    points.append(np.array([a,b,c]))
    if len(points) == 0:
        return np.zeros((0,3))
    points = np.vstack(points)
    return points
    
def read_bin_point_cloud(bin_path: str) -> np.ndarray:
    """
    Read point cloud from .bin file
    Input:
        bin_path: str, path to .bin file
    Output:
        points: np.ndarray, shape (N, 3) containing XYZ coordinates
    """
    points_list = []
    with open(bin_path, 'rb') as f:
        # Read all points from binary file
        byte_data = f.read()
        
        # Assuming the format is float32 for each X,Y,Z coordinate
        # If your .bin includes intensity/other fields, adjust the struct format and unpacking
        num_points = len(byte_data) // (4 * 3)  # 4 bytes per float, 3 coordinates per point
        
        for i in range(num_points):
            x, y, z = struct.unpack('fff', byte_data[i*12:(i+1)*12])  # 12 bytes per point (3 * 4)
            points_list.append([x, y, z])
    
    return np.array(points_list)

# Example usage:
# point_cloud = read_bin_point_cloud('CSCI739/samples/velodyne/000044.bin')
# volume = point_cloud_to_volume(point_cloud, voxel_size=32, radius=1.0)
# print(volume.shape)
    
    
    