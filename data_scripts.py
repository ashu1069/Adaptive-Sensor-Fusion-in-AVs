import numpy as np
import open3d as o3d
import os
import shutil
from tqdm import tqdm
import sys

def read_velodyne_bin(file_path):
    # Load the .bin file
    point_cloud = np.fromfile(file_path, dtype=np.float32)
        
    # Reshape the array to (N, 4), where N is the number of points
    point_cloud = point_cloud.reshape(-1, 4)    
    # Extract x, y, z coordinates
    xyz = point_cloud[:, :3]
    return xyz

def visualize_point_cloud(xyz):
    output_file_path = "output_point_cloud.ply"
    
    # Create Open3D PointCloud object
    pcd = o3d.geometry.PointCloud()
    
    # Set points
    pcd.points = o3d.utility.Vector3dVector(xyz)
    
    # # Visualize
    o3d.visualization.draw_geometries([pcd])
    o3d.io.write_point_cloud(output_file_path, pcd)

def extract_sensor_data(root_data_path:str, new_sensor_data_path:str, 
                     target_dir_name:str):
    '''
    Desc:   Basis on the current training and test data, fetches data from new 
            sensor data dir
    
    Args:
    root_data_dir : str : dir where the data already exists
    new_sensor_data_path : str : dir where new sensor data exists
    target_dir_name : str : name of new sensor as needed in root_data_dir
    
    Creator : Sarthakaushal
    Date : 10/11/24
    
    Returns : None
    '''
    sub_folders = os.listdir(root_data_path+'/image_left')
    sub_folders = [x  for x in sub_folders if '.' not in x ]
    metadata = {}
    for folder in sub_folders:
        metadata[folder] = os.listdir(os.path.join(root_data_path, 'image_left', folder))
    target_data_path = os.path.join(root_data_path, 'image_right')    
    if not target_dir_name in os.listdir(root_data_path):
        os.mkdir(target_data_path)
    
    for folder in metadata.keys():
        imgs = metadata[folder]
        img_path = os.path.join(new_sensor_data_path, folder)
        list_of_new_images = os.listdir(img_path)
        
        # Check if the image tag exists in the new sensor data
        for img in imgs:
            if not img in list_of_new_images:
                raise FileNotFoundError
        
        # transfer data to new location
        if folder not in os.listdir(target_data_path):
            os.mkdir(os.path.join(target_data_path, folder))
        for img in tqdm(imgs):
            shutil.copy(os.path.join(new_sensor_data_path, folder, img),
                        os.path.join(target_data_path,folder,img))

if __name__ == "__main__":
    # Path to your .bin file
    bin_file_path = "data/dev_datakit/velodyne/training/000044.bin"
    
    # Read and visualize
    point_cloud_xyz = read_velodyne_bin(bin_file_path)
    visualize_point_cloud(point_cloud_xyz)
    
    root_data_path = 'data/dev_datakit'
    new_sensor_data_path = 'data/data_object_image_3'
    if 'image_right' not in os.listdir(root_data_path):
        extract_sensor_data(root_data_path, new_sensor_data_path, 'image_right')
