import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

def load_calib(calib_file):
    """
    Load KITTI calibration file
    Returns calibration matrices
    """
    with open(calib_file, 'r') as f:
        lines = f.readlines()

    # Parse P2 (camera matrix)
    P2 = np.array([float(info) for info in lines[2].split()[1:]]).reshape(3, 4)
    print(f'P2: {P2}')
    
    # Parse R0_rect
    R0_rect = np.array([float(info) for info in lines[4].split()[1:]]).reshape(3, 3)
    # Add row and column to make it 4x4
    R0_rect = np.pad(R0_rect, ((0, 1), (0, 1)), mode='constant')
    R0_rect[3, 3] = 1
    print(f'R0_rect: {R0_rect}')
    
    # Parse Tr_velo_to_cam
    Tr_velo_to_cam = np.array([float(info) for info in lines[5].split()[1:]]).reshape(3, 4)
    # Add row to make it 4x4
    Tr_velo_to_cam = np.vstack((Tr_velo_to_cam, [0, 0, 0, 1]))
    print(f'Tr_velo_to_cam: {Tr_velo_to_cam}')
    return P2, R0_rect, Tr_velo_to_cam

def lidar_to_image(points, P2, R0_rect, Tr_velo_to_cam):
    """
    Transform LiDAR points to image space
    Args:
        points: nx3 array of LiDAR points
        P2: Camera projection matrix
        R0_rect: Rectification matrix
        Tr_velo_to_cam: Transformation from velodyne to camera
    Returns:
        points_img: nx2 array of image coordinates
        points_depth: nx1 array of depths
    """
    # Add ones to make homogeneous coordinates (nx4)
    points_h = np.hstack((points, np.ones((points.shape[0], 1))))

    # Transform from velodyne to camera
    points_cam = Tr_velo_to_cam.dot(points_h.T)

    # Apply rectification
    points_rect = R0_rect.dot(points_cam)

    # Project to image
    points_proj = P2.dot(points_rect)

    # Normalize
    points_proj = points_proj.T
    points_img = points_proj[:, :2] / points_proj[:, 2:3]
    
    # Get depth
    points_depth = points_proj[:, 2]
    print(f'points_depth: {points_depth}')

    return points_img, points_depth

# Example usage:
def project_lidar_to_image(lidar_file, calib_file, img_shape=(375, 1242)):
    """
    Project LiDAR points to image
    Args:
        lidar_file: Path to LiDAR bin file
        calib_file: Path to calibration file
        img_shape: Shape of the image (height, width)
    Returns:
        valid_points: Points that project into image
        valid_depth: Depth of valid points
        valid_intensity: Intensity of valid points
    """
    # Load calibration
    P2, R0_rect, Tr_velo_to_cam = load_calib(calib_file)
    
    # Load LiDAR points with intensity
    points_with_intensity = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)
    points = points_with_intensity[:, :3]  # XYZ
    intensity = points_with_intensity[:, 3]  # Intensity
    
    # Project to image
    points_img, points_depth = lidar_to_image(points, P2, R0_rect, Tr_velo_to_cam)
    
    # Filter points that are in front of the camera and within image bounds
    mask = (points_depth > 0) & \
           (points_img[:, 0] >= 0) & (points_img[:, 0] < img_shape[1]) & \
           (points_img[:, 1] >= 0) & (points_img[:, 1] < img_shape[0])
    
    return points_img[mask], points_depth[mask], intensity[mask]

def generate_maps(lidar_file, calib_file, img_shape=(375, 1242)):
    """
    Generate depth and reflectance maps from LiDAR points
    Args:
        lidar_file: Path to LiDAR bin file
        calib_file: Path to calibration file
        img_shape: Shape of the image (height, width)
    Returns:
        depth_map: Height x Width depth map
        reflectance_map: Height x Width reflectance map
    """
    # Get projected points
    points_img, points_depth, points_intensity = project_lidar_to_image(lidar_file, calib_file, img_shape)
    
    # Initialize maps
    depth_map = np.zeros(img_shape)
    reflectance_map = np.zeros(img_shape)
    
    # Convert to integer coordinates
    points_img = points_img.astype(np.int32)
    
    # For each point, update maps
    for i in range(len(points_depth)):
        x, y = points_img[i]
        depth = points_depth[i]
        intensity = points_intensity[i]
        
        # Update depth only if it's closer than existing depth
        if depth_map[y, x] == 0 or depth < depth_map[y, x]:
            depth_map[y, x] = depth
            reflectance_map[y, x] = intensity
    
    return depth_map, reflectance_map

if __name__ == '__main__':
    # Create results directory if it doesn't exist
    results_dir = 'CSCI739/results'
    os.makedirs(results_dir, exist_ok=True)
    
    lidar_file = 'CSCI739/samples/velodyne/000044.bin'
    calib_file = 'CSCI739/samples/calibration/000044.txt'
    
    # Generate maps
    depth_map, reflectance_map = generate_maps(lidar_file, calib_file)
    
    # Normalize maps for visualization
    depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_map_normalized = depth_map_normalized.astype(np.uint8)
    
    reflectance_map_normalized = cv2.normalize(reflectance_map, None, 0, 255, cv2.NORM_MINMAX)
    reflectance_map_normalized = reflectance_map_normalized.astype(np.uint8)
    
    # Apply colormap for better visualization
    depth_map_color = cv2.applyColorMap(depth_map_normalized, cv2.COLORMAP_JET)
    reflectance_map_color = cv2.applyColorMap(reflectance_map_normalized, cv2.COLORMAP_JET)
    
    # Show original image and maps
    img = cv2.imread('CSCI739/samples/image_left/000044.png')
    
    # Display results
    cv2.imshow('Original Image', img)
    cv2.imshow('Depth Map', depth_map_color)
    cv2.imshow('Reflectance Map', reflectance_map_color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Save the maps in the results folder
    cv2.imwrite(os.path.join(results_dir, 'depth_map.png'), depth_map_normalized)
    cv2.imwrite(os.path.join(results_dir, 'depth_map_color.png'), depth_map_color)
    cv2.imwrite(os.path.join(results_dir, 'reflectance_map.png'), reflectance_map_normalized)
    cv2.imwrite(os.path.join(results_dir, 'reflectance_map_color.png'), reflectance_map_color)
    
    # Save raw values
    np.save(os.path.join(results_dir, 'depth_map_raw.npy'), depth_map)
    np.save(os.path.join(results_dir, 'reflectance_map_raw.npy'), reflectance_map)

