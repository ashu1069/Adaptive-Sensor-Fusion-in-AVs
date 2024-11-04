import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from lidar_coordinate_transformation import read_calibration_file
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d

def disparity_to_depth(disparity, fb_constant=389.34):
    """
    Convert disparity map to depth map.
    
    Parameters:
    - disparity: numpy array of disparity values
    - fb_constant: focal length * baseline constant (default is 390.34 for KITTI)
    
    Returns:
    - depth: numpy array of depth values
    """
    
    # Convert disparity to depth (avoid division by zero)
    depth_map = np.zeros(disparity.shape, np.float32)
    valid_disparity_mask = disparity > 0  # Mask where disparity is valid
    depth_map[valid_disparity_mask] = fb_constant / disparity[valid_disparity_mask]
    # Replace zero and negative disparities to avoid division issues
    disparity[disparity <= 0] = 0.1  # Set to a small positive value

    # Convert disparity to depth
    depth = fb_constant / disparity
    
    return depth

def cv2_depth_gen(imgL_path, imgR_path, viz=False):
    """
    Generate depth map from stereo images using OpenCV's StereoSGBM algorithm.
    
    Args:
        imgL_path: Path to the left image
        imgR_path: Path to the right image
        viz: Boolean to display disparity and depth maps
    
    Returns:
        depth: Numpy array of depth values
        left_img: Numpy array of left image
        right_img: Numpy array of right image
    """
    imgL = cv.imread(imgL_path, cv.IMREAD_GRAYSCALE)
    imgR = cv.imread(imgR_path, cv.IMREAD_GRAYSCALE)
    
    #background matcher params
    sad_window = 10
    num_disparities = sad_window*16
    block_size = 11
    matcher_name = 'sgbm'
    
    matcher = cv.StereoSGBM.create(numDisparities=num_disparities,
                                        minDisparity=0,
                                        blockSize=block_size,
                                        P1 = 8 * 3 * 5 ** 2,
                                        P2 = 32 * 3 * 5 ** 2,
                                        mode=cv.STEREO_SGBM_MODE_SGBM_3WAY,
                                        disp12MaxDiff=1,
                                        uniquenessRatio=5,
                                        speckleWindowSize=200,
                                        speckleRange=2
                                       )
    # Calculate disparity using the StereoSGBM algorithm
    disparity = matcher.compute(imgL, imgR).astype(np.float32)
    disparity /= 16.0  # Scale down to real disparity

    # Calculate depth
    depth = disparity_to_depth(disparity)
    # Normalize depth for visualization purposes
    # depth_display = (depth / depth.max() * 255).astype(np.uint8)

    # Display disparity and depth maps
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Disparity Map")
    plt.imshow(disparity, cmap='plasma')
    plt.colorbar(label="Disparity (pixels)")

    plt.subplot(1, 2, 2)
    plt.title("Depth Map")
    plt.imshow(depth, cmap='plasma')
    plt.colorbar(label="Depth (meters)")

    plt.show()
    return depth , cv.imread(imgL_path), cv.imread(imgR_path)

def generate_pseudo_pointcloud(depth_map, P2, min_depth=0, max_depth=80):
    """
    Generate pseudo point cloud from depth map using camera parameters
    
    Args:
        depth_map: numpy array of depth values
        P2: Camera projection matrix (3x4) from KITTI
        min_depth: Minimum depth threshold
        max_depth: Maximum depth threshold
    
    Returns:
        points_3d: Nx3 array of 3D points
    """
    # Get image dimensions
    height, width = depth_map.shape
    
    # Create mesh grid of image coordinates
    rows, cols = np.meshgrid(range(height), range(width), indexing='ij')
    
    # Stack coordinates and reshape
    pixels = np.stack([cols.flatten(), rows.flatten(), np.ones_like(cols.flatten())])
    
    # Get camera intrinsic parameters from P2
    fx = P2[0, 0]
    fy = P2[1, 1]
    cx = P2[0, 2]
    cy = P2[1, 2]
    
    # Create camera intrinsic matrix
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]])
    
    # Back-project pixels to 3D points
    points_3d = np.linalg.inv(K) @ pixels
    
    # Multiply by depth
    depth_flat = depth_map.flatten()
    points_3d *= depth_flat[None, :]
    
    # Filter points based on depth threshold
    valid_points = (depth_flat > min_depth) & (depth_flat < max_depth)
    points_3d = points_3d[:, valid_points].T
    
    return points_3d

def visualize_open3d(points, colors=None, window_name="Open3D Visualization"):
    """
    Visualize point cloud using Open3D
    
    Args:
        points: Nx3 numpy array of points
        colors: Nx3 numpy array of RGB colors (optional)
        window_name: Title of visualization window
    
    Returns:
        None
    """
    # Create point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    else:
        # Default color if none provided
        pcd.paint_uniform_color([0.5, 0.5, 0.5])
    
    # Create coordinate frame
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
    
    # Visualize
    o3d.visualization.draw_geometries([pcd, coord_frame], 
                                    window_name=window_name,
                                    width=800, height=600)

if __name__ == "__main__":
    img_name = '000044.png'
    left_img_folder_path = 'data/dev_datakit/image_left/training'
    right_img_folder_path = 'data/dev_datakit/image_right/training'

    # # Replace with paths to your images
    depth, left_img, right_img = cv2_depth_gen(
        f'{left_img_folder_path}/{img_name}', 
        f'{right_img_folder_path}/{img_name}',
        True
    )

    calib_folder_path = 'data/dev_datakit/calibration/training/calib'

    cam_calib_file = read_calibration_file(f'{calib_folder_path}/{img_name.split(".")[0]}.txt')

    pseudo_point_cloud = generate_pseudo_pointcloud(
        depth,np.array(cam_calib_file['P2']).reshape(3,4)
    )

    print(pseudo_point_cloud)

    visualize_open3d(pseudo_point_cloud)