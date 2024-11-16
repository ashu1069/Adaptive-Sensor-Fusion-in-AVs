from evidential_deep_learning.evidential import EvidentialDetectionHead
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def load_calib(calib_path):
    """Load KITTI calibration file."""
    with open(calib_path, 'r') as f:
        lines = f.readlines()
    
    calibs = {}
    for line in lines:
        line = line.strip()  # Remove whitespace and newlines
        if line:  # Skip empty lines
            try:
                key, value = line.split(':', 1)
                calibs[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                continue  # Skip lines that don't contain key-value pairs
    
    # Reshape matrices
    if 'P2' in calibs:
        calibs['P2'] = calibs['P2'].reshape(3, 4)  # Camera projection matrix
    if 'R0_rect' in calibs:
        calibs['R0_rect'] = calibs['R0_rect'].reshape(3, 3)  # Rectification matrix
    if 'Tr_velo_to_cam' in calibs:
        calibs['Tr_velo_to_cam'] = calibs['Tr_velo_to_cam'].reshape(3, 4)  # Velodyne to camera transform
    
    return calibs

if __name__ == "__main__":
    # Load calibration
    calib = load_calib('samples/calibration/000044.txt')  # Adjust path to your calib file
    
    # Load point cloud and transform to camera coordinates
    point_cloud = np.fromfile('samples/velodyne/000044.bin', dtype=np.float32)
    point_cloud = point_cloud.reshape(-1, 4)[:, :3]  # Reshape to (N, 3), dropping intensity
    
    # Transform point cloud to camera coordinates
    points_h = np.hstack((point_cloud, np.ones((point_cloud.shape[0], 1))))  # Homogeneous coordinates
    points_cam = points_h @ calib['Tr_velo_to_cam'].T
    points_rect = points_cam @ calib['R0_rect'].T
    
    # Convert to torch tensor and add batch dimension
    point_cloud = torch.from_numpy(points_rect).float()
    point_cloud = point_cloud.unsqueeze(0)  # Add batch dimension
    point_cloud = point_cloud.transpose(1, 2)  # Shape: (1, 3, N)
    
    # Load and preprocess image
    image = plt.imread('samples/image_left/000044.png')  # Adjust path
    image = torch.from_numpy(image).float()
    if image.ndim == 3:
        image = image.permute(2, 0, 1)  # Convert to (C, H, W)
    image = image.unsqueeze(0)  # Add batch dimension
    
    # Ensure image is the right size
    if image.shape[-2:] != (224, 224):
        image = torch.nn.functional.interpolate(image, size=(224, 224))
    
    # Initialize evidential detection head
    detection_head = EvidentialDetectionHead(num_classes=3)
    
    # Set model to evaluation mode
    detection_head.eval()
    
    # Test forward pass with torch.no_grad()
    with torch.no_grad():
        outputs = detection_head(point_cloud, image)
    
    # Print detailed information about the outputs
    print("\nDetailed Evidential Parameters:")
    print("-" * 50)
    
    # Classification related parameters
    print("\nClassification Parameters:")
    print(f"Evidence values:\n{outputs['evidence'].detach().numpy()}")
    print(f"\nClass probabilities:\n{outputs['class_probs'].detach().numpy()}")
    print(f"\nClassification uncertainty:\n{outputs['class_uncertainty'].detach().numpy()}")
    
    # Regression related parameters
    print("\nRegression Parameters:")
    print(f"Bounding box predictions:\n{outputs['bbox_pred'].detach().numpy()}")
    print(f"\nEpistemic uncertainty:\n{outputs['bbox_epistemic_uncertainty'].detach().numpy()}")
    print(f"\nAleatoric uncertainty:\n{outputs['bbox_aleatoric_uncertainty'].detach().numpy()}")
    
    # Visualizations
    plt.figure(figsize=(15, 10))
    
    # Plot classification probabilities and uncertainty
    plt.subplot(2, 2, 1)
    sns.heatmap(outputs['class_probs'].detach().numpy(), 
                annot=True, 
                fmt='.2f',
                cmap='YlOrRd')
    plt.title('Class Probabilities')
    plt.xlabel('Class')
    plt.ylabel('Sample')
    
    # Plot classification uncertainty
    plt.subplot(2, 2, 2)
    plt.bar(range(len(outputs['class_uncertainty'])), 
           outputs['class_uncertainty'].detach().numpy().flatten())
    plt.title('Classification Uncertainty')
    plt.xlabel('Sample')
    plt.ylabel('Uncertainty')
    
    # Plot regression uncertainties
    plt.subplot(2, 2, 3)
    epistemic = outputs['bbox_epistemic_uncertainty'].detach().numpy()
    plt.boxplot([epistemic[:, i] for i in range(4)], 
                labels=['x', 'y', 'w', 'h'])
    plt.title('Epistemic Uncertainty Distribution')
    plt.ylabel('Uncertainty')
    
    plt.subplot(2, 2, 4)
    aleatoric = outputs['bbox_aleatoric_uncertainty'].detach().numpy()
    plt.boxplot([aleatoric[:, i] for i in range(4)], 
                labels=['x', 'y', 'w', 'h'])
    plt.title('Aleatoric Uncertainty Distribution')
    plt.ylabel('Uncertainty')
    
    plt.tight_layout()
    plt.show()
    
    # Print shapes
    print("\nOutput Tensor Shapes:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: {value.shape}")