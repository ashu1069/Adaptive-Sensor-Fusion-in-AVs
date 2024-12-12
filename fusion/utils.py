import torch
from ultralytics import YOLO
import numpy as np
import sys
import os
import torchvision.ops as ops

# Add the parent directory to the Python path to find the pointnet module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pointnet.pointnet_cls import PointNet

class MultiModalFeatureExtractor:
    def __init__(self, yolo_model_path, pointnet_model_path):
        """
        Initialize feature extractors for both modalities
        """
        # Move device initialization to the top
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Now we can use self.device
        self.yolo_model = YOLO(yolo_model_path)
            
        self.pointnet_model = PointNet(classes=8)
        checkpoint = torch.load(pointnet_model_path)
        self.pointnet_model.load_state_dict(checkpoint['model_state_dict'])
        
        # Get the transform module for feature extraction
        self.transform = self.pointnet_model.transform
        
        # Move models to GPU if available
        self.pointnet_model.to(self.device)
        
        # Set model to evaluation mode
        self.pointnet_model.eval()
        
        # Initialize placeholders for intermediate features
        self.intermediate_features = []

    def extract_yolo_features(self, image):
        """
        Extract features from image using YOLOv8
        Args:
            image: RGB image tensor in BCHW format (1, 3, 640, 640)
        Returns:
            features: list of feature maps at three scales
        """
        # # Ensure image is in the correct format and on device
        # if isinstance(image, np.ndarray):
        #     image = torch.from_numpy(image).float()
        # if len(image.shape) == 3:
        #     image = image.unsqueeze(0)
        image = image.to(self.device)
        # Clear any existing features
        self.intermediate_features = []
        hooks = []
        
        def hook_fn(module, input, output):
            """Hook function to capture intermediate features"""
            self.intermediate_features.append(output.clone())
        
        try:
            # Register hooks for feature extraction
            target_layers = [15, 18, 21]  # P3, P4, P5 feature maps from backbone
            
            for idx in target_layers:
                hook = self.yolo_model.model.model[idx].register_forward_hook(hook_fn)
                hooks.append(hook)
            
            with torch.no_grad():
                _ = self.yolo_model(image, verbose=False)
                
            # Only keep the first occurrence of each unique feature map shape
            seen_shapes = set()
            unique_features = []
            for feat in self.intermediate_features:
                shape = tuple(feat.shape)
                if shape not in seen_shapes:
                    seen_shapes.add(shape)
                    unique_features.append(feat)
            
            features = unique_features[:3]  # Keep only the first 3 unique feature maps
            
        finally:
            # Always remove hooks and clear features
            for hook in hooks:
                hook.remove()
            self.intermediate_features = []
        
        return features

    def extract_pointnet_features(self, point_cloud):
        """
        Extract features from LiDAR point cloud using PointNet
        Args:
            point_cloud: Nx3 numpy array of points
        Returns:
            features: extracted features as tensor (B, 1024)
        """
        # Preprocess point cloud
        if isinstance(point_cloud, np.ndarray):
            point_cloud = torch.from_numpy(point_cloud).float()
        
        # Add batch dimension if needed
        if len(point_cloud.shape) == 2:
            point_cloud = point_cloud.unsqueeze(0)
            
        point_cloud = point_cloud.transpose(1, 2)  # Change to (B, 3, N) format
        point_cloud = point_cloud.to(self.device)
        
        with torch.no_grad():
            # Get the global features from the transform module
            global_features, _, _ = self.transform(point_cloud)
        
        return global_features  # Shape: (B, 1024)

    def fuse_features(self, image, point_cloud):
        """
        Extract and fuse features from both modalities
        Args:
            image: RGB image as numpy array
            point_cloud: Nx3 numpy array of points
        Returns:
            yolo_features: spatial features from YOLO (B, C, H, W)
            pointnet_features: global features from PointNet (B, F)
        """
        # Extract features from both modalities
        yolo_features = self.extract_yolo_features(image)     # Shape: (B, C, H, W)
        pointnet_features = self.extract_pointnet_features(point_cloud)  # Shape: (B, F)
        
        # Return both feature types without flattening YOLO features
        return {
            'spatial_features': yolo_features,        # Maintains spatial information
            'global_features': pointnet_features      # Global point cloud features
        }

def post_process_detections(detections, conf_thresh=0.5, nms_thresh=0.4):
    processed_detections = []
    device = detections[0].device  # Get device from first detection
    
    for scale_detections in detections:
        # Ensure detections are on the correct device and have the right format
        scale_detections = scale_detections.float().to(device)
        
        # Verify expected shape (bs, num_predictions, 6)
        if len(scale_detections.shape) == 2:
            scale_detections = scale_detections.unsqueeze(0)
            
        batch_size = scale_detections.shape[0]
        processed_batch = []
        
        # Process each item in the batch
        for batch_idx in range(batch_size):
            batch_dets = scale_detections[batch_idx]
            
            # Filter out low confidence predictions
            conf_mask = batch_dets[..., 4] > conf_thresh
            batch_dets = batch_dets[conf_mask]
            
            if len(batch_dets) == 0:
                processed_batch.append(batch_dets)
                continue
                
            # Get boxes, scores, and labels
            boxes = batch_dets[:, 0:4]  # x, y, w, h
            scores = batch_dets[:, 4]
            labels = batch_dets[:, 5].int()
            
            # Convert from xywh to xyxy format for NMS
            boxes_xyxy = torch.zeros_like(boxes)
            boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2  # x1 = x - w/2
            boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2  # y1 = y - h/2
            boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2  # x2 = x + w/2
            boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2  # y2 = y + h/2
            
            # Apply NMS
            keep = ops.nms(boxes_xyxy, scores, nms_thresh)
            batch_dets = batch_dets[keep]
            
            processed_batch.append(batch_dets)
            
        # Stack batch results if there are any detections
        if any(len(x) > 0 for x in processed_batch):
            processed_detections.append(torch.stack([x for x in processed_batch if len(x) > 0]))
        else:
            # If no detections, return empty tensor with correct shape
            processed_detections.append(torch.zeros((batch_size, 0, 6), device=device))
    
    return processed_detections

'''
def test_extract_yolo_features():
    """Test YOLO feature extraction"""
    yolo_model_path = "/home/stu12/s11/ak1825/CSCI739/yolov8n.pt"  # Update with your model path
    pointnet_model_path = "/home/stu12/s11/ak1825/CSCI_files/pointnet_checkpoints/model_epoch_14.pth"  # Update with your model path
    
    extractor = MultiModalFeatureExtractor(yolo_model_path, pointnet_model_path)
    
    # Test with different input formats
    # 1. Test with numpy array
    sample_image_np = np.random.rand(3, 640, 640).astype(np.float32)
    features_np = extractor.extract_yolo_features(sample_image_np)
    assert len(features_np) == 3, "Should return 3 feature maps"
    
    # 2. Test with torch tensor
    sample_image_torch = torch.rand(1, 3, 640, 640)
    features_torch = extractor.extract_yolo_features(sample_image_torch)
    assert len(features_torch) == 3, "Should return 3 feature maps"
    
    # Verify feature map shapes (example expected shapes)
    expected_shapes = [(1, 256, 80, 80), (1, 512, 40, 40), (1, 1024, 20, 20)]
    for feat, expected in zip(features_torch, expected_shapes):
        assert len(feat.shape) == 4, f"Feature should be 4D, got shape {feat.shape}"
        assert feat.shape[0] == 1, f"Batch size should be 1, got {feat.shape[0]}"

def test_extract_pointnet_features():
    """Test PointNet feature extraction"""
    yolo_model_path = "/home/stu12/s11/ak1825/CSCI739/yolov8n.pt"  # Update with your model path
    pointnet_model_path = "/home/stu12/s11/ak1825/CSCI_files/pointnet_checkpoints/model_epoch_14.pth"  # Update with your model path
    
    extractor = MultiModalFeatureExtractor(yolo_model_path, pointnet_model_path)
    
    # Test with different input formats
    # 1. Test with numpy array
    sample_points_np = np.random.rand(1024, 3).astype(np.float32)
    features_np = extractor.extract_pointnet_features(sample_points_np)
    assert features_np.shape == (1, 1024), f"Expected shape (1, 1024), got {features_np.shape}"
    
    # 2. Test with torch tensor
    sample_points_torch = torch.rand(1, 1024, 3)
    features_torch = extractor.extract_pointnet_features(sample_points_torch)
    assert features_torch.shape == (1, 1024), f"Expected shape (1, 1024), got {features_torch.shape}"

def test_fuse_features():
    """Test feature fusion"""
    yolo_model_path = "/home/stu12/s11/ak1825/CSCI739/yolov8n.pt"  # Update with your model path
    pointnet_model_path = "/home/stu12/s11/ak1825/CSCI_files/pointnet_checkpoints/model_epoch_14.pth"  # Update with your model path
    
    extractor = MultiModalFeatureExtractor(yolo_model_path, pointnet_model_path)
    
    sample_image = np.random.rand(3, 640, 640).astype(np.float32)
    sample_points = np.random.rand(1024, 3).astype(np.float32)
    
    features = extractor.fuse_features(sample_image, sample_points)
    
    # Verify dictionary structure
    assert 'spatial_features' in features, "Missing spatial_features in output"
    assert 'global_features' in features, "Missing global_features in output"
    
    # Verify spatial features
    spatial_features = features['spatial_features']
    assert len(spatial_features) == 3, "Should have 3 spatial feature maps"
    
    # Verify global features
    global_features = features['global_features']
    assert global_features.shape == (1, 1024), f"Expected shape (1, 1024), got {global_features.shape}"

# def test_post_process_detections():
#     """Test detection post-processing"""
#     # Create dummy detections (batch_size, num_detections, 6)
#     # Format: [x1, y1, x2, y2, confidence, class]
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     detections = [
#         torch.tensor([
#             [100, 100, 200, 200, 0.9, 1],
#             [110, 110, 210, 210, 0.8, 1],  # Overlapping box
#             [300, 300, 400, 400, 0.7, 2],
#         ]).to(device)
#     ]
    
#     processed = post_process_detections(detections, conf_thresh=0.5, nms_thresh=0.4)
    
#     # Verify output
#     assert len(processed) == 1, "Should have same number of scales as input"
#     assert processed[0].shape[1] == 6, "Each detection should have 6 values"
#     assert processed[0].shape[0] <= detections[0].shape[0], "NMS should reduce or maintain number of detections"

# def test_with_real_data():
#     """Test all functions with real data from dataset"""
#     import cv2
#     from pathlib import Path
    
#     # Setup paths
#     yolo_model_path = "/home/stu12/s11/ak1825/CSCI739/yolov8n.pt"  # Update with your model path
#     pointnet_model_path = "/home/stu12/s11/ak1825/CSCI_files/pointnet_checkpoints/model_epoch_14.pth"  # Update with your model path
#     data_root = Path("/home/stu12/s11/ak1825/CSCI_files/training")  # Update with your dataset path
    
#     # Initialize feature extractor
#     extractor = MultiModalFeatureExtractor(yolo_model_path, pointnet_model_path)
    
#     # Load sample data
#     # Assuming your dataset structure has image and point cloud pairs
#     sample_img_path = data_root / "image_2" / "000000.png"  # Update path format
#     sample_pc_path = data_root / "velodyne" / "000000.bin"  # Update path format
    
#     # Load and preprocess image
#     image = cv2.imread(str(sample_img_path))
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     image = cv2.resize(image, (640, 640))
    
#     # Store the CHW format for feature extraction
#     image_chw = image.transpose(2, 0, 1) / 255.0  # CHW format for feature extraction
    
#     # Use original HWC format for YOLO inference
#     results = extractor.yolo_model(image)  # Pass HWC format image
#     detections = results[0].boxes.data  # Get detection boxes
#     processed = post_process_detections([detections])
    
#     # Use CHW format for feature extraction
#     yolo_features = extractor.extract_yolo_features(image_chw)
    
#     # Load and preprocess point cloud
#     # Assuming binary format, adjust based on your data format
#     point_cloud = np.fromfile(str(sample_pc_path), dtype=np.float32).reshape(-1, 4)
#     point_cloud = point_cloud[:, :3]  # Keep only x, y, z
    
#     print("Testing with real data...")
#     print(f"Image shape: {image.shape}")
#     print(f"Point cloud shape: {point_cloud.shape}")
    
#     # Test YOLO feature extraction
#     print("\nTesting YOLO feature extraction...")
#     # Convert image to BCHW format before passing to extract_yolo_features
#     image_bchw = torch.from_numpy(image_chw).unsqueeze(0)  # Add batch dimension
#     yolo_features = extractor.extract_yolo_features(image_bchw)
#     for i, feat in enumerate(yolo_features):
#         print(f"YOLO feature map {i} shape: {feat.shape}")
    
#     # Test PointNet feature extraction
#     print("\nTesting PointNet feature extraction...")
#     pointnet_features = extractor.extract_pointnet_features(point_cloud)
#     print(f"PointNet features shape: {pointnet_features.shape}")
    
#     # Test feature fusion
#     print("\nTesting feature fusion...")
#     fused_features = extractor.fuse_features(image, point_cloud)
#     print("Spatial features shapes:", [f.shape for f in fused_features['spatial_features']])
#     print("Global features shape:", fused_features['global_features'].shape)
    
#     # Test post-processing
#     print("\nTesting post-processing...")
#     # Run YOLO inference to get detections
#     results = extractor.yolo_model(image)
#     detections = results[0].boxes.data  # Get detection boxes
#     processed = post_process_detections([detections])
#     print(f"Processed detections: {len(processed[0])} objects found")
    
#     print("\nAll real data tests completed successfully!")

if __name__ == "__main__":
    # Run all tests including real data test
    test_extract_yolo_features()
    test_extract_pointnet_features()
    test_fuse_features()
    print("All tests passed!")
'''