# Adaptive Multi-sensor Fusion for Robust Object Detection in Autonomous Vehicles using Evidential Deep Learning
### Adaptive Multi-sensor Fusion

- Sensors: LiDAR, Multi-stereo camera, Monocular RGB
- Implement a mechanism to dynamically adjust the fusion weights based on sensor reliability and environmental conditions
    - Design separate network branches for each sensor type
    - Implement feature extraction layer specific to each modality (e.g. PointNet for LiDAR, CNN for Image)
        - PoitnNet++ for LiDAR
    - Design a fusion mechanism that can dynamically adjust weights for each sensor
    - Implement attention mechanism or gating networks for adaptive fusion
 
### Evidential Deep Learning for Object Detection

- We can adapt an existing object detection architecture (e.g. YOLO) to output evidential parameters instead of class probabilities
    - Choose a base object detection architecture
    - Implement or adapt the chosen architecture for multi-modal/multi-sensor inputs
- Uncertainty Quantification: We can use Dirichlet distribution parameters for classification uncertainty and Inverse-Wishart distribution for bounding box regression uncertainty
    - Modify the output layer instead to produce evidential parameters instead of class probabilities
        - This has a scaling issue in terms of the number of classes
        - We can put a threshold to select a certain set of classes for uncertainty quantification

### Robustness through Adversarial Training

- Generate adversarial examples for each sensor modality
    - Fast Gradient Sign Method Attack (Image)
    - Projected Gradient Descent Attack (Image)
    - Point Perturbation Attack (LiDAR)
- Incorporate adversarial training to improve model robustness

#### Data Preprocessing
While exploring the Kitti Datset here are some of the operations that are applied to meet certain rewquirements for processing them through ML algorithms:
- Lidar
    - Problems: 
        - Different number of opservation points per lidar scan
        - May need transformation - Is in a different coordinate system than camera coordinates
    - Solutions:
        - Voxelization with fixed number of output points 
        - Functionality to transform the point cloud coordinates
- Camers
    - Problems:
        - Diffrent image sizes
    - Solutions:
        - Resize the image : resized based on the model/user's needs
- labels
    - Problems:
        - Need to check if the bounding box coordinates are in the camera's perspective or not!
    - Solution:
        - Ask professor and google

The data loder gets the following data elemets in each batch:
- Images from left camera
- Lidar Point clouds, if calib is provided the points are transformed to cam space
- Labels with all data points