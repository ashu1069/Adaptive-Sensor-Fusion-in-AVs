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
