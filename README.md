# Multi-Modal 3D Object Detection

This project implements a multi-modal 3D object detection system that fuses LiDAR point cloud data with camera images using the KITTI dataset.

## Features

- Multi-modal fusion of LiDAR and camera data
- Feature extraction using YOLOv8 and PointNet
- Adaptive fusion mechanism with element-wise and cross-attention
- Feature Pyramid Network (FPN) for multi-scale detection
- Custom detection head with scale-specific predictions
- KITTI dataset support with robust data loading and preprocessing

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/fusion.git
cd fusion
```

```bash
pip install -r requirements.txt
```

## Project Structure

