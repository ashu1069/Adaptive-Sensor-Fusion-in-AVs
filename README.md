# Multi-Modal 2D Object Detection

This project implements a multi-modal 2D object detection system that fuses LiDAR point cloud data with camera images using the KITTI dataset.

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
git clone https://github.com/ashu1069/CSCI739.git
cd fusion
```

```bash
pip install -r requirements.txt
```
1. Prepare the KITTI dataset:

    - Download the KITTI object detection dataset
    - Organize the data in the following structure:

            data/
                ├── calibration/
                │   └── training/
                │       └── calib/
                ├── left_images/
                │   └── training/
                │       └── image_2/
                ├── labels/
                │   └── training/
                │       └── label_2/
                └── velodyne/
                    └── training/
                        └── velodyne/

2. Train the model:

```bash
python train.py --dataset_root /path/to/dataset --yolo_path /path/to/yolov8n.pt --pointnet_path /path/to/pointnet.pth
--batch_size 16 --epochs 100 --lr 0.001 --save_dir /path/to/save/model
```

3. Evaluate the model:

```bash
python evaluate.py --dataset_root /path/to/dataset --weights /path/to/save/model/checkpoint.pth --output_dir /path/to/save/results
```

## Model Architecture
The system consists of three main components:
1. **Feature Extractors**: Pre-trained YOLOv8 and PointNet models
2. **Fusion Module**: Adaptive fusion with element-wise operations and cross-attention
3. **Detection Head**: Multi-scale object detection with FPN

