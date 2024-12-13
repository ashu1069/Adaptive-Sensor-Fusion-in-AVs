import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from torchvision.transforms import functional as TF
from torchvision.ops import box_convert
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from torchvision import transforms
from kitti_data_handler import PointCloudDataset, LazyDataLoader
import sys
from torchvision.ops import generalized_box_iou_loss, box_iou
import torch.optim as optim
from collections import OrderedDict
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
import torch.nn.functional as F
from torchvision.ops import box_iou
import sys
import mlflow
import mlflow.pytorch
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.transforms import ColorJitter
import torchvision
import os
import argparse
from terminaltables import AsciiTable
from pointnet.pointnet_cls import PointNet
from detr.attentions import ContextGuidedFusion

# 1. Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(self, x):
        b, c, h, w = x.shape

        # Create grid of shape (h, w)
        y_embed = torch.arange(h, device=x.device).unsqueeze(1).repeat(1, w)
        # print(y_embed.size())
        x_embed = torch.arange(w, device=x.device).unsqueeze(0).repeat(h, 1)
        # print(x_embed.size())
        # Normalize by the dimensions of the feature map
        y_embed = y_embed / h
        x_embed = x_embed / w

        # Compute positional encodings for x and y
        dim_t = torch.arange(self.hidden_dim // 2, device=x.device).float()
        dim_t = 10000 ** (2 * (dim_t // 2) / self.hidden_dim)
        # print(dim_t.size())
        
        pos_x = x_embed.unsqueeze(-1) / dim_t
        pos_y = y_embed.unsqueeze(-1) / dim_t
        # print(pos_x.size(), pos_y.size())
        
        # pos_x = torch.stack((pos_x.sin(), pos_x.cos()), dim=-1).flatten(-2)
        # pos_y = torch.stack((pos_y.sin(), pos_y.cos()), dim=-1).flatten(-2)
        # print(pos_x.size(), pos_y.size())
        

        pos = torch.cat((pos_y, pos_x), dim=-1).permute(2, 0, 1)
        return pos.unsqueeze(0).repeat(b, 1, 1, 1)



# 2. Transformer Encoder-Decoder
class Transformer(nn.Module):
    def __init__(self, hidden_dim=256, nheads=8, num_encoder_layers=6, num_decoder_layers=6):
        super().__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nheads, ),
            num_layers=num_encoder_layers
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=nheads),
            num_layers=num_decoder_layers
        )

    def forward(self, src, pos, query_embed):
        b, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1) + pos.flatten(2).permute(2, 0, 1)
        memory = self.encoder(src)
        # print(memory.shape)
        query_embed = query_embed.unsqueeze(1).repeat(1, b, 1)
        tgt = torch.zeros_like(query_embed)

        hs = self.decoder(tgt, memory)
        # print(hs.transpose(0, 1).shape)
        return hs.transpose(0, 1)


# 3. DETR Model
class DETR(nn.Module):
    def __init__(self, num_classes=91, num_queries=100, hidden_dim=256, 
                 nheads=8, device='cuda', 
                 pointnet_model_path='pointnet/model_epoch_14.pth',
                 attention_type='simple'): # simple or context_guided
        super().__init__()
        self.attention_type = attention_type
        self.img_feat_dim = 2048
        self.lidar_feat_dim = 1024
        
        self.device = device
        # Move backbones
        self.backbone = self._create_backbone('resnet50', pretrained=True, device=self.device)
        # self.pc_backbone = PointNetCls(num_classes=num_classes).to(self.device)
        self.pointnet_model = PointNet(classes=num_classes)
        checkpoint = torch.load(pointnet_model_path)
        self.pointnet_model.load_state_dict(checkpoint['model_state_dict'])
        # Move all other components to the same device
        self.conv = nn.Conv2d(2048, hidden_dim, 1, 1).to(self.device)
        self.positional_encoding = PositionalEncoding(hidden_dim).to(self.device)
        
        if attention_type == 'simple':
            self.fusion_layer = nn.Conv2d(hidden_dim+self.lidar_feat_dim,
                                          hidden_dim, kernel_size=1).to(self.device)
        elif attention_type == 'context_guided':
            self.fusion_layer = ContextGuidedFusion(
                self.img_feat_dim, 
                self.lidar_feat_dim, 
                hidden_dim).to(self.device)
        
        self.transformer = Transformer(hidden_dim, nheads).to(self.device)
        
        self.num_queries = num_queries
        self.query_embed = nn.Embedding(num_queries, hidden_dim).to(self.device)
        
        self.dropout = nn.Dropout(0.2)
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1).to(self.device)
        self.bbox_embed = nn.Linear(hidden_dim, 4).to(self.device)

    def forward(self, x, point_cloud):
        # Ensure input is on the correct device
        x = x.to(self.device)
        
        # Only check parametric layers
        parametric_layers = ['conv1', 'bn1', 'layer1', 'layer2', 'layer3', 'layer4']
        for name, module in self.backbone.named_children():
            x = module(x)
        
        features = x  # Use the processed features
        
        # sys.exit()
        features = self.conv(features)
        pos = self.positional_encoding(features)
        point_features = self.extract_pointnet_features(point_cloud)
        B, C = point_features.shape
        
        point_features_expanded = point_features.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, features.shape[2], features.shape[3])
        # Fuse using concatenation
        fused_features = torch.cat([features, point_features_expanded], dim=1)
        if self.attention_type == 'simple':
            fused_features = self.fusion_layer(fused_features)
        elif self.attention_type == 'context_guided':
            fused_features = self.fusion_layer(features, point_features_expanded)
        hs = self.transformer(fused_features, pos, self.query_embed.weight)
        hs = self.dropout(hs)
        outputs_class = self.class_embed(hs)
        outputs_bbox = self.bbox_embed(hs).sigmoid()
        return outputs_class, outputs_bbox

    def _create_backbone(self, backbone_name, pretrained, device):
        if backbone_name == 'resnet50':
            from torchvision.models import resnet50, ResNet50_Weights
            
            backbone = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
            
            # Move to device
            backbone = backbone.to(device)
            
            backbone = nn.Sequential(OrderedDict([
                ('conv1', backbone.conv1),
                ('bn1', backbone.bn1),
                ('relu', backbone.relu),
                ('maxpool', backbone.maxpool),
                ('layer1', backbone.layer1),
                ('layer2', backbone.layer2),
                ('layer3', backbone.layer3),
                ('layer4', backbone.layer4)
            ]))
            
            return backbone
        else:
            raise ValueError(f"Backbone {backbone_name} not supported")
    
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
            global_features, _, _ = self.pointnet_model.transform(point_cloud)
        
        return global_features  # Shape: (B, 1024)

# 4. Post-Processing
def post_process(outputs_class, outputs_bbox, conf_threshold=0.7):
    probs = F.softmax(outputs_class[-1], dim=-1)
    scores, labels = probs[:, :-1].max(dim=-1)
    boxes = outputs_bbox[-1]

    # Filter predictions by confidence threshold
    keep = scores > conf_threshold
    scores = scores[keep]
    labels = labels[keep]
    boxes = boxes[keep]

    # Convert boxes to [x_min, y_min, x_max, y_max]
    boxes = box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy")
    return scores, labels, boxes


# 5. Hungarian Matcher
class HungarianMatcher:
    def __init__(self, class_weight=1.0, bbox_weight=5.0, giou_weight=2.0):
        """
        Initialize the matcher with weights for different components of the cost function.
        """
        self.class_weight = class_weight
        self.bbox_weight = bbox_weight
        self.giou_weight = giou_weight

    def compute_cost_matrix(self, outputs, targets):
        """
        Compute the cost matrix for matching predictions to ground truth.

        Args:
        - outputs: dict with 'pred_logits' and 'pred_boxes' (predictions).
        - targets: list of dicts with 'labels' and 'boxes' (ground truth).

        Returns:
        - cost_matrix: Tensor of shape [batch_size, num_queries, num_targets].
        """
        pred_logits = outputs["pred_logits"]  # [batch_size, num_queries, num_classes]
        pred_boxes = outputs["pred_boxes"]    # [batch_size, num_queries, 4]

        batch_size = pred_boxes.shape[0]  # Batch size

        cost_matrices = []  # Store cost matrices for each batch

        for b in range(batch_size):
            # Extract predictions and targets for the current batch
            pred_boxes_b = pred_boxes[b]  # [num_queries, 4]
            tgt_boxes_b = targets[b]["boxes"]  # [num_targets, 4]

            # Classification cost
            pred_probs_b = F.softmax(pred_logits[b], dim=-1)  # [num_queries, num_classes]
            tgt_labels_b = targets[b]["labels"]  # [num_targets]
            class_cost = -pred_probs_b[:, tgt_labels_b]  # Negative log-probability of correct class

            # Bounding box L1 cost
            bbox_cost = torch.cdist(pred_boxes_b, tgt_boxes_b, p=1)  # Pairwise L1 distance

            # GIoU cost
            giou = box_iou(pred_boxes_b, tgt_boxes_b)[1]  # Generalized IoU
            giou_cost = 1 - giou  # GIoU is maximized, so we use (1 - GIoU) as the cost

            # Combine costs with weights
            cost_matrix = (
                self.class_weight * class_cost
                + self.bbox_weight * bbox_cost
                + self.giou_weight * giou_cost
            )
            cost_matrices.append(cost_matrix)

        # Stack cost matrices for the entire batch
        return cost_matrices

    def match(self, outputs, targets):
        """
        Perform Hungarian matching using the cost matrix.

        Args:
        - outputs: dict with 'pred_logits' and 'pred_boxes' (predictions).
        - targets: list of dicts with 'labels' and 'boxes' (ground truth).

        Returns:
        - indices: List of tuples (query_indices, target_indices) for each batch.
        """
        cost_matrix = self.compute_cost_matrix(outputs, targets)

        batch_size = len(cost_matrix)
        indices = []
        for b in range(batch_size):
            # print('\n-------', targets[b])
            # Perform Hungarian matching for each batch
            query_indices, target_indices = linear_sum_assignment(cost_matrix[b].cpu().detach().numpy())
            # print(len(targets[b]), query_indices, target_indices)
            for i in target_indices:
                assert i<len(targets[b]['labels'])
            indices.append((torch.tensor(query_indices), torch.tensor(target_indices)))
            

        return indices


# 6. DETR Loss
class DETRLoss(nn.Module):
    def __init__(self, num_classes, class_weights):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = HungarianMatcher(class_weight=2.0, bbox_weight=8.0, giou_weight=4.0)
        self.class_weights = class_weights

    def forward(self, outputs_class, outputs_bbox, targets):
        # Prepare outputs and targets
        outputs = {
            "pred_logits": outputs_class,
            "pred_boxes": outputs_bbox
        }

        # Hungarian matching
        matched_indices = self.matcher.match(outputs, targets)

        # Loss initialization
        total_class_loss = 0.0
        total_bbox_loss = 0.0
        total_giou_loss = 0.0
        # print(targets[0]['labels'].size)
        # print(outputs_class)

        # Compute losses for matched pairs
            #each bach will have specific matches
        i=-1

        for (query_indices, target_indices) in matched_indices:
            i+=1
            pred_logits = outputs_class[i][query_indices]
            pred_boxes = outputs_bbox[i][query_indices]

            gt_labels = targets[i]["labels"][target_indices]
            gt_boxes = targets[i]["boxes"][target_indices]

            # Classification loss
            total_class_loss += F.cross_entropy(pred_logits, gt_labels, weight=self.class_weights)

            # Bounding box regression loss
            total_bbox_loss += F.l1_loss(pred_boxes, gt_boxes, reduction="mean")
            
            # Generalized IoU loss - ensure boxes are in correct format
            # Convert boxes to xyxy format if they aren't already
            pred_boxes_xyxy = box_convert(pred_boxes, in_fmt="cxcywh", out_fmt="xyxy")
            gt_boxes_xyxy = box_convert(gt_boxes, in_fmt="cxcywh", out_fmt="xyxy")
            
            # Calculate GIoU loss
            giou_loss = generalized_box_iou_loss(
                pred_boxes_xyxy,
                gt_boxes_xyxy,
                reduction="mean"
            )
            total_giou_loss += giou_loss

        # Average the losses over the batch
        batch_size = len(matched_indices)
        total_class_loss = total_class_loss / batch_size
        total_bbox_loss = total_bbox_loss / batch_size
        total_giou_loss = total_giou_loss / batch_size

        # Weight the different loss components
        weighted_class_loss = 2.0 * total_class_loss
        weighted_bbox_loss = 5.0 * total_bbox_loss
        weighted_giou_loss = 2.0 * total_giou_loss

        total_loss = weighted_class_loss + weighted_bbox_loss + weighted_giou_loss

        return total_loss, weighted_class_loss, weighted_bbox_loss, weighted_giou_loss


def process_targets(targets):
    """
    Convert target data into the required format for DETR loss function.

    Args:
        targets (list of tensors): Each tensor contains rows with class label 
                                   and bounding box coordinates as [label, x_min, y_min, x_max, y_max].
        image_height (int): Height of the input image.
        image_width (int): Width of the input image.

    Returns:
        list of dict: List of dictionaries with "labels" and "boxes" for each image in the batch.
    """
    processed_targets = []

    for image_targets in targets:
        # Extract class labels and bounding boxes
        labels = image_targets[:, 0].long()  # First column is the class label
        bboxes = image_targets[:, 1:]       # Remaining columns are bounding box coordinates [x_min, y_min, x_max, y_max]

        # Convert bounding boxes to [cx, cy, w, h] format
        # cx = (bboxes[:, 0] + bboxes[:, 2]) / 2  # Center x
        # cy = (bboxes[:, 1] + bboxes[:, 3]) / 2  # Center y
        # w = bboxes[:, 2] - bboxes[:, 0]         # Width
        # h = bboxes[:, 3] - bboxes[:, 1]         # Height

        # Normalize the bounding boxes to [0, 1]
        # cx /= image_width
        # cy /= image_height
        # w /= image_width
        # h /= image_height

        # Add to processed targets
        processed_targets.append({
            "labels": labels,
            "boxes": bboxes
        })

    return processed_targets


def compute_class_weights(train_loader):
    class_counts = torch.zeros(8)
    total_samples = 0
    
    print("Computing class weights...")
    for data in tqdm(train_loader.dataloader):
        for batch_idx in range(len(data['labels'])):
            labels = torch.vstack(data['labels'][batch_idx])[:, 0]
            for label in labels:
                class_counts[int(label)] += 1
                total_samples += 1
    
    # Avoid division by zero
    class_counts = torch.clamp(class_counts, min=1.0)
    
    # Inverse frequency weighting
    weights = total_samples / (len(class_counts) * class_counts)
    
    # Normalize weights
    weights = weights / weights.sum() * len(class_counts)
    
    print("Class distribution:", class_counts)
    print("Computed weights:", weights)
    
    return weights.to(device)


# MLflow configuration
# mlflow.set_tracking_uri("file:./mlruns")  # Local tracking
# experiment_name = "DETR_Training V2"
# mlflow.set_experiment(experiment_name)
# torch.random.manual_seed(28)



# # Configuration
# data_dir = "/home/sm2678/csci_739_term_project/CSCI739/data/"
# camera_dir = "left_images/{}/image_2"
# lidar_dir = "velodyne/{}/velodyne/"
# calib_dir = "calibration/{}/calib"
# label_dir = "labels/{}/label_2"
# mode = "training"
# num_classes = 8

# # Training parameters
# num_epochs = 300
# batch_size = 32
# learning_rate = 1e-4
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# num_queries = 12
# patience = 15  # for early stopping

# # Modified transform with augmentation
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     RandomHorizontalFlip(p=0.5),
#     ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
#     RandomAffine(degrees=0, translate=(0.1, 0.1)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])



# # Initialize dataset and dataloader
# train_dataset = PointCloudDataset(
#     data_dir, lidar_dir, camera_dir, calib_dir, label_dir,
#     num_points=50000,# Post voxelization num points
#     mode="training",
#     return_image=True,
#     return_calib=True,
#     return_labels=True,
#     image_transform=transform
# )
# wanted_labels = [0,4,5,6,7]


# train_loader = LazyDataLoader(
#     dataset=train_dataset,
#     batch_size=batch_size,
#     shuffle=True,
#     num_workers=12,
#     image_size=(224, 224),
#     label_data_loc=wanted_labels
# )

# # After initializing the train_loader, add validation loader
# val_dataset = PointCloudDataset(
#     data_dir, lidar_dir, camera_dir, calib_dir, label_dir,
#     num_points=50000,
#     mode="validation",  # Change to validation mode
#     return_image=True,
#     return_calib=True,
#     return_labels=True,
#     image_transform=transform
# )

# val_loader = LazyDataLoader(
#     dataset=val_dataset,
#     batch_size=batch_size,
#     shuffle=False,  # Don't shuffle validation data
#     num_workers=12,
#     image_size=(224, 224),
#     label_data_loc=wanted_labels
# )
# # Choose one of the weight initialization methods
# # class_weights = compute_class_weights(train_loader)  # Option 1
# # class_weights = torch.tensor([0.0317, 0.3124, 0.8328, 0.1980, 4.1238, 0.5767, 1.8501, 0.0744, 0.0001]).to(device)
# class_weights = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 0.1]).to(device)
# print("The class weights are computed: ", class_weights)
# # class_weights = get_predefined_weights()  # Option 2

# # Initialize model with device
# model = DETR(num_classes=num_classes, num_queries=num_queries, device=device)
# model = model.to(device)  # Ensures all parameters are on the same device
# criterion = DETRLoss(num_classes=num_classes, 
#             class_weights=class_weights).to(device)

# # Optimizer
# optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
# model.eval()

def compute_iou_matrix(pred_boxes, target_boxes):
    """
    Compute IoU between all pairs of boxes
    """
    return box_iou(pred_boxes, target_boxes)

def compute_metrics(all_pred_labels, all_true_labels):
    """Compute various classification metrics"""
    accuracy = accuracy_score(all_true_labels, all_pred_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_true_labels, 
        all_pred_labels, 
        average='weighted'
    )
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

# Modify validate function to return metrics
def validate(model, val_loader, criterion, device, epoch):
    model.eval()
    total_val_loss = 0.0
    all_pred_labels = []
    all_true_labels = []
    all_pred_boxes = []
    all_true_boxes = []
    all_pred_scores = []
    
    with torch.no_grad():
        progress_bar = tqdm(val_loader.dataloader, desc='Validation')
        for idx, data in enumerate(progress_bar):
            batch_label = []
            for batch_idx in range(len(data['labels'])):
                batch_label.append(torch.vstack(data['labels'][batch_idx]))

            images = data['images'].to(device)
            point_cloud = data['point_clouds'].to(device)
            batch_label = process_targets(batch_label) # give me output in the format of [x_min, y_min, x_max, y_max]
            for target in batch_label:
                target['labels'] = target['labels'].to(device)
                target['boxes'] = target['boxes'].to(device)

            outputs_class, outputs_bbox = model(images, point_cloud)
            total_loss, cl_loss, bo_loss, giou_loss = criterion(outputs_class, outputs_bbox, batch_label)
            total_val_loss += total_loss.item()

            # Post-process predictions
            for batch_idx in range(len(outputs_class)):
                pred_scores = F.softmax(outputs_class[batch_idx], dim=-1)
                pred_labels = torch.argmax(pred_scores, dim=-1)
                pred_boxes = outputs_bbox[batch_idx]
                
                target_labels = batch_label[batch_idx]['labels']
                target_boxes = batch_label[batch_idx]['boxes']

                # Compute IoU between predicted and target boxes
                iou_matrix = compute_iou_matrix(pred_boxes, target_boxes)
                
                # For each target box, find the predicted box with highest IoU
                for target_idx in range(len(target_labels)):
                    ious = iou_matrix[:, target_idx]
                    max_iou_idx = torch.argmax(ious)
                    
                    # If IoU > 0.5, consider it a match
                    
                    if ious[max_iou_idx] > 0.5:
                        all_pred_labels.append(pred_labels[max_iou_idx].cpu().item())
                        all_true_labels.append(target_labels[target_idx].cpu().item())
                        all_pred_boxes.append(pred_boxes[max_iou_idx].cpu().numpy())
                        all_true_boxes.append(target_boxes[target_idx].cpu().numpy())
                        all_pred_scores.append(pred_scores[max_iou_idx][pred_labels[max_iou_idx]].cpu().item())
                

    # Compute confusion matrix
    if all_pred_labels and all_true_labels:  # Check if we have any valid predictions
        class_names = {
            0: "Car",
            1: "Van",
            2: "Truck",
            3: "Pedestrian",
            4: "Person_sitting",
            5: "Cyclist",
            6: "Tram",
            7: "Misc"
        }
        cm = confusion_matrix(all_true_labels, all_pred_labels)
        print(cm)
        
        true_positives = np.zeros(len(all_pred_labels))
        for i in range(len(all_pred_labels)):
            if all_pred_labels[i] == all_true_labels[i]:
                true_positives[i] = 1
        # Convert lists to numpy arrays before calling ap_per_class
        true_positives = np.array(true_positives, dtype=np.float32)
        pred_scores = np.array(all_pred_scores, dtype=np.float32)
        pred_labels = np.array(all_pred_labels, dtype=np.int32)
        true_labels = np.array(all_true_labels, dtype=np.int32)

        # Calculate metrics using ap_per_class
        metrics_output = ap_per_class(
            true_positives,
            pred_scores,
            pred_labels,
            true_labels
        )
    
        # Print formatted evaluation statistics
        print_eval_stats(metrics_output, class_names)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title(f'Confusion Matrix (IoU > 0.5) - Epoch {epoch + 1}')
        plt.savefig(f'results/detr/fusedCAM/confusion_matrix_epoch_{epoch + 1}.png')
        plt.close()

    avg_val_loss = total_val_loss / len(val_loader.dataloader)
    metrics = {}
    # if all_pred_labels and all_true_labels:
        # Compute additional metrics
    metrics = compute_metrics(all_pred_labels, all_true_labels)

    
    return avg_val_loss, metrics

def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves
    # Arguments
        recall:    The recall curve (list)
        precision: The precision curve (list)
    # Returns
        Average precision, precision curve, recall curve
    """
    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap

def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # Convert inputs to numpy arrays if they're not already
    tp = np.array(tp)
    conf = np.array(conf)
    pred_cls = np.array(pred_cls)
    target_cls = np.array(target_cls)

    # Sort by objectness
    i = np.argsort(-conf)  # sort in descending order
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    for c in tqdm(unique_classes, desc="Computing AP"):
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects

        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum()
            tpc = (tp[i]).cumsum()

            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(recall_curve[-1])

            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(precision_curve[-1])

            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))

    # Compute F1 score (harmonic mean of precision and recall)
    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype("int32")

def print_eval_stats(metrics_output, class_names):
    """
    Print evaluation statistics in a formatted table
    Args:
        metrics_output: tuple of (precision, recall, AP, f1, ap_class)
        class_names: dictionary mapping class indices to class names
    """
    if metrics_output is not None:
        precision, recall, AP, f1, ap_class = metrics_output
        
        # Create table
        ap_table = [["Index", "Class", "AP", "Precision", "Recall", "F1"]]
        for i, c in enumerate(ap_class):
            class_name = class_names.get(c, f"Class_{c}")  # Get class name or default to Class_index
            ap_table.append([
                c,
                class_name,
                f"{AP[i]:.5f}",
                f"{precision[i]:.5f}",
                f"{recall[i]:.5f}",
                f"{f1[i]:.5f}"
            ])
        
        # Create and print table
        table_string = AsciiTable(ap_table).table
        print("\n---------- mAP per Class----------")
        print(table_string)
        print(f"\n---------- Total mAP {AP.mean():.5f} ----------\n")
        
    else:
        print("\n---- mAP not measured (no detections found by model) ----\n")

def inference_mode(model_path, val_loader, device, iou_threshold=0.5, hidden_dim=128):
    # Load model
    model = DETR(num_classes=8, num_queries=12, device=device, hidden_dim=hidden_dim)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)

    all_pred_labels = []
    all_true_labels = []
    all_pred_boxes = []
    all_true_boxes = []
    all_pred_scores = []
    
    with torch.no_grad():
        progress_bar = tqdm(val_loader.dataloader, desc='Inference')
        for idx, data in enumerate(progress_bar):
            batch_label = []
            for batch_idx in range(len(data['labels'])):
                batch_label.append(torch.vstack(data['labels'][batch_idx]))

            images = data['images'].to(device)
            point_cloud = data['point_clouds'].to(device)
            batch_label = process_targets(batch_label)
            for target in batch_label:
                target['labels'] = target['labels'].to(device)
                target['boxes'] = target['boxes'].to(device)
            outputs_class, outputs_bbox = model(images, point_cloud)
            # Post-process predictions
            for batch_idx in range(len(outputs_class)):
                pred_scores = F.softmax(outputs_class[batch_idx], dim=-1)
                pred_labels = torch.argmax(pred_scores, dim=-1)
                pred_boxes = outputs_bbox[batch_idx]

                
                target_labels = batch_label[batch_idx]['labels']
                target_boxes = batch_label[batch_idx]['boxes']

                # print(pred_boxes.shape, target_boxes.shape)
                # print(pred_boxes, target_boxes)
                # sys.exit()
                
                # Compute IoU between predicted and target boxes
                iou_matrix = compute_iou_matrix(pred_boxes, target_boxes)
                
                # For each target box, find the predicted box with highest IoU
                for target_idx in range(len(target_labels)):
                    ious = iou_matrix[:, target_idx]
                    max_iou_idx = torch.argmax(ious)
                    
                    # Only consider predictions with IoU > threshold
                    if ious[max_iou_idx] > iou_threshold:
                        all_pred_labels.append(pred_labels[max_iou_idx].cpu().item())
                        all_true_labels.append(target_labels[target_idx].cpu().item())
                        all_pred_boxes.append(pred_boxes[max_iou_idx].cpu().numpy())
                        all_true_boxes.append(target_boxes[target_idx].cpu().numpy())
                        all_pred_scores.append(pred_scores[max_iou_idx][pred_labels[max_iou_idx]].cpu().item())

    # Define class names (modify according to your classes)
    class_names = {
        0: "Car",
        1: "Van",
        2: "Truck",
        3: "Pedestrian",
        4: "Person_sitting",
        5: "Cyclist",
        6: "Tram",
        7: "Misc"
    }
    
    true_positives = np.zeros(len(all_pred_labels))
    for i in range(len(all_pred_labels)):
        if all_pred_labels[i] == all_true_labels[i]:
            true_positives[i] = 1
    # Convert lists to numpy arrays before calling ap_per_class
    true_positives = np.array(true_positives, dtype=np.float32)
    pred_scores = np.array(all_pred_scores, dtype=np.float32)
    pred_labels = np.array(all_pred_labels, dtype=np.int32)
    true_labels = np.array(all_true_labels, dtype=np.int32)

    # Calculate metrics using ap_per_class
    metrics_output = ap_per_class(
        true_positives,
        pred_scores,
        pred_labels,
        true_labels
    )
    
    # Print formatted evaluation statistics
    print_eval_stats(metrics_output, class_names)

    # Create confusion matrix
    cm = confusion_matrix(all_true_labels, all_pred_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title(f'Confusion Matrix (IoU > {iou_threshold})')
    plt.savefig(f'results/detr/fusedCAM/confusion_matrix_inference_iou_{iou_threshold}.png')
    plt.close()

    return {
        'precision': metrics_output[0],
        'recall': metrics_output[1],
        'ap': metrics_output[2],
        'f1': metrics_output[3],
        'mAP': metrics_output[2].mean(),
        'unique_classes': metrics_output[4]
    }, all_pred_boxes, all_true_boxes

def train_detr(data_dir, num_epochs=300, batch_size=32, learning_rate=1e-4, 
               num_queries=12, patience=15, best_model_path=None, 
               wanted_labels=[0,4,5,6,7], num_hidden_dim=256, 
               attention_type='simple', img_size=(1242, 375),):
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Directory setup
    camera_dir = "left_images/{}/image_2"
    lidar_dir = "velodyne/{}/velodyne/"
    calib_dir = "calibration/{}/calib"
    label_dir = "labels/{}/label_2"
    
    img_size = (1242, 375)
    
    # Transform setup
    transform = transforms.Compose([
        transforms.Resize(img_size),
        # RandomHorizontalFlip(p=0.5),
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        # RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(), # automatically scales the image to [0, 1] Do not apply external normalization
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Initialize datasets and dataloaders
    train_dataset = PointCloudDataset(
        data_dir, lidar_dir, camera_dir, calib_dir, label_dir,
        num_points=50000,
        mode="training",
        return_image=True,
        return_calib=True,
        return_labels=True,
        image_transform=transform
    )

    train_loader = LazyDataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=12,
        image_size=img_size,
        label_data_loc=wanted_labels
    )

    val_dataset = PointCloudDataset(
        data_dir, lidar_dir, camera_dir, calib_dir, label_dir,
        num_points=50000,
        mode="validation",
        return_image=True,
        return_calib=True,
        return_labels=True,
        image_transform=transform
    )

    val_loader = LazyDataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=12,
        image_size=img_size,
        label_data_loc=wanted_labels
    )

    # Model parameters
    class_weights = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 0.1]).to(device)
    num_classes = 8
    
    # Initialize model and criterion
    model = DETR(num_classes=num_classes, num_queries=num_queries, 
                 device=device, hidden_dim=num_hidden_dim, 
                 attention_type=attention_type)
    model = model.to(device)
    criterion = DETRLoss(num_classes=num_classes, 
                         class_weights=class_weights).to(device)
    
    # Initialize optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), 
                           lr=learning_rate, 
                           weight_decay=1e-4,  # Added weight decay
                           betas=(0.9, 0.999),  # Default AdamW betas
                           eps=1e-8)  # Default AdamW epsilon
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,  # reduce LR by half when plateauing
        patience=5,   # wait 5 epochs before reducing LR
        verbose=True,
        min_lr=1e-7
    )
    
    # Load previous model if exists
    start_epoch = 0
    best_val_loss = float('inf')
    if best_model_path is not None and os.path.exists(best_model_path):
        print(f"Loading best model from {best_model_path}")
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = 0
        best_val_loss = 100
        print(f"Resuming from epoch {start_epoch} with validation loss {best_val_loss}")
    
    # MLflow setup
    experiment_name = "DETR_Training V2"
    mlflow.set_experiment(experiment_name)
    
    # Training parameters to log
    params = {
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "num_classes": num_classes,
        "num_queries": num_queries,
        "hidden_dim": 256,
        "nheads": 8,
        "weight_decay": 1e-4,
        "wanted_labels": wanted_labels,
        "image_size": (224, 224),
        "class_weights": class_weights.tolist(),
        "attention_type": attention_type
    }
    
    # Training loop
    mlflow.end_run()
    with mlflow.start_run() as run:
        mlflow.log_params(params)
        mlflow.pytorch.log_model(model, "model")
        
        old_weights = None
        patience_counter = 0
        
        for epoch in range(start_epoch, num_epochs):
            if epoch == start_epoch:
                old_weights = {name: param.clone().detach() 
                              for name, param in model.named_parameters()}
            
            # Training phase
            model.train()
            running_loss = 0.0
            running_cl_loss = 0.0
            running_bo_loss = 0.0
            running_giou_loss = 0.0
            
            progress_bar = tqdm(train_loader.dataloader, desc='Training')
            
            for idx, data in enumerate(progress_bar):
                batch_label = []
                # print(data['labels'])
                for batch_idx in range(len(data['labels'])):
                    batch_label.append(torch.vstack(data['labels'][batch_idx]))
                # print(len(batch_label))
                images = data['images'].to(device)
                
                point_cloud = data['point_clouds'].to(device)
                batch_label = process_targets(batch_label)
                for target in batch_label:
                    target['labels'] = target['labels'].to(device)
                    target['boxes'] = target['boxes'].to(device)
                # print(target['boxes'])
                # sys.exit()
                # Regular training step without autocast
                outputs_class, outputs_bbox = model(images, point_cloud)
                total_loss, cl_loss, bo_loss, giou_loss = criterion(outputs_class, outputs_bbox, batch_label)
                total_loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                running_loss += total_loss.item()
                running_cl_loss += cl_loss.item()
                running_bo_loss += bo_loss.item()
                running_giou_loss += giou_loss.item()
            
            # Calculate average losses
            avg_train_loss = running_loss / len(train_loader.dataloader)
            avg_cl_loss = running_cl_loss / len(train_loader.dataloader)
            avg_bo_loss = running_bo_loss / len(train_loader.dataloader)
            avg_giou_loss = running_giou_loss / len(train_loader.dataloader)
            
            print(f"Epoch {epoch + 1}, Training Loss: {avg_train_loss:.4f}, "
                  f"cl_loss: {avg_cl_loss:.4f}, bo_loss: {avg_bo_loss:.4f}, "
                  f"giou_loss: {avg_giou_loss:.4f}")
            
            # Log metrics
            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
            mlflow.log_metric("Avg_cl_loss", avg_cl_loss, step=epoch)
            mlflow.log_metric("Avg_bo_loss", avg_bo_loss, step=epoch)
            mlflow.log_metric("Avg_giou_loss", avg_giou_loss, step=epoch)
            
            # Scheduler step with training loss
            scheduler.step(avg_train_loss)
            current_lr = optimizer.param_groups[0]['lr']
            mlflow.log_metric("learning_rate", current_lr, step=epoch)
            
            # Validation phase
            if (epoch + 1) % 1 == 0:
                val_loss, val_metrics = validate(model, val_loader, criterion, device, epoch)
                print(f"Validation Loss: {val_loss:.4f}, Validation Metrics: {val_metrics}")
                
                # Calculate mAP if available in metrics
                current_map = val_metrics.get("f1", val_loss)  # Use F1 as fallback if mAP not available
                
                # Log metrics including current learning rate
                current_lr = optimizer.param_groups[0]['lr']
                mlflow.log_metric("val_loss", val_loss, step=epoch)
                mlflow.log_metric("learning_rate", current_lr, step=epoch)
                mlflow.log_metric("val_accuracy", val_metrics["accuracy"], step=epoch)
                mlflow.log_metric("val_precision", val_metrics["precision"], step=epoch)
                mlflow.log_metric("val_recall", val_metrics["recall"], step=epoch)
                mlflow.log_metric("val_f1", val_metrics["f1"], step=epoch)
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'val_loss': val_loss,
                        'metrics': val_metrics
                    }, f'best_model_4_fusedCAM_{epoch + 1}.pth')
                else:
                    patience_counter += 1
                
                # Early stopping check
                if patience_counter >= patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs")
                    break
    
    return model, best_val_loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DETR Training and Inference')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'inference'],
                      help='Mode to run the model (train or inference)')
    parser.add_argument('--data_dir', type=str, 
                      default="/home/sm2678/csci_739_term_project/CSCI739/data/",
                      help='Path to data directory')
    parser.add_argument('--model_path', type=str, default='best_model_3_ep80.pth',
                      help='Path to model checkpoint (required for inference)')
    parser.add_argument('--batch_size', type=int, default=8,
                      help='Batch size for training/inference')
    parser.add_argument('--num_queries', type=int, default=12,
                      help='Number of object queries')
    parser.add_argument('--iou_threshold', type=float, default=0.5,
                      help='IoU threshold for inference')
    parser.add_argument('--num_hidden_dim', type=int, default=256,
                      help='Number of hidden dimensions')
    parser.add_argument('--attention_type', type=str, default='simple',
                      help='Attention type (simple or context_guided)')
    
    args = parser.parse_args()
    img_size = (1242, 375)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Setup transform
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    if args.mode == 'train':
        print("Starting training mode...")
        model, best_val_loss = train_detr(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_queries=args.num_queries,
            num_hidden_dim=args.num_hidden_dim,
            attention_type=args.attention_type,
            img_size=img_size,
            best_model_path=args.model_path
        )
        print(f"Training completed. Best validation loss: {best_val_loss:.4f}")
        
    else:  # Inference mode
        print("Starting inference mode...")
        # Initialize validation dataset and loader
        val_dataset = PointCloudDataset(
            args.data_dir,
            "velodyne/{}/velodyne/",
            "left_images/{}/image_2",
            "calibration/{}/calib",
            "labels/{}/label_2",
            num_points=50000,
            mode="validation",
            return_image=True,
            return_calib=True,
            return_labels=True,
            image_transform=transform
        )

        val_loader = LazyDataLoader(
            dataset=val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=12,
            image_size=img_size,
            label_data_loc=[0,4,5,6,7]
        )

        # Run inference
        metrics, pred_boxes, true_boxes = inference_mode(
            model_path=args.model_path,
            val_loader=val_loader,
            device=device,
            iou_threshold=args.iou_threshold,
            hidden_dim=args.num_hidden_dim,
            attention_type=args.attention_type
        )

# nohup bash -c "CUDA_VISIBLE_DEVICES=5 python train_detr_v4_2.py" > logs/train_detr_v4_fusedCAM.txt 2>&1 &