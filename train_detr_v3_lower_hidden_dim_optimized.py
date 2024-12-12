import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from kitti_data_handler import PointCloudDataset, LazyDataLoader
import sys
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F
import sys
import mlflow
import mlflow.pytorch
from sklearn.metrics import confusion_matrix
import seaborn as sns
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.transforms import ColorJitter
import os
import argparse
from torchvision.ops import box_convert
import time
# ------------------------------
from detr.loss import DETRLoss
from detr.model_components import DETR
from detr.trainer_utils import ap_per_class, print_eval_stats, process_targets
from detr.trainer_utils import compute_iou_matrix, compute_metrics

# Modify validate function to return metrics
def validate(model, val_loader, criterion, device, epoch, iou_threshold):
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
            batch_label = process_targets(batch_label)
            for target in batch_label:
                target['labels'] = target['labels'].to(device)
                target['boxes'] = target['boxes'].to(device)

            outputs_class, outputs_bbox = model(images)
            total_loss, cl_loss, bo_loss, giou_loss = criterion(outputs_class, outputs_bbox, batch_label)
            total_val_loss += total_loss.item()

            # Post-process predictions
            for batch_idx in range(len(outputs_class)):
                pred_scores = F.softmax(outputs_class[batch_idx], dim=-1)
                pred_labels = torch.argmax(pred_scores, dim=-1)
                pred_boxes = box_convert(outputs_bbox[batch_idx], in_fmt="cxcywh", out_fmt="xyxy")
                
                target_labels = batch_label[batch_idx]['labels']
                target_boxes = box_convert(batch_label[batch_idx]['boxes'], in_fmt="cxcywh", out_fmt="xyxy")

                # Compute IoU between predicted and target boxes
                iou_matrix = compute_iou_matrix(pred_boxes, target_boxes)
                
                # For each target box, find the predicted box with highest IoU
                for target_idx in range(len(target_labels)):
                    ious = iou_matrix[:, target_idx]
                    max_iou_idx = torch.argmax(ious)
                    
                    # If IoU > 0.5, consider it a match
                    
                    if ious[max_iou_idx] >= iou_threshold:
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
        plt.savefig(f'results/detr/final_model/confusion_matrix_epoch_{epoch + 1}.png')
        plt.close()

    avg_val_loss = total_val_loss / len(val_loader.dataloader)
    metrics = {}
    # if all_pred_labels and all_true_labels:
        # Compute additional metrics
    metrics = compute_metrics(all_pred_labels, all_true_labels)

    
    return avg_val_loss, metrics


def inference_mode(model_path, val_loader, device, iou_threshold=0.5, num_hidden_dim=128, num_classes=8):
    # Load model
    model = DETR(num_classes=num_classes, num_queries=12, device=device, hidden_dim=num_hidden_dim)
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
            batch_label = process_targets(batch_label)
            for target in batch_label:
                target['labels'] = target['labels'].to(device)
                target['boxes'] = target['boxes'].to(device)
            outputs_class, outputs_bbox = model(images)
            # Post-process predictions
            for batch_idx in range(len(outputs_class)):
                pred_scores = F.softmax(outputs_class[batch_idx], dim=-1)
                pred_labels = torch.argmax(pred_scores, dim=-1)
                pred_boxes = outputs_bbox[batch_idx]

                # Convert boxes to xyxy format if they aren't already
                pred_boxes_xyxy = box_convert(pred_boxes, in_fmt="cxcywh", out_fmt="xyxy")
                # gt_boxes_xyxy = box_convert(gt_boxes, in_fmt="cxcywh", out_fmt="xyxy")
                
                target_labels = batch_label[batch_idx]['labels']
                target_boxes = box_convert(batch_label[batch_idx]['boxes'], in_fmt="cxcywh", out_fmt="xyxy")
                
                print('\n\n Pred Boxes',(pred_boxes_xyxy))
                print('\n\n Target Boxes',(target_boxes))
                
                # sys.exit()

                # print(pred_boxes.shape, target_boxes.shape)
                # print(pred_boxes, target_boxes)
                # sys.exit()
                
                # Compute IoU between predicted and target boxes
                # ALERT: Expects the boxes to be in xyxy format
                iou_matrix = compute_iou_matrix(pred_boxes_xyxy, target_boxes)
                print('\n\n IoU Matrix',(iou_matrix))
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
    plt.savefig(f'results/detr/final_model/confusion_matrix_inference_iou_{iou_threshold}.png')
    plt.close()

    return {
        'precision': metrics_output[0],
        'recall': metrics_output[1],
        'ap': metrics_output[2],
        'f1': metrics_output[3],
        'mAP': metrics_output[2].mean(),
        'unique_classes': metrics_output[4]
    }, all_pred_boxes, all_true_boxes


def train_detr(data_dir, num_epochs=300, batch_size=32, learning_rate=1e-4, num_queries=12, patience=15, 
               best_model_path=None, wanted_labels=[0,4,5,6,7], num_hidden_dim=256, iou_threshold=0.1):
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Directory setup
    camera_dir = "left_images/{}/image_2"
    lidar_dir = "velodyne/{}/velodyne/"
    calib_dir = "calibration/{}/calib"
    label_dir = "labels/{}/label_2"
    
    image_size = (375, 1242)
    
    # Transform setup
    transform = transforms.Compose([
        transforms.Resize(image_size),
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
        image_size=image_size,
        label_data_loc=wanted_labels,
        pin_memory=True,
        persistent_workers=True,
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
        image_size=image_size,
        label_data_loc=wanted_labels,
        pin_memory=True,
        persistent_workers=True,
    )

    # Model parameters
    class_weights = torch.tensor([1, 1, 1, 0.1]).to(device)
    num_classes = 3
    
    # Initialize model and criterion
    model = DETR(num_classes=num_classes, num_queries=num_queries, device=device, hidden_dim=num_hidden_dim)
    model = model.to(device)
    criterion = DETRLoss(num_classes=num_classes, class_weights=class_weights).to(device)
    
    # Optimizer and scheduler setup
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
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
        "image_size": (1242, 375),
        "class_weights": class_weights.tolist()
    }
    
    # Training loop
    mlflow.end_run()
    with mlflow.start_run() as run:
        mlflow.log_params(params)
        mlflow.pytorch.log_model(model, "model")
        
        old_weights = None
        patience_counter = 0
        
        for epoch in range(start_epoch, num_epochs):
            
            # Training phase
            model.train()
            running_loss = 0.0
            running_cl_loss = 0.0
            running_bo_loss = 0.0
            running_giou_loss = 0.0
            
            progress_bar = tqdm(train_loader.dataloader, desc='Training')
            
            for idx, data in enumerate(progress_bar):
                batch_label = []
                batch_fb = []
                # print(data['labels'], data['labels'][0].shape)
                # sys.exit()
                for batch_idx in range(len(data['labels'])):
                    batch_label.append(torch.vstack(data['labels'][batch_idx]))
                images = data['images'].to(device)
                batch_label = process_targets(batch_label)
                for target in batch_label:
                    target['labels'] = target['labels'].to(device)
                    target['boxes'] = target['boxes'].to(device)

                outputs_class, outputs_bbox = model(images)# bs, num_queries, num_classes
                total_loss, cl_loss, bo_loss, giou_loss = criterion(outputs_class, outputs_bbox, batch_label)# batch_label as cxcywh

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

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
            
            # Scheduler step
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            mlflow.log_metric("learning_rate", current_lr, step=epoch)
            
            # Validation phase
            if (epoch + 1) % 1 == 0:
                val_loss, val_metrics = validate(model, val_loader, criterion, device, epoch, iou_threshold)
                print(f"Validation Loss: {val_loss:.4f}, Validation Metrics: {val_metrics}")
                
                # Log validation metrics
                mlflow.log_metric("val_loss", val_loss, step=epoch)
                mlflow.log_metric("val_accuracy", val_metrics["accuracy"], step=epoch)
                mlflow.log_metric("val_precision", val_metrics["precision"], step=epoch)
                mlflow.log_metric("val_recall", val_metrics["recall"], step=epoch)
                mlflow.log_metric("val_f1", val_metrics["f1"], step=epoch)
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    save_path = best_model_path.split('.')[0]+f'_{time.strftime("%Y%m%d_%H%M%S")}.pth'
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'val_loss': val_loss,
                        'metrics': val_metrics
                    }, save_path)
                    print(f"New best model saved at {save_path}")
                else:
                    print(f"Validation loss did not improve. Patience counter: {patience_counter}")
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
    parser.add_argument('--iou_threshold', type=float, default=0.1,
                      help='IoU threshold for inference')
    parser.add_argument('--num_hidden_dim', type=int, default=256,
                      help='Number of hidden dimensions')
    parser.add_argument('--num_classes', type=int, default=8,
                      help='Number of classes')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Setup transform
    transform = transforms.Compose([
        transforms.Resize((1242, 375)),
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
            best_model_path=args.model_path,
            patience=40,
            num_epochs=400,
            iou_threshold=args.iou_threshold
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
            image_size=(224, 224),
            label_data_loc=[0,4,5,6,7]
        )

        # Run inference
        metrics, pred_boxes, true_boxes = inference_mode(
            model_path=args.model_path,
            val_loader=val_loader,
            device=device,
            iou_threshold=args.iou_threshold,
            num_hidden_dim=args.num_hidden_dim,
            num_classes=args.num_classes
        )