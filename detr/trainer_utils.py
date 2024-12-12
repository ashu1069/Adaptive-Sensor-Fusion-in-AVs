import numpy as np
from tqdm import tqdm
from terminaltables import AsciiTable
from torchvision.ops import box_convert
from torchvision.ops import box_iou
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch

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
            'labels': Tensor of class labels
            'boxes': Tensor of bounding boxes in [x_min, y_min, x_max, y_max] format
    """
    processed_targets = []

    for image_targets in targets:
        # Extract class labels and bounding boxes
        labels = image_targets[:, 0].long()  # First column is the class label
        bboxes = image_targets[:, 1:]       # Remaining columns are bounding box coordinates [x_min, y_min, x_max, y_max]
        
        bboxes = box_convert(bboxes, in_fmt="xyxy", out_fmt="cxcywh")
        
        # Add to processed targets
        processed_targets.append({
            "labels": labels,
            "boxes": bboxes
        })

    return processed_targets

def compute_class_weights(train_loader, device='cuda'):
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

def filter_valid_labels(labels, bboxes, label_mapping):
    """
    Filter out labels that are not in the label mapping and their corresponding bboxes.
    
    Args:
        labels (torch.Tensor): Tensor of label indices
        bboxes (torch.Tensor): Tensor of bounding boxes corresponding to labels
        label_mapping (dict): Dictionary mapping label names/ids to indices
        
    Returns:
        tuple: Filtered (labels, bboxes) tensors with only valid labels and their bboxes
    """
    valid_mask = torch.tensor([int(l) in label_mapping.values() for l in labels])
    
    filtered_labels = labels[valid_mask]
    filtered_bboxes = bboxes[valid_mask]
    
    return filtered_labels, filtered_bboxes

if __name__ == "__main__":
    mapping = {
        "Car": 0,
        "Pedestrian": 1,
        "Cyclist": 2
    }
    labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
    bboxes = torch.tensor([
        [0, 0, 1, 1], 
        [1, 1, 2, 2], 
        [2, 2, 3, 3], 
        [3, 3, 4, 4], 
        [4, 4, 5, 5], 
        [5, 5, 6, 6], 
        [6, 6, 7, 7], 
        [7, 7, 8, 8]
    ])
    filtered_labels, filtered_bboxes = filter_valid_labels(labels, bboxes, mapping)
    print(filtered_labels, filtered_bboxes)