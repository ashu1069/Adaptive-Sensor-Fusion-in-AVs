import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import box_iou, box_convert
from scipy.optimize import linear_sum_assignment
from torchvision.ops import generalized_box_iou_loss
from detr.fb_utils import box_cxcywh_to_xyxy, generalized_box_iou
import sys
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
            
            
            # print(bbox_cost,pred_boxes_b, tgt_boxes_b)
            # GIoU cost
            # ToDO: check if this is correct
            giou = generalized_box_iou_loss(pred_boxes_b, tgt_boxes_b, reduction="mean")  # Generalized IoU
            giou_cost = 1- giou  # GIoU is maximized, so we use (1 - GIoU) as the cost

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



class DETRLoss(nn.Module):
    '''
    DETR Loss function which uses Hungarian matching to match predictions to ground truth
    params:
        num_classes: number of classes
        class_weights: weights for each class
        use_FB_hm: whether to use FB's Hungarian matcher
    '''
    def __init__(self, num_classes, class_weights, use_FB_hm=True):
        super().__init__()
        self.num_classes = num_classes
        if not use_FB_hm:
            self.matcher = HungarianMatcher(class_weight=2.0, bbox_weight=8.0, giou_weight=4.0)
        else:
            self.matcher = build_matcher(cost_class=1, cost_bbox=1, cost_giou=1)
        
        self.class_weights = class_weights

    def forward(self, outputs_class, outputs_bbox, targets):
        '''
        Forward pass for DETR loss
        params:
            outputs_class: predicted class logits
            outputs_bbox: predicted bounding box coordinates in cxcywh format
            targets: ground truth labels and boxes in cxcywh format
        '''
        # Prepare outputs and targets
        outputs = {
            "pred_logits": outputs_class,
            "pred_boxes": outputs_bbox
        }

        # Hungarian matching
        matched_indices = self.matcher.forward(outputs, targets)
        # print(matched_indices)
        # sys.exit()
        # Loss initialization
        total_class_loss = 0.0
        total_bbox_loss = 0.0
        total_giou_loss = 0.0

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
            # pred_boxes_xyxy = box_convert(pred_boxes, in_fmt="cxcywh", out_fmt="xyxy")
            # gt_boxes_xyxy = box_convert(gt_boxes, in_fmt="cxcywh", out_fmt="xyxy")
            
            # Calculate GIoU loss
            # print(pred_boxes, gt_boxes)
            # print(generalized_box_iou(pred_boxes, gt_boxes))
            # sys.exit()
            giou_loss = 1- generalized_box_iou_loss(
                pred_boxes,
                gt_boxes,
                reduction="mean"
            )
            
            total_giou_loss += giou_loss

        # Average the losses over the batch
        batch_size = len(matched_indices)
        total_class_loss = total_class_loss / batch_size
        total_bbox_loss = total_bbox_loss / batch_size
        total_giou_loss = total_giou_loss / batch_size

        # Weight the different loss components
        weighted_class_loss = total_class_loss
        weighted_bbox_loss = total_bbox_loss
        weighted_giou_loss = total_giou_loss

        total_loss = weighted_class_loss + weighted_bbox_loss + weighted_giou_loss
        print('total_loss', total_loss.item(), 'weighted_class_loss', 
              weighted_class_loss.item(), 'weighted_bbox_loss', weighted_bbox_loss.item(), 
              'weighted_giou_loss', weighted_giou_loss.item())
        return total_loss, weighted_class_loss, weighted_bbox_loss, weighted_giou_loss


class HungarianMatcher_FB:
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        # print(outputs["pred_logits"].shape)
        # print(outputs["pred_boxes"].shape)
        # print(targets[0], len(targets))
        # sys.exit()
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes
        # print(out_bbox)
        # print(box_cxcywh_to_xyxy(out_bbox))
        # print(box_cxcywh_to_xyxy(out_bbox))
        # sys.exit()
        cost_giou = -generalized_box_iou(
            box_cxcywh_to_xyxy(out_bbox), 
            box_cxcywh_to_xyxy(tgt_bbox)
        )
        
        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(cost_class=1, cost_bbox=5, cost_giou=2):
        # return HungarianMatcher(cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou)
        return HungarianMatcher_FB(cost_class, cost_bbox, cost_giou)
