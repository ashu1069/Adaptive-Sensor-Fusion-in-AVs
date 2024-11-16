import os
import sys
# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.resnet_detection_head import DetectionHead

class EvidentialDetectionHead(DetectionHead):
    def __init__(self, num_classes=3, confidence_threshold=0.5):
        super(EvidentialDetectionHead, self).__init__(num_classes, confidence_threshold)
        
        # Modify the output layers for evidential learning
        # Classification: Output evidence for Dirichlet distribution
        self.classification = nn.Linear(128, num_classes)
        
        # Regression: Output parameters for NIG distribution
        # For each bbox coordinate (x,y,w,h), we need (μ, v, α, β)
        self.bbox_regression = nn.Linear(128, 4 * 4)  # 4 parameters for each of the 4 coordinates
        
    def compute_classification_uncertainty(self, evidence):
        """
        Compute uncertainty metrics for classification using Dirichlet distribution
        """
        # Convert evidence to alpha parameters (α = e + 1)
        alpha = evidence + 1
        
        # Calculate strength of evidence
        S = torch.sum(alpha, dim=1, keepdim=True)
        
        # Calculate expected probabilities
        prob = alpha / S
        
        # Calculate uncertainty (total variance)
        uncertainty = alpha.shape[1] / S
        
        return prob, uncertainty
        
    def compute_regression_uncertainty(self, nig_params):
        """
        Compute uncertainty metrics for regression using NIG distribution
        """
        # Split parameters
        mu, v, alpha, beta = torch.chunk(nig_params, 4, dim=1)
        
        # Ensure positive values for v, alpha, beta
        v = F.softplus(v)
        alpha = F.softplus(alpha) + 1  # alpha > 1
        beta = F.softplus(beta)
        
        # Calculate epistemic uncertainty (variance)
        epistemic_uncertainty = beta / (v * (alpha - 1))
        
        # Calculate aleatoric uncertainty
        aleatoric_uncertainty = 2 * beta * (1 + v) / (v * (alpha - 1))
        
        return mu, epistemic_uncertainty, aleatoric_uncertainty

    def forward(self, point_cloud=None, image=None):
        """
        Forward pass with uncertainty estimation
        """
        # Get fused features using parent class
        lidar_features = None
        image_features = None
        
        if point_cloud is not None:
            lidar_features, _, _ = self.pointnet(point_cloud)
            
        if image is not None:
            image_features = self.resnet(image)
        
        fused_features = self.fusion_module(
            lidar_features=lidar_features,
            image_features=image_features
        )
        
        # Pass through detection layers
        shared_features = self.detection_layers(fused_features)
        
        # Get evidential outputs
        evidence = F.relu(self.classification(shared_features))  # Evidence must be positive
        nig_params = self.bbox_regression(shared_features)
        
        # Compute uncertainties
        class_probs, class_uncertainty = self.compute_classification_uncertainty(evidence)
        bbox_pred, bbox_epistemic, bbox_aleatoric = self.compute_regression_uncertainty(nig_params)
        
        return {
            'class_probs': class_probs,
            'class_uncertainty': class_uncertainty,
            'bbox_pred': bbox_pred,
            'bbox_epistemic_uncertainty': bbox_epistemic,
            'bbox_aleatoric_uncertainty': bbox_aleatoric,
            'evidence': evidence,
            'nig_params': nig_params
        }

    def evidential_loss(self, outputs, targets, class_labels):
        """
        Compute evidential loss for both classification and regression
        """
        # Classification loss (Dirichlet NLL)
        evidence = outputs['evidence']
        alpha = evidence + 1
        S = torch.sum(alpha, dim=1, keepdim=True)
        
        class_loss = torch.sum(
            class_labels * (torch.digamma(S) - torch.digamma(alpha)), dim=1
        )
        
        # Regression loss (NIG NLL)
        nig_params = outputs['nig_params']
        mu, v, alpha, beta = torch.chunk(nig_params, 4, dim=1)
        
        v = F.softplus(v)
        alpha = F.softplus(alpha) + 1
        beta = F.softplus(beta)
        
        reg_loss = 0.5 * torch.log(np.pi / v) \
                  - alpha * torch.log(beta) \
                  + (alpha + 0.5) * torch.log(
                      v * (targets - mu)**2 + beta
                  ) \
                  + torch.lgamma(alpha) \
                  - (alpha - 1) * (torch.digamma(alpha) - torch.log(beta))
        
        return class_loss.mean() + reg_loss.mean()

if __name__ == "__main__":
    # Example usage
    batch_size = 4
    point_cloud = torch.randn(batch_size, 3, 1024)
    image = torch.randn(batch_size, 3, 224, 224)
    
    # Initialize evidential detection head
    detection_head = EvidentialDetectionHead(num_classes=3)
    
    # Test forward pass
    outputs = detection_head(point_cloud, image)
    
    print("\nEvidential Detection Head outputs:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"{key} shape: {value.shape}")
