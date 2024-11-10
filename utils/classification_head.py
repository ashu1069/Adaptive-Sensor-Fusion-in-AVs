import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassificationHead(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, num_classes=10, dropout_rate=0.5):
        super(ClassificationHead, self).__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x):
        """
        Args:
            x: Fused features from AdaptiveFusion module (batch_size, input_dim)
        Returns:
            logits: Classification logits (batch_size, num_classes)
        """
        logits = self.classifier(x)
        return logits

if __name__ == "__main__":
    # Test the classification head
    batch_size = 4
    input_dim = 512
    num_classes = 10
    
    # Create sample input (simulating fused features)
    fused_features = torch.randn(batch_size, input_dim)
    
    # Initialize classification head
    classifier = ClassificationHead(input_dim=input_dim, num_classes=num_classes)
    
    # Forward pass
    logits = classifier(fused_features)
    
    print("\nClassification Head test:")
    print(f"Input shape: {fused_features.shape}")
    print(f"Output logits shape: {logits.shape}")
    
    # Test with softmax probabilities
    probs = F.softmax(logits, dim=1)
    print(f"Output probabilities shape: {probs.shape}")
