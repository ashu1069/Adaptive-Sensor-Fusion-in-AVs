import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

class ImageFeatureExtractor(nn.Module):
    def __init__(self, output_dim=2048):
        super(ImageFeatureExtractor, self).__init__()
        # Load pretrained ResNet50
        resnet = models.resnet50(pretrained=True)
        
        # Remove the final fully connected layer
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        # Freeze all parameters
        for param in self.features.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        # Extract features
        features = self.features(x)
        # Flatten the features
        features = features.view(features.size(0), -1)
        return features

# Define image preprocessing
def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

# Example usage
def extract_features(image_path):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model
    model = ImageFeatureExtractor().to(device)
    model.eval()
    
    # Load and preprocess image
    transform = get_transform()
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    
    # Extract features
    with torch.no_grad():
        features = model(image)
        
    return features

# Example
if __name__ == "__main__":
    # Replace with your image path
    image_path = "samples/image_left/000044.png"
    features = extract_features(image_path)
    print(f"Extracted features shape: {features.shape}")  # Should be torch.Size([1, 2048])