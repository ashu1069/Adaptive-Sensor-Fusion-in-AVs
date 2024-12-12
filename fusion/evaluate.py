import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from dataloader import KITTIDataset, collate_fn
from model import MultiModalDetector
from args import get_args

class Evaluator:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Class names
        self.class_names = ['Car', 'Van', 'Truck', 'Pedestrian', 
                           'Person_sitting', 'Cyclist', 'Tram', 'DontCare']
        
        # Load dataset
        self.dataset = KITTIDataset(
            root_dir=args.data_dir,
            split='training',  # or 'testing' if you have a test set
            yolo_path=args.yolo_path,
            pointnet_path=args.pointnet_path
        )
        
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn
        )
        
        # Initialize and load model
        self.model = MultiModalDetector(num_classes=args.num_classes).to(self.device)
        self.load_model()
        
        # Add a flag for sample evaluation
        self.sample_size = 4
    
    def load_model(self):
        if not os.path.exists(self.args.model_path):
            raise FileNotFoundError(f"Model file not found: {self.args.model_path}")
        
        checkpoint = torch.load(self.args.model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from epoch {checkpoint['epoch']}")
    
    def plot_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(os.path.join(self.args.output_dir, 'confusion_matrix.png'))
        plt.close()
    
    def evaluate(self):
        self.model.eval()
        all_predictions = []
        all_targets = []
        all_confidences = []
        
        print("\nRunning evaluation...")
        with torch.no_grad():
            for batch in tqdm(self.dataloader, desc='Evaluating'):
                print("Processing batch...")
                # Move data to device
                lidar_features = batch['lidar_features'].to(self.device)
                image_features = [feat.to(self.device) for feat in batch['image_features']]
                targets = batch['targets'].to(self.device)
                
                # Forward pass
                _, _, predictions = self.model(
                    lidar_features, image_features, targets
                )
                
                # Process predictions for each scale
                for scale_predictions in predictions:
                    for batch_pred in scale_predictions:
                        if batch_pred is not None and len(batch_pred) > 0:  # Check if there are any predictions
                            # batch_pred contains [boxes, scores, classes]
                            pred_classes = batch_pred[:, -1]  # Get class predictions
                            confidences = batch_pred[:, -2]   # Get confidence scores
                            
                            # Store results
                            all_predictions.extend(pred_classes.cpu().numpy())
                            all_confidences.extend(confidences.cpu().numpy())
                
                # Store targets (only valid ones)
                valid_targets = targets[targets.sum(dim=1) != 0]  # Filter out zero-padded targets
                if len(valid_targets) > 0:
                    all_targets.extend(valid_targets[:, 0].cpu().numpy())
                
                print("Batch processed.")
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        all_confidences = np.array(all_confidences)

        # Ensure we have predictions
        if len(all_predictions) == 0:
            print("Warning: No predictions were made!")
            return

        # Calculate metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            all_targets, 
            all_predictions, 
            labels=range(len(self.class_names)),
            zero_division=0
        )
        
        # Print detailed results
        print("\nDetailed Evaluation Results:")
        print("\nClass-wise Metrics:")
        print("Class            Precision  Recall     F1        Support")
        print("-" * 55)
        
        for i, class_name in enumerate(self.class_names):
            print(f"{class_name:<15} {precision[i]:9.3f} {recall[i]:9.3f} {f1[i]:9.3f} {support[i]:9d}")
        
        # Calculate and print macro and weighted averages
        macro_precision = np.mean(precision)
        macro_recall = np.mean(recall)
        macro_f1 = np.mean(f1)
        weighted_precision = np.average(precision, weights=support)
        weighted_recall = np.average(recall, weights=support)
        weighted_f1 = np.average(f1, weights=support)
        total_support = sum(support)
        
        print("-" * 55)
        print(f"Macro Average   {macro_precision:9.3f} {macro_recall:9.3f} {macro_f1:9.3f} {total_support:9d}")
        print(f"Weighted Avg    {weighted_precision:9.3f} {weighted_recall:9.3f} {weighted_f1:9.3f} {total_support:9d}")
        
        # Calculate and print average confidence
        avg_confidence = np.mean(all_confidences)
        print(f"\nAverage Prediction Confidence: {avg_confidence:.3f}")
        
        # Plot and save confusion matrix
        self.plot_confusion_matrix(all_targets, all_predictions)
        print(f"\nConfusion matrix has been saved to {self.args.output_dir}/confusion_matrix.png")
    
    def evaluate_samples(self):
        self.model.eval()
        sample_results = []
        
        print("\nEvaluating sample images...")
        with torch.no_grad():
            # Get just one batch
            batch = next(iter(self.dataloader))
            
            # Move data to device
            lidar_features = batch['lidar_features'].to(self.device)
            image_features = [feat.to(self.device) for feat in batch['image_features']]
            targets = batch['targets'].to(self.device)
            image_paths = batch['image_path']  # Get image paths
            
            # Forward pass
            _, _, predictions = self.model(
                lidar_features, image_features, targets
            )
            
            # Process only the first 4 samples
            for idx in range(min(self.sample_size, len(image_paths))):
                sample_dict = {
                    'image_path': image_paths[idx],
                    'predictions': [],
                    'confidences': [],
                    'targets': []
                }
                
                # Get predictions for this sample
                for scale_predictions in predictions:
                    if scale_predictions[idx] is not None and len(scale_predictions[idx]) > 0:
                        pred_classes = scale_predictions[idx][:, -1]
                        confidences = scale_predictions[idx][:, -2]
                        sample_dict['predictions'].extend(pred_classes.cpu().numpy())
                        sample_dict['confidences'].extend(confidences.cpu().numpy())
                
                # Get targets for this sample
                if targets[idx].sum() != 0:  # If target is not empty
                    sample_dict['targets'].extend(targets[idx][targets[idx][:, 0] != 0][:, 0].cpu().numpy())
                
                sample_results.append(sample_dict)
        
        # Print results for each sample
        print("\nSample Evaluation Results:")
        print("-" * 80)
        
        for idx, sample in enumerate(sample_results):
            print(f"\nSample {idx + 1}")
            print(f"Image: {os.path.basename(sample['image_path'])}")
            print("\nPredictions:")
            for pred, conf in zip(sample['predictions'], sample['confidences']):
                print(f"Class: {self.class_names[int(pred)]:15} Confidence: {conf:.3f}")
            
            print("\nGround Truth:")
            for target in sample['targets']:
                print(f"Class: {self.class_names[int(target)]}")
            print("-" * 80)
    
    def predict_samples(self, num_samples=4):
        self.model.eval()
        
        with torch.no_grad():
            # Get a single batch
            batch = next(iter(self.dataloader))
            
            # Prepare inputs
            lidar_features = batch['lidar_features'].to(self.device)
            image_features = [feat.to(self.device) for feat in batch['image_features']]
            targets = batch['targets'].to(self.device)
            
            # Get predictions
            _, _, predictions = self.model(lidar_features, image_features, targets)
            
            # Process only 4 samples
            for i in range(min(num_samples, len(batch['image_path']))):
                print(f"\nSample {i+1}: {batch['image_path'][i]}")
                
                print("Predictions:")
                for scale_preds in predictions:
                    if scale_preds[i] is not None and len(scale_preds[i]) > 0:
                        classes = scale_preds[i][:, -1]
                        scores = scale_preds[i][:, -2]
                        for cls, score in zip(classes, scores):
                            print(f"{self.class_names[int(cls)]}: {score:.3f}")
                
                print("\nGround Truth:")
                valid_targets = targets[i][targets[i][:, 0] != 0][:, 0]
                for t in valid_targets:
                    print(f"{self.class_names[int(t)]}")
                print("-" * 50)

def main():
    args = get_args()
    evaluator = Evaluator(args)
    evaluator.predict_samples()

if __name__ == '__main__':
    main()