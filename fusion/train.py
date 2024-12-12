import torch
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import torch.optim as optim
from tqdm import tqdm
import logging
from datetime import datetime
from dataloader import KITTIDataset, collate_fn
from model import MultiModalDetector

class Trainer:
    def __init__(self, 
                 model_config,
                 dataset_config,
                 training_config):
        """
        Args:
            model_config (dict): Model configuration containing:
                - num_classes (int): Number of object classes
            dataset_config (dict): Dataset configuration containing:
                - root_dir (str): Path to KITTI dataset
                - yolo_path (str): Path to YOLO model
                - pointnet_path (str): Path to PointNet model
            training_config (dict): Training configuration containing:
                - batch_size (int): Batch size for training
                - learning_rate (float): Learning rate
                - num_epochs (int): Number of training epochs
                - val_split (float): Validation split ratio
                - device (str, optional): Device to use ('cuda' or 'cpu')
        """
        # Setup device
        self.device = (torch.device(training_config.get('device', 'cuda') 
                      if torch.cuda.is_available() else 'cpu'))
        
        # Initialize model
        self.model = MultiModalDetector(
            num_classes=model_config['num_classes']
        ).to(self.device)
        
        # Initialize dataset
        dataset = KITTIDataset(
            root_dir=dataset_config['root_dir'],
            split='training',
            yolo_path=dataset_config['yolo_path'],
            pointnet_path=dataset_config['pointnet_path']
        )
        
        # Split dataset
        val_size = int(len(dataset) * training_config['val_split'])
        train_size = len(dataset) - val_size
        train_subset, val_subset = random_split(
            dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_subset, 
            batch_size=training_config['batch_size'],
            shuffle=True,
            num_workers=2,
            collate_fn=collate_fn,
            pin_memory=False
        )
        
        self.val_loader = DataLoader(
            val_subset,
            batch_size=training_config['batch_size'],
            shuffle=False,
            num_workers=2,
            collate_fn=collate_fn,
            pin_memory=False
        )
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=training_config['learning_rate']
        )
        
        # Training parameters
        self.num_epochs = training_config['num_epochs']
        self.best_val_loss = float('inf')
        
        # Setup logging
        self.setup_logging()
        
    def setup_logging(self):
        """Initialize logging configuration"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / f'training_{timestamp}.log'),
                logging.StreamHandler()
            ]
        )
        
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(self.train_loader, desc='Training')
        for batch in progress_bar:
            # Move batch data to device
            batch = {
                'lidar_features': torch.stack([lidar_feat.squeeze(0).to(self.device) for lidar_feat in batch['lidar_features']]),  # [B, C]
                'image_features': [feat.squeeze(1).to(self.device) for feat in batch['image_features']],  # List of [B, C, H, W]
                'targets': [target.float().to(self.device) for target in batch['targets']]  # List of [num_objects, 6]
            }

            # for idx, img_feat in enumerate(batch['image_features']):
            #     batch['image_features'][idx] = img_feat.squeeze(1)
            #     print(f"img_feat shape: {batch['image_features'][idx].shape}")

            # # for lidar_feat in batch['lidar_features']:
            # #     print(f"lidar_feat shape: {lidar_feat.shape}")

            # print(f"lidar_features shape: {batch['lidar_features'].shape}")

            # exit(1)

            self.optimizer.zero_grad()
            
            # Forward pass
            output = self.model(batch)
            loss = output['loss']
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Update progress bar with loss components
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}'
            })
            
        return total_loss / len(self.train_loader)
    
    @torch.no_grad()
    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        all_detections = []
        
        progress_bar = tqdm(self.val_loader, desc='Validation')
        for batch in progress_bar:
            # Move batch data to device
            # batch = {
            #     'lidar_features': batch['lidar_features'].squeeze(0).to(self.device),  # [B, C]
            #     'image_features': [feat.squeeze(1).to(self.device) for feat in batch['image_features']],  # List of [B, C, H, W]
            #     'targets': [target.float().to(self.device) for target in batch['targets']]  # List of [num_objects, 6]
            # }
            batch = {
                'lidar_features': torch.stack([lidar_feat.squeeze(0).to(self.device) for lidar_feat in batch['lidar_features']]),  # [B, C]
                'image_features': [feat.squeeze(1).to(self.device) for feat in batch['image_features']],  # List of [B, C, H, W]
                'targets': [target.float().to(self.device) for target in batch['targets']]  # List of [num_objects, 6]
            }
            
            # Forward pass
            output = self.model(batch)
            
            if output['loss'] is not None:
                total_loss += output['loss'].item()
            
            # Collect detections
            all_detections.extend(output['detections'])
            
            if output['loss'] is not None:
                progress_bar.set_postfix({'val_loss': f'{output["loss"].item():.4f}'})
            
        return total_loss / len(self.val_loader), all_detections
    
    def train(self):
        """Main training loop"""
        for epoch in range(self.num_epochs):
            logging.info(f'\nEpoch {epoch+1}/{self.num_epochs}')
            
            # Training phase
            train_loss = self.train_epoch()
            logging.info(f'Training Loss: {train_loss:.4f}')
            
            # Validation phase
            val_loss, detections = self.validate()
            logging.info(f'Validation Loss: {val_loss:.4f}')
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch, val_loss)
                logging.info('Saved best model checkpoint')
            
            # Log detection statistics
            self.log_detection_stats(detections)
    
    def save_checkpoint(self, epoch, val_loss):
        """Save model checkpoint"""
        checkpoint_dir = Path('checkpoints')
        checkpoint_dir.mkdir(exist_ok=True)
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
        }, checkpoint_dir / 'best_model.pth')
    
    def log_detection_stats(self, detections):
        """Log detection statistics"""
        total_detections = sum(len(d) for d in detections)
        logging.info(f'Total detections: {total_detections}')
        
        if total_detections > 0:
            # Log class distribution
            class_counts = {}
            for detection_set in detections:
                # Only process detection_set if it's not empty
                if detection_set.size(0) > 0:
                    # Access class_idx (first element of the 6 values)
                    class_ids = detection_set[:, :, 0].long()  # Shape: [B, H*W]
                    print(f"class_ids shape: {class_ids.shape}")
                    print(f"class_ids: {class_ids}")
                    # Flatten and count classes
                    for class_id in class_ids.flatten():
                        class_id = int(class_id.item())
                        class_counts[class_id] = class_counts.get(class_id, 0) + 1
            
            if class_counts:  # Only log if we have any valid classes
                logging.info('Class distribution:')
                for class_id, count in class_counts.items():
                    logging.info(f'  Class {class_id}: {count} detections')

if __name__ == "__main__":
    # Configuration
    model_config = {
        'num_classes': 8  # KITTI classes
    }
    
    dataset_config = {
        'root_dir': '/home/stu12/s11/ak1825/CSCI_files',
        'yolo_path': '/home/stu12/s11/ak1825/CSCI739/yolov8n.pt',
        'pointnet_path': '/home/stu12/s11/ak1825/CSCI_files/pointnet_checkpoints/model_epoch_14.pth'
    }
    
    training_config = {
        'batch_size': 16,
        'learning_rate': 1e-4,
        'num_epochs': 50,
        'val_split': 0.2
    }
    
    # Initialize and start training
    trainer = Trainer(model_config, dataset_config, training_config)
    trainer.train()
