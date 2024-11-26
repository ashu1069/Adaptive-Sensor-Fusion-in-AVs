import os
import torch
import torch.optim as optim
from tqdm import tqdm

from kitti_dataloader import get_dataloader, initialize_models
from fusion import FusionModule
from detection_head import DetectionHead, detection_loss

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize models
        self._init_models()
        
        # Initialize optimizer
        trainable_params = list(self.fusion_module.parameters()) + \
                          list(self.detection_head.parameters())
        self.optimizer = optim.Adam(trainable_params, lr=config['learning_rate'])
        
        # Initialize dataloaders
        self._init_dataloaders()

    def _init_models(self):
        # Initialize backbone models
        self.img_backbone, self.lidar_backbone = initialize_models(
            pointnet_weights_path=self.config.get('pointnet_weights_path')
        )
        
        # Initialize fusion module
        self.fusion_module = FusionModule(
            img_channels=2048,
            lidar_channels=512
        ).to(self.device)
        
        # Initialize detection head
        self.detection_head = DetectionHead(
            in_channels=2048,
            num_classes=9  # KITTI classes
        ).to(self.device)

    def _init_dataloaders(self):
        # Training dataloader
        self.train_loader = get_dataloader(
            root_path=self.config['data_path'],
            split='training',
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['num_workers'],
            img_backbone=self.img_backbone,
            lidar_backbone=self.lidar_backbone
        )
        
        # # Validation dataloader - commented out since we don't have validation data
        # self.val_loader = get_dataloader(
        #     root_path=self.config['data_path'],
        #     split='testing',
        #     batch_size=self.config['batch_size'],
        #     shuffle=False,
        #     num_workers=self.config['num_workers'],
        #     img_backbone=self.img_backbone,
        #     lidar_backbone=self.lidar_backbone
        # )

    def train_epoch(self, epoch):
        self.fusion_module.train()
        self.detection_head.train()
        
        total_loss = 0
        total_cls_loss = 0
        total_box_loss = 0
        
        pbar = tqdm(self.train_loader.dataloader, desc=f'Epoch {epoch}')
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            image_features = batch['image_features'].to(self.device)
            lidar_features = batch['lidar_features'].to(self.device)
            targets = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                       for k, v in t.items()} for t in batch['targets']]
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            fused_features, attention_weights = self.fusion_module(
                image_features, lidar_features
            )
            detections = self.detection_head(fused_features, attention_weights)
            
            # Compute loss with the full batch of targets
            loss_dict = detection_loss(
                detections, 
                targets,  # Now passing the full list of targets
                cls_weight=self.config['cls_weight'],
                box_weight=self.config['box_weight']
            )
            
            # Backward pass
            loss_dict['total_loss'].backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss_dict['total_loss'].item()
            total_cls_loss += loss_dict['cls_loss'].item()
            total_box_loss += loss_dict['box_loss'].item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': total_loss / (batch_idx + 1),
                'cls_loss': total_cls_loss / (batch_idx + 1),
                'box_loss': total_box_loss / (batch_idx + 1)
            })
        
        # Calculate average metrics
        metrics = {
            'train/total_loss': total_loss / len(self.train_loader.dataloader),
            'train/cls_loss': total_cls_loss / len(self.train_loader.dataloader),
            'train/box_loss': total_box_loss / len(self.train_loader.dataloader)
        }
        
        return metrics

    @torch.no_grad()
    # def validate(self, epoch):
    #     self.fusion_module.eval()
    #     self.detection_head.eval()
        
    #     total_loss = 0
    #     total_cls_loss = 0
    #     total_box_loss = 0
        
    #     for batch in tqdm(self.val_loader.dataloader, desc='Validation'):
    #         # Move data to device
    #         image_features = batch['image_features'].to(self.device)
    #         lidar_features = batch['lidar_features'].to(self.device)
    #         targets = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
    #                   for k, v in batch['targets'][0].items()}
            
    #         # Forward pass
    #         fused_features, attention_weights = self.fusion_module(
    #             image_features, lidar_features
    #         )
    #         detections = self.detection_head(fused_features, attention_weights)
            
    #         # Compute loss
    #         loss_dict = detection_loss(
    #             detections, 
    #             targets,
    #             cls_weight=self.config['cls_weight'],
    #             box_weight=self.config['box_weight']
    #         )
            
    #         # Update metrics
    #         total_loss += loss_dict['total_loss'].item()
    #         total_cls_loss += loss_dict['cls_loss'].item()
    #         total_box_loss += loss_dict['box_loss'].item()
        
    #     # Calculate average metrics
    #     metrics = {
    #         'val/total_loss': total_loss / len(self.val_loader.dataloader),
    #         'val/cls_loss': total_cls_loss / len(self.val_loader.dataloader),
    #         'val/box_loss': total_box_loss / len(self.val_loader.dataloader)
    #     }
        
        # return metrics

    def save_checkpoint(self, epoch, metrics, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'fusion_state_dict': self.fusion_module.state_dict(),
            'detection_state_dict': self.detection_head.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics
        }
        
        # Save latest checkpoint
        path = os.path.join(self.config['checkpoint_dir'], f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, path)
        
        # Save best checkpoint if needed
        if is_best:
            best_path = os.path.join(self.config['checkpoint_dir'], 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"Saved best model with validation loss: {metrics['train/total_loss']:.4f}")

    def train(self):
        best_loss = float('inf')
        
        for epoch in range(self.config['num_epochs']):
            print(f"\nEpoch {epoch+1}/{self.config['num_epochs']}")
            
            # Training
            train_metrics = self.train_epoch(epoch)
            
            # Save checkpoint
            is_best = train_metrics['train/total_loss'] < best_loss
            if is_best:
                best_loss = train_metrics['train/total_loss']
            
            self.save_checkpoint(epoch, train_metrics, is_best)
            
            # Print metrics
            print(f"\nEpoch {epoch} metrics:")
            print("Training:")
            print(f"  Total Loss: {train_metrics['train/total_loss']:.4f}")
            print(f"  Classification Loss: {train_metrics['train/cls_loss']:.4f}")
            print(f"  Box Loss: {train_metrics['train/box_loss']:.4f}")

def main():
    config = {
        'data_path': 'CSCI_files/dev_datakit',
        'pointnet_weights_path': 'CSCI_files/pointnet_checkpoint_epoch_200.pth',
        'checkpoint_dir': 'checkpoints',
        'batch_size': 4,
        'num_workers': 4,
        'learning_rate': 1e-4,
        'num_epochs': 100,
        'cls_weight': 1.0,
        'box_weight': 1.0
    }
    
    # Create checkpoint directory
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    
    # Initialize trainer
    trainer = Trainer(config)
    
    # Start training
    trainer.train()

if __name__ == '__main__':
    main()
