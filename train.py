import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
import numpy as np

from models import get_model
from utils import create_data_loaders, dice_coefficient, iou_score


class SegmentationTrainer:
    """Trainer class for segmentation models."""
    
    def __init__(self, model, device='cuda', learning_rate=1e-3):
        """
        Args:
            model: PyTorch model
            device: Device to train on ('cuda' or 'cpu')
            learning_rate: Learning rate for optimizer
        """
        self.model = model
        self.device = device
        self.model = self.model.to(device)
        
        # Loss function: combine BCE with soft Dice loss to reduce
        # all-background collapse on imbalanced segmentation masks.
        self.bce_criterion = nn.BCELoss()
        self.lambda_bce = 0.5
        self.lambda_dice = 0.5
        
        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=10
        )
        
        # Training history
        self.train_history = {
            'loss': [],
            'dice': [],
            'iou': []
        }
        self.val_history = {
            'loss': [],
            'dice': [],
            'iou': []
        }

    def compute_loss(self, outputs, masks, smooth=1e-6):
        """Compute combined BCE + soft Dice loss."""
        bce_loss = self.bce_criterion(outputs, masks)

        outputs_flat = outputs.view(outputs.size(0), -1)
        masks_flat = masks.view(masks.size(0), -1)
    
        # Compute dice with smoothing to avoid division by zero
        intersection = (outputs_flat * masks_flat).sum(dim=1)
        dice = (2 * intersection + smooth) / (
            outputs_flat.sum(dim=1) + masks_flat.sum(dim=1) + smooth
        )
        dice_loss = 1 - dice.mean()

        return self.lambda_bce * bce_loss + self.lambda_dice * dice_loss
    
    def train_epoch(self, train_loader, epoch_idx=0):
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0
        total_dice = 0
        total_iou = 0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc="Training")
        for images, masks in pbar:
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.compute_loss(outputs, masks)
            
            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()
            
            # Calculate metrics
            pred_binary = (outputs > 0.5).float()
            dice = dice_coefficient(pred_binary, masks).item()
            iou = iou_score(pred_binary, masks).item()
            
            # Update totals
            total_loss += loss.item()
            total_dice += dice
            total_iou += iou
            num_batches += 1
            
            pbar.set_postfix({
                'loss': f'{total_loss / num_batches:.4f}',
                'dice': f'{total_dice / num_batches:.4f}',
                'iou': f'{total_iou / num_batches:.4f}'
            })
        
        # Compute average metrics for the epoch
        avg_loss = total_loss / num_batches
        avg_dice = total_dice / num_batches
        avg_iou = total_iou / num_batches
        
        self.train_history['loss'].append(avg_loss)
        self.train_history['dice'].append(avg_dice)
        self.train_history['iou'].append(avg_iou)
        
        return avg_loss, avg_dice, avg_iou
    
    def validate(self, val_loader):
        """Validate on validation set."""
        self.model.eval()
        
        total_loss = 0
        total_dice = 0
        total_iou = 0
        num_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validating")
            for images, masks in pbar:
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                outputs = self.model(images)
                loss = self.compute_loss(outputs, masks)
                
                # Calculate metrics
                pred_binary = (outputs > 0.5).float()
                dice = dice_coefficient(pred_binary, masks).item()
                iou = iou_score(pred_binary, masks).item()
                
                # Update totals
                total_loss += loss.item()
                total_dice += dice
                total_iou += iou
                num_batches += 1
                
                pbar.set_postfix({
                    'loss': f'{total_loss / num_batches:.4f}',
                    'dice': f'{total_dice / num_batches:.4f}',
                    'iou': f'{total_iou / num_batches:.4f}'
                })
        
        # Compute average metrics for the epoch
        avg_loss = total_loss / num_batches
        avg_dice = total_dice / num_batches
        avg_iou = total_iou / num_batches
        
        self.val_history['loss'].append(avg_loss)
        self.val_history['dice'].append(avg_dice)
        self.val_history['iou'].append(avg_iou)
        
        return avg_loss, avg_dice, avg_iou
    
    def train(self, train_loader, val_loader, num_epochs=100, checkpoint_dir='checkpoints'):
        """
        Full training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs to train
            checkpoint_dir: Directory to save checkpoints
        """
        # Create checkpoint directory
        Path(checkpoint_dir).mkdir(exist_ok=True)
        
        best_dice = -1.0
        best_epoch = -1
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Train
            train_loss, train_dice, train_iou = self.train_epoch(train_loader, epoch_idx=epoch)
            print(f"Train - Loss: {train_loss:.4f}, Dice: {train_dice:.4f}, IoU: {train_iou:.4f}")
            
            # Validate
            val_loss, val_dice, val_iou = self.validate(val_loader)
            print(f"Val   - Loss: {val_loss:.4f}, Dice: {val_dice:.4f}, IoU: {val_iou:.4f}")
            
            # Learning rate scheduling
            self.scheduler.step(val_dice)
            
            # Save best checkpoint
            if val_dice > best_dice:
                best_dice = val_dice
                best_epoch = epoch
                
                checkpoint_path = Path(checkpoint_dir) / 'best_model.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_dice': val_dice,
                    'val_iou': val_iou,
                }, checkpoint_path)
                print(f"Saved best model to {checkpoint_path}")
            
        
        print(f"\nTraining complete! Best Dice: {best_dice:.4f} at epoch {best_epoch + 1}")
        return self.train_history, self.val_history


def train_segmentation_model(model_type='unet', image_dir='./data/images', 
                             mask_dir='./data/masks', batch_size=10, 
                             num_epochs=100, learning_rate=1e-3, 
                             img_size=256, checkpoint_dir='./checkpoints'):
    """
    Train a segmentation model from scratch.
    
    Args:
        model_type: 'unet' or 'resnet_unet'
        image_dir: Path to images directory
        mask_dir: Path to masks directory
        batch_size: Training batch size
        num_epochs: Number of epochs
        learning_rate: Learning rate
        img_size: Image size for resizing
        checkpoint_dir: Directory to save checkpoints
        
    Returns:
        trainer, train_history, val_history
    """
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create data loaders
    print(f"Loading data from {image_dir} and {mask_dir}...")
    train_loader, val_loader, total_samples = create_data_loaders(
        image_dir, mask_dir, batch_size=batch_size, img_size=img_size, 
        split_dir=checkpoint_dir
    )
    print(f"Dataset size: {total_samples}")
    
    # Create model
    print(f"Creating {model_type} model...")
    model = get_model(model_type, in_channels=3, out_channels=1, device=device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    
    # Create trainer
    trainer = SegmentationTrainer(model, device=device, learning_rate=learning_rate)
    
    # Train
    train_history, val_history = trainer.train(
        train_loader, val_loader, num_epochs=num_epochs, checkpoint_dir=checkpoint_dir
    )
    
    return trainer, train_history, val_history
