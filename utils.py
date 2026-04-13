import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import json
from pathlib import Path


def dice_coefficient(pred, target, smooth=1e-6):
    """
    Calculate Dice coefficient between prediction and target.
    
    Args:
        pred: Predicted binary mask (B, H, W) or (B, 1, H, W)
        target: Target binary mask (B, H, W) or (B, 1, H, W)
        smooth: Smoothing constant to avoid division by zero
        
    Returns:
        Dice coefficient score
    """
    if len(pred.shape) == 4:
        pred = pred.squeeze(1)
    if len(target.shape) == 4:
        target = target.squeeze(1)
    
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    
    # Compute dice coefficient: 2 * intersection / (sum of sizes)
    # Add smoothing term in numerator and denominator to avoid division by zero
    intersection = (pred_flat * target_flat).sum()
    dice = (2 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    
    return dice


def iou_score(pred, target, smooth=1e-6):
    """
    Calculate Intersection over Union (IoU) score.
    
    Args:
        pred: Predicted binary mask (B, H, W) or (B, 1, H, W)
        target: Target binary mask (B, H, W) or (B, 1, H, W)
        smooth: Smoothing constant to avoid division by zero
        
    Returns:
        IoU score
    """
    if len(pred.shape) == 4:
        pred = pred.squeeze(1)
    if len(target.shape) == 4:
        target = target.squeeze(1)
    
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    
    # Compute IoU: intersection / union
    # Add smoothing term in numerator and denominator to avoid division by zero
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    
    return iou


def create_data_loaders(image_dir, mask_dir, batch_size=10, img_size=256, 
                       train_split=0.8, num_workers=0, seed=42, save_split=True, 
                       split_dir='./checkpoints'):
    """
    Create data loaders for training and validation.
    
    Args:
        image_dir: Path to images directory
        mask_dir: Path to masks directory
        batch_size: Batch size for data loader
        img_size: Image size for resizing (img_size x img_size)
        train_split: Fraction of data to use for training
        num_workers: Number of workers for data loading
        seed: Random seed for reproducibility
        save_split: Whether to save train/val split info to JSON
        split_dir: Directory to save split info
        
    Returns:
        train_loader, val_loader, dataset info, (train_indices, val_indices)
    """
    dataset = PolypDataset(image_dir, mask_dir, img_size=img_size)
    
    # Split dataset
    train_size = int(len(dataset) * train_split)
    val_size = len(dataset) - train_size
    
    # Seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create train and validation splits
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Save split information into JSON
    if save_split:
        train_indices = train_dataset.indices
        val_indices = val_dataset.indices
        
        train_filenames = [dataset.image_files[i].name for i in train_indices]
        val_filenames = [dataset.image_files[i].name for i in val_indices]
        
        split_info = {
            'train_split': train_split,
            'seed': seed,
            'total_images': len(dataset),
            'train_count': len(train_filenames),
            'val_count': len(val_filenames),
            'train_images': sorted(train_filenames),
            'val_images': sorted(val_filenames)
        }
        
        Path(split_dir).mkdir(exist_ok=True)
        split_file = Path(split_dir) / 'train_val_split.json'
        with open(split_file, 'w') as f:
            json.dump(split_info, f, indent=2)
        print(f"Saved train/val split info to {split_file}")
    
    return train_loader, val_loader, len(dataset)


class PolypDataset(Dataset):
    """
    PyTorch Dataset for polyp segmentation.
    Loads images and corresponding masks, assumes same filename in both directories.
    """
    
    def __init__(self, image_dir, mask_dir, img_size=256, augment=False):
        """
        Args:
            image_dir: Path to images directory
            mask_dir: Path to masks directory
            img_size: Size to resize images to (img_size x img_size)
            augment: Whether to apply augmentation
        """
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.img_size = img_size
        self.augment = augment
        
        # Get all JPG image files
        self.image_files = sorted(self.image_dir.glob('*.jpg'))
        
        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {image_dir}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.image_files[idx]
        image = cv2.imread(str(image_path))
        
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load corresponding mask
        mask_path = self.mask_dir / image_path.name
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        if mask is None:
            raise ValueError(f"Could not load mask: {mask_path}")
        
        # Resize
        image = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        
        # Normalize image to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Convert mask to binary (0 or 1)
        mask = (mask > 127).astype(np.float32)
        
        # Convert to torch tensors
        image = torch.from_numpy(image).permute(2, 0, 1)  # (3, H, W)
        mask = torch.from_numpy(mask).unsqueeze(0)  # (1, H, W)
        
        return image, mask
