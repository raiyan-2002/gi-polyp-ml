"""
Jupyter notebook-style script for polyp segmentation training and evaluation.
Run this to train models and evaluate results.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from train import train_segmentation_model
from evaluate import ModelEvaluator
from inference import SegmentationInference


def plot_training_history(train_hist, val_hist, save_path=None):
    """Plot training history."""
    epochs = range(1, len(train_hist['loss']) + 1)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Loss
    axes[0].plot(epochs, train_hist['loss'], 'b-', label='Train Loss', marker='o')
    axes[0].plot(epochs, val_hist['loss'], 'r-', label='Val Loss', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Dice
    axes[1].plot(epochs, train_hist['dice'], 'b-', label='Train Dice', marker='o')
    axes[1].plot(epochs, val_hist['dice'], 'r-', label='Val Dice', marker='s')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Dice Coefficient')
    axes[1].set_title('Training and Validation Dice')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # IoU
    axes[2].plot(epochs, train_hist['iou'], 'b-', label='Train IoU', marker='o')
    axes[2].plot(epochs, val_hist['iou'], 'r-', label='Val IoU', marker='s')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('IoU Score')
    axes[2].set_title('Training and Validation IoU')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training plot to {save_path}")
    else:
        plt.show()


def main():
    print("="*60)
    print("Polyp Segmentation - Medical Image Segmentation Project")
    print("="*60)
    
    # Configuration
    config = {
        'image_dir': './data/images',
        'mask_dir': './data/masks',
        'batch_size': 10,
        'img_size': 256,
        'num_epochs': 100,
        'learning_rate': 1e-3,
        'checkpoint_dir': './checkpoints',
        'results_dir': './results',
    }
    
    # Create directories
    Path(config['checkpoint_dir']).mkdir(exist_ok=True)
    Path(config['results_dir']).mkdir(exist_ok=True)
    
    print(f"\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    # ===== Train U-Net =====
    print(f"\n{'='*60}")
    print("Training U-Net Model")
    print(f"{'='*60}")
    
    trainer_unet, train_hist_unet, val_hist_unet = train_segmentation_model(
        model_type='unet',
        image_dir=config['image_dir'],
        mask_dir=config['mask_dir'],
        batch_size=config['batch_size'],
        num_epochs=config['num_epochs'],
        learning_rate=config['learning_rate'],
        img_size=config['img_size'],
        checkpoint_dir=config['checkpoint_dir']
    )
    
    # Plot U-Net results
    plot_path = Path(config['results_dir']) / 'unet_training_history.png'
    plot_training_history(train_hist_unet, val_hist_unet, save_path=str(plot_path))
    
    best_unet_path = Path(config['checkpoint_dir']) / 'best_model.pth'
    
    # ===== Train ResNet-UNet =====
    print(f"\n{'='*60}")
    print("Training ResNet-UNet Model")
    print(f"{'='*60}")
    
    # Rename checkpoint for U-Net
    if best_unet_path.exists():
        backup_path = Path(config['checkpoint_dir']) / 'unet_best_model.pth'
        best_unet_path.rename(backup_path)
    
    trainer_resnet, train_hist_resnet, val_hist_resnet = train_segmentation_model(
        model_type='resnet_unet',
        image_dir=config['image_dir'],
        mask_dir=config['mask_dir'],
        batch_size=config['batch_size'],
        num_epochs=config['num_epochs'],
        learning_rate=config['learning_rate'],
        img_size=config['img_size'],
        checkpoint_dir=config['checkpoint_dir']
    )
    
    # Plot ResNet-UNet results
    plot_path = Path(config['results_dir']) / 'resnet_unet_training_history.png'
    plot_training_history(train_hist_resnet, val_hist_resnet, save_path=str(plot_path))
    
    best_resnet_path = Path(config['checkpoint_dir']) / 'best_model.pth'
    if best_resnet_path.exists():
        backup_path = Path(config['checkpoint_dir']) / 'resnet_unet_best_model.pth'
        best_resnet_path.rename(backup_path)
    
    # ===== Evaluate Models =====
    print(f"\n{'='*60}")
    print("Evaluating Models")
    print(f"{'='*60}")
    
    # Evaluate U-Net
    print("\nEvaluating U-Net...")
    evaluator_unet = ModelEvaluator(
        str(Path(config['checkpoint_dir']) / 'unet_best_model.pth'),
        model_type='unet',
        device=device,
        img_size=config['img_size']
    )
    results_unet = evaluator_unet.evaluate_dataset(
        config['image_dir'],
        config['mask_dir'],
        threshold=0.5
    )
    evaluator_unet.print_results(results_unet)
    
    # Evaluate ResNet-UNet
    print("\nEvaluating ResNet-UNet...")
    evaluator_resnet = ModelEvaluator(
        str(Path(config['checkpoint_dir']) / 'resnet_unet_best_model.pth'),
        model_type='resnet_unet',
        device=device,
        img_size=config['img_size']
    )
    results_resnet = evaluator_resnet.evaluate_dataset(
        config['image_dir'],
        config['mask_dir'],
        threshold=0.5
    )
    evaluator_resnet.print_results(results_resnet)
    
    # ===== Model Comparison =====
    print(f"\n{'='*60}")
    print("Model Comparison")
    print(f"{'='*60}")
    print(f"\nU-Net:")
    print(f"  Dice: {results_unet['dice_mean']:.4f} ± {results_unet['dice_std']:.4f}")
    print(f"  IoU:  {results_unet['iou_mean']:.4f} ± {results_unet['iou_std']:.4f}")
    print(f"\nResNet-UNet:")
    print(f"  Dice: {results_resnet['dice_mean']:.4f} ± {results_resnet['dice_std']:.4f}")
    print(f"  IoU:  {results_resnet['iou_mean']:.4f} ± {results_resnet['iou_std']:.4f}")
    
    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    models = ['U-Net', 'ResNet-UNet']
    dice_means = [results_unet['dice_mean'], results_resnet['dice_mean']]
    dice_stds = [results_unet['dice_std'], results_resnet['dice_std']]
    iou_means = [results_unet['iou_mean'], results_resnet['iou_mean']]
    iou_stds = [results_unet['iou_std'], results_resnet['iou_std']]
    
    axes[0].bar(models, dice_means, yerr=dice_stds, capsize=5, alpha=0.7, color=['blue', 'green'])
    axes[0].set_ylabel('Dice Coefficient')
    axes[0].set_title('Model Comparison - Dice')
    axes[0].set_ylim([0, 1])
    
    axes[1].bar(models, iou_means, yerr=iou_stds, capsize=5, alpha=0.7, color=['blue', 'green'])
    axes[1].set_ylabel('IoU Score')
    axes[1].set_title('Model Comparison - IoU')
    axes[1].set_ylim([0, 1])
    
    plt.tight_layout()
    comparison_path = Path(config['results_dir']) / 'model_comparison.png'
    plt.savefig(str(comparison_path), dpi=150, bbox_inches='tight')
    print(f"\nSaved comparison plot to {comparison_path}")
    
    print(f"\n{'='*60}")
    print("Project Complete!")
    print(f"{'='*60}")
    print(f"Results saved to: {config['results_dir']}")
    print(f"Checkpoints saved to: {config['checkpoint_dir']}")


if __name__ == '__main__':
    main()
