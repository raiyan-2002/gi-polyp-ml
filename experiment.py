"""
Jupyter notebook-style script for polyp segmentation training and evaluation.
Run this to train models and evaluate results.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import csv
from pathlib import Path

from train import train_segmentation_model
from evaluate import ModelEvaluator

def save_training_metrics_to_csv(train_hist, val_hist, model_name, csv_path):
    """Save training metrics to CSV file."""
    num_epochs = len(train_hist['loss'])
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(['Epoch', 'Train Loss', 'Train Dice', 'Train IoU', 
                       'Val Loss', 'Val Dice', 'Val IoU'])
        # Write data
        for epoch in range(num_epochs):
            writer.writerow([
                epoch + 1,
                f"{train_hist['loss'][epoch]:.6f}",
                f"{train_hist['dice'][epoch]:.6f}",
                f"{train_hist['iou'][epoch]:.6f}",
                f"{val_hist['loss'][epoch]:.6f}",
                f"{val_hist['dice'][epoch]:.6f}",
                f"{val_hist['iou'][epoch]:.6f}",
            ])
    
    print(f"Saved {model_name} training metrics to {csv_path}")


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
        'models_to_train': [
            {
                'type': 'unet',
                'name': 'U-Net',
                'ckpt': 'unet_best_model.pth',
                'metrics_csv': 'unet_training_metrics.csv',
                'plot_png': 'unet_training_history.png',
            },
            {
                'type': 'attention_unet',
                'name': 'Attention U-Net',
                'ckpt': 'attention_unet_best_model.pth',
                'metrics_csv': 'attention_unet_training_metrics.csv',
                'plot_png': 'attention_unet_training_history.png',
            },
            {
                'type': 'resnet_unet',
                'name': 'ResNet-UNet',
                'ckpt': 'resnet_unet_best_model.pth',
                'metrics_csv': 'resnet_unet_training_metrics.csv',
                'plot_png': 'resnet_unet_training_history.png',
            },
        ],
    }
    
    # Create directories
    Path(config['checkpoint_dir']).mkdir(exist_ok=True)
    Path(config['results_dir']).mkdir(exist_ok=True)
    
    print(f"\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    # ===== Train All Models =====
    for model_cfg in config['models_to_train']:
        print(f"\n{'='*60}")
        print(f"Training {model_cfg['name']} Model")
        print(f"{'='*60}")

        _, train_hist, val_hist = train_segmentation_model(
            model_type=model_cfg['type'],
            image_dir=config['image_dir'],
            mask_dir=config['mask_dir'],
            batch_size=config['batch_size'],
            num_epochs=config['num_epochs'],
            learning_rate=config['learning_rate'],
            img_size=config['img_size'],
            checkpoint_dir=config['checkpoint_dir']
        )

        # Save training curve for the model.
        plot_path = Path(config['results_dir']) / model_cfg['plot_png']
        plot_training_history(train_hist, val_hist, save_path=str(plot_path))

        # Save epoch metrics for the model.
        csv_path = Path(config['results_dir']) / model_cfg['metrics_csv']
        save_training_metrics_to_csv(train_hist, val_hist, model_cfg['name'], str(csv_path))

        best_model_path = Path(config['checkpoint_dir']) / 'best_model.pth'
        if best_model_path.exists():
            target_path = Path(config['checkpoint_dir']) / model_cfg['ckpt']
            best_model_path.rename(target_path)
    
    # ===== Evaluate Models =====
    print(f"\n{'='*60}")
    print("Evaluating Models")
    print(f"{'='*60}")
    
    model_results = {}
    for model_cfg in config['models_to_train']:
        print(f"\nEvaluating {model_cfg['name']}...")
        evaluator = ModelEvaluator(
            str(Path(config['checkpoint_dir']) / model_cfg['ckpt']),
            model_type=model_cfg['type'],
            device=device,
            img_size=config['img_size']
        )
        results = evaluator.evaluate_dataset(
            config['image_dir'],
            config['mask_dir'],
            threshold=0.5
        )
        evaluator.print_results(results)
        model_results[model_cfg['name']] = results
    
    # ===== Model Comparison =====
    print(f"\n{'='*60}")
    print("Model Comparison")
    print(f"{'='*60}")
    for model_cfg in config['models_to_train']:
        name = model_cfg['name']
        print(f"\n{name}:")
        print(f"  Dice: {model_results[name]['dice_mean']:.4f} ± {model_results[name]['dice_std']:.4f}")
        print(f"  IoU:  {model_results[name]['iou_mean']:.4f} ± {model_results[name]['iou_std']:.4f}")
    
    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    models = [m['name'] for m in config['models_to_train']]
    dice_means = [model_results[name]['dice_mean'] for name in models]
    dice_stds = [model_results[name]['dice_std'] for name in models]
    iou_means = [model_results[name]['iou_mean'] for name in models]
    iou_stds = [model_results[name]['iou_std'] for name in models]

    colors = ['tab:blue', 'tab:orange', 'tab:green']
    axes[0].bar(models, dice_means, yerr=dice_stds, capsize=5, alpha=0.7, color=colors[:len(models)])
    axes[0].set_ylabel('Dice Coefficient')
    axes[0].set_title('Model Comparison - Dice')
    axes[0].set_ylim([0, 1])
    
    axes[1].bar(models, iou_means, yerr=iou_stds, capsize=5, alpha=0.7, color=colors[:len(models)])
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
