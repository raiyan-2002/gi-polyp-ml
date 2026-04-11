"""
Evaluation script for comprehensive model assessment.
"""
import torch
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

from models import get_model
from utils import PolypDataset, dice_coefficient, iou_score


class ModelEvaluator:
    """Comprehensive model evaluation."""
    
    def __init__(self, model_path, model_type='unet', device='cuda', img_size=256):
        """
        Args:
            model_path: Path to saved model checkpoint
            model_type: 'unet' or 'resnet_unet'
            device: Device to use
            img_size: Input image size
        """
        self.device = device
        self.img_size = img_size
        
        # Create and load model
        self.model = get_model(model_type, in_channels=3, out_channels=1, device=device)
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
    
    def evaluate_dataset(self, image_dir, mask_dir, threshold=0.5, batch_size=10):
        """
        Evaluate model on entire dataset.
        
        Args:
            image_dir: Path to images directory
            mask_dir: Path to masks directory
            threshold: Threshold for binary mask
            batch_size: Batch size for evaluation
            
        Returns:
            Dictionary with evaluation metrics
        """
        dataset = PolypDataset(image_dir, mask_dir, img_size=self.img_size)
        
        all_dice = []
        all_iou = []
        
        print(f"Evaluating on {len(dataset)} samples...")
        
        with torch.no_grad():
            for idx in tqdm(range(len(dataset))):
                image, mask = dataset[idx]
                image = image.unsqueeze(0).to(self.device)
                mask = mask.unsqueeze(0).to(self.device)
                
                # Forward pass
                output = self.model(image)
                pred_binary = (output > threshold).float()
                
                # Calculate metrics
                dice = dice_coefficient(pred_binary, mask).item()
                iou = iou_score(pred_binary, mask).item()
                
                all_dice.append(dice)
                all_iou.append(iou)
        
        # Compute statistics
        all_dice = np.array(all_dice)
        all_iou = np.array(all_iou)
        
        results = {
            'dice_mean': np.mean(all_dice),
            'dice_std': np.std(all_dice),
            'dice_min': np.min(all_dice),
            'dice_max': np.max(all_dice),
            'iou_mean': np.mean(all_iou),
            'iou_std': np.std(all_iou),
            'iou_min': np.min(all_iou),
            'iou_max': np.max(all_iou),
            'all_dice': all_dice,
            'all_iou': all_iou,
        }
        
        return results
    
    def print_results(self, results):
        """Print evaluation results."""
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        print(f"Dice Coefficient:")
        print(f"  Mean: {results['dice_mean']:.4f}")
        print(f"  Std:  {results['dice_std']:.4f}")
        print(f"  Min:  {results['dice_min']:.4f}")
        print(f"  Max:  {results['dice_max']:.4f}")
        print(f"\nIntersection over Union (IoU):")
        print(f"  Mean: {results['iou_mean']:.4f}")
        print(f"  Std:  {results['iou_std']:.4f}")
        print(f"  Min:  {results['iou_min']:.4f}")
        print(f"  Max:  {results['iou_max']:.4f}")
        print("="*50 + "\n")
    
    def plot_results(self, results, output_path=None):
        """Plot evaluation results."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Dice distribution
        axes[0].hist(results['all_dice'], bins=30, alpha=0.7, color='blue', edgecolor='black')
        axes[0].axvline(results['dice_mean'], color='red', linestyle='--', 
                       label=f"Mean: {results['dice_mean']:.4f}")
        axes[0].set_xlabel('Dice Coefficient')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Dice Coefficient Distribution')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # IoU distribution
        axes[1].hist(results['all_iou'], bins=30, alpha=0.7, color='green', edgecolor='black')
        axes[1].axvline(results['iou_mean'], color='red', linestyle='--', 
                       label=f"Mean: {results['iou_mean']:.4f}")
        axes[1].set_xlabel('IoU Score')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('IoU Score Distribution')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved evaluation plot to {output_path}")
        else:
            plt.show()


def evaluate_model(model_path, image_dir, mask_dir, model_type='unet', 
                   threshold=0.5, output_dir=None):
    """
    Comprehensive model evaluation.
    
    Args:
        model_path: Path to saved model checkpoint
        image_dir: Path to images directory
        mask_dir: Path to masks directory
        model_type: 'unet' or 'resnet_unet'
        threshold: Threshold for binary mask
        output_dir: Directory to save results (optional)
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create evaluator
    evaluator = ModelEvaluator(model_path, model_type=model_type, 
                              device=device)
    
    # Evaluate
    results = evaluator.evaluate_dataset(image_dir, mask_dir, threshold=threshold)
    
    # Print results
    evaluator.print_results(results)
    
    # Save results
    if output_dir:
        Path(output_dir).mkdir(exist_ok=True)
        
        # Plot
        plot_path = Path(output_dir) / 'evaluation_results.png'
        evaluator.plot_results(results, output_path=str(plot_path))
        
        # Save metrics as text
        metrics_path = Path(output_dir) / 'metrics.txt'
        with open(metrics_path, 'w') as f:
            f.write("EVALUATION RESULTS\n")
            f.write("="*50 + "\n")
            f.write(f"Dice Coefficient:\n")
            f.write(f"  Mean: {results['dice_mean']:.4f}\n")
            f.write(f"  Std:  {results['dice_std']:.4f}\n")
            f.write(f"  Min:  {results['dice_min']:.4f}\n")
            f.write(f"  Max:  {results['dice_max']:.4f}\n")
            f.write(f"\nIntersection over Union (IoU):\n")
            f.write(f"  Mean: {results['iou_mean']:.4f}\n")
            f.write(f"  Std:  {results['iou_std']:.4f}\n")
            f.write(f"  Min:  {results['iou_min']:.4f}\n")
            f.write(f"  Max:  {results['iou_max']:.4f}\n")
        
        print(f"Saved metrics to {metrics_path}")
    
    return results
