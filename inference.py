"""
Inference script for polyp segmentation.
"""
import torch
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from models import get_model
from utils import dice_coefficient, iou_score


class SegmentationInference:
    """Inference class for segmentation models."""
    
    def __init__(self, model_path, model_type='unet', device='cuda', img_size=256):
        """
        Args:
            model_path: Path to saved model checkpoint
            model_type: 'unet' or 'resnet_unet'
            device: Device to use for inference
            img_size: Input image size
        """
        self.device = device
        self.img_size = img_size
        
        # Create model
        self.model = get_model(model_type, in_channels=3, out_channels=1, device=device)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Loaded model from {model_path}")
        print(f"Validation Dice: {checkpoint['val_dice']:.4f}")
        print(f"Validation IoU: {checkpoint['val_iou']:.4f}")
    
    def preprocess_image(self, image_path):
        """
        Preprocess image for inference.
        
        Args:
            image_path: Path to image
            
        Returns:
            Preprocessed image tensor and original image
        """
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_image = image.copy()
        
        image = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        
        return image.to(self.device), original_image
    
    def predict(self, image_path, threshold=0.5):
        """
        Predict segmentation mask for an image.
        
        Args:
            image_path: Path to input image
            threshold: Threshold for binary mask
            
        Returns:
            Predicted mask, original image
        """
        image_tensor, original_image = self.preprocess_image(image_path)
        
        with torch.no_grad():
            output = self.model(image_tensor)
        
        # Convert to numpy
        pred_mask = output.squeeze().cpu().numpy()
        pred_binary = (pred_mask > threshold).astype(np.uint8) * 255
        
        # Resize back to original size
        original_h, original_w = original_image.shape[:2]
        pred_binary = cv2.resize(pred_binary, (original_w, original_h), 
                                 interpolation=cv2.INTER_NEAREST)
        
        return pred_mask, pred_binary
    
    def evaluate_image(self, image_path, mask_path, threshold=0.5):
        """
        Evaluate model on a single image with ground truth mask.
        
        Args:
            image_path: Path to image
            mask_path: Path to ground truth mask
            threshold: Threshold for binary mask
            
        Returns:
            Dice score, IoU score, predicted mask, ground truth mask
        """
        pred_mask, _ = self.predict(image_path, threshold=threshold)
        
        # Load ground truth mask
        gt_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if gt_mask is None:
            raise ValueError(f"Could not load mask: {mask_path}")
        
        # Resize ground truth to model input size
        gt_mask = cv2.resize(gt_mask, (self.img_size, self.img_size), 
                            interpolation=cv2.INTER_NEAREST)
        gt_mask = (gt_mask > 127).astype(np.float32)
        
        # Convert predictions to same format
        pred_binary = (pred_mask > threshold).astype(np.float32)
        
        # Calculate metrics
        pred_tensor = torch.from_numpy(pred_binary).unsqueeze(0).unsqueeze(0)
        gt_tensor = torch.from_numpy(gt_mask).unsqueeze(0).unsqueeze(0)
        
        dice = dice_coefficient(pred_tensor, gt_tensor).item()
        iou = iou_score(pred_tensor, gt_tensor).item()
        
        return dice, iou, pred_mask, gt_mask


def predict_and_visualize(model_path, image_path, model_type='unet', 
                          output_path=None, threshold=0.5):
    """
    Predict segmentation and save/display visualization.
    
    Args:
        model_path: Path to saved model
        image_path: Path to input image
        model_type: 'unet' or 'resnet_unet'
        output_path: Path to save output visualization (optional)
        threshold: Threshold for binary mask
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load model
    inferencer = SegmentationInference(model_path, model_type=model_type, 
                                       device=device)
    
    # Predict
    pred_mask, original_image = inferencer.preprocess_image(image_path)
    
    with torch.no_grad():
        output = inferencer.model(pred_mask)
    
    pred_mask = output.squeeze().cpu().numpy()
    pred_binary = (pred_mask > threshold).astype(np.uint8) * 255
    
    # Resize back to original
    original_h, original_w = original_image.shape[:2]
    pred_binary_resized = cv2.resize(pred_binary, (original_w, original_h), 
                                     interpolation=cv2.INTER_NEAREST)
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(pred_mask, cmap='gray')
    axes[1].set_title('Predicted Mask (Soft)')
    axes[1].axis('off')
    
    axes[2].imshow(pred_binary_resized, cmap='gray')
    axes[2].set_title(f'Predicted Mask (Binary, threshold={threshold})')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {output_path}")
    else:
        plt.show()
    
    return pred_binary_resized
