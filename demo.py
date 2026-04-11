"""
Simple demo script showing how to use the polyp segmentation system.
"""

import torch
from pathlib import Path

# Example 1: Train U-Net model
print("="*60)
print("Example 1: Training U-Net Model")
print("="*60)

from train import train_segmentation_model

trainer, train_hist, val_hist = train_segmentation_model(
    model_type='unet',
    image_dir='./data/images',
    mask_dir='./data/masks',
    batch_size=10,
    num_epochs=50,  # For demo
    learning_rate=1e-3,
    img_size=256,
    checkpoint_dir='./checkpoints'
)

best_model_path = './checkpoints/best_model.pth'
print(f"Model saved to: {best_model_path}")

# Example 2: Evaluate the model
print("\n" + "="*60)
print("Example 2: Evaluating Model")
print("="*60)

from evaluate import evaluate_model

results = evaluate_model(
    model_path=best_model_path,
    image_dir='./data/images',
    mask_dir='./data/masks',
    model_type='unet',
    threshold=0.5,
    output_dir='./results'
)

# Example 3: Make predictions on a single image
print("\n" + "="*60)
print("Example 3: Making Predictions")
print("="*60)

from inference import predict_and_visualize

# Get first image
image_files = list(Path('./data/images').glob('*.jpg'))
if image_files:
    first_image = str(image_files[0])
    
    pred_mask = predict_and_visualize(
        model_path=best_model_path,
        image_path=first_image,
        model_type='unet',
        output_path='./results/prediction_demo.png',
        threshold=0.5
    )
    print(f"Prediction visualization saved!")

print("\n" + "="*60)
print("Demo Complete!")
print("="*60)
