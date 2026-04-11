"""
Main script for training and evaluating polyp segmentation models.
Includes examples for both U-Net and ResNet-UNet models.
"""
import argparse
from pathlib import Path
import torch

from train import train_segmentation_model
from evaluate import evaluate_model
from inference import predict_and_visualize


def main():
    parser = argparse.ArgumentParser(
        description='Polyp Segmentation - Medical Image Segmentation'
    )
    
    parser.add_argument('--mode', choices=['train', 'evaluate', 'predict'], 
                       default='train',
                       help='Mode: train, evaluate, or predict')
    parser.add_argument('--model-type', choices=['unet', 'resnet_unet'], 
                       default='unet',
                       help='Model architecture: unet or resnet_unet')
    parser.add_argument('--image-dir', default='./data/images',
                       help='Path to images directory')
    parser.add_argument('--mask-dir', default='./data/masks',
                       help='Path to masks directory')
    parser.add_argument('--batch-size', type=int, default=10,
                       help='Training batch size')
    parser.add_argument('--num-epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--img-size', type=int, default=256,
                       help='Image size for resizing')
    parser.add_argument('--checkpoint-dir', default='./checkpoints',
                       help='Directory for saving checkpoints')
    parser.add_argument('--model-path', 
                       help='Path to saved model for evaluation/inference')
    parser.add_argument('--image-path',
                       help='Path to single image for prediction')
    parser.add_argument('--output-dir', default='./results',
                       help='Directory for saving results')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Threshold for binary mask')
    
    args = parser.parse_args()
    
    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"PyTorch version: {torch.__version__}")
    print(f"Using device: {device}")
    
    if args.mode == 'train':
        print(f"\n{'='*60}")
        print(f"Training {args.model_type} model for polyp segmentation")
        print(f"{'='*60}\n")
        
        trainer, train_hist, val_hist = train_segmentation_model(
            model_type=args.model_type,
            image_dir=args.image_dir,
            mask_dir=args.mask_dir,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            img_size=args.img_size,
            checkpoint_dir=args.checkpoint_dir
        )
        
        print(f"\nTraining complete!")
        print(f"Best checkpoint saved to: {Path(args.checkpoint_dir) / 'best_model.pth'}")
        
    elif args.mode == 'evaluate':
        if not args.model_path:
            raise ValueError("--model-path required for evaluation mode")
        
        print(f"\n{'='*60}")
        print(f"Evaluating {args.model_type} model")
        print(f"{'='*60}\n")
        
        results = evaluate_model(
            model_path=args.model_path,
            image_dir=args.image_dir,
            mask_dir=args.mask_dir,
            model_type=args.model_type,
            threshold=args.threshold,
            output_dir=args.output_dir
        )
        
    elif args.mode == 'predict':
        if not args.model_path or not args.image_path:
            raise ValueError("--model-path and --image-path required for prediction mode")
        
        print(f"\n{'='*60}")
        print(f"Predicting with {args.model_type} model")
        print(f"{'='*60}\n")
        
        Path(args.output_dir).mkdir(exist_ok=True)
        output_path = Path(args.output_dir) / 'prediction.png'
        
        pred_mask = predict_and_visualize(
            model_path=args.model_path,
            image_path=args.image_path,
            model_type=args.model_type,
            output_path=str(output_path),
            threshold=args.threshold
        )
        
        print(f"\nPrediction complete!")
        print(f"Output saved to: {output_path}")


if __name__ == '__main__':
    main()
