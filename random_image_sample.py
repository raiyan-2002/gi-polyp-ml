import argparse
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from evaluate import ModelEvaluator
from utils import PolypDataset

IMAGE_DIR = "./data_separated/images"
MASK_DIR = "./data_separated/masks"
UNET_MODEL_PATH = "./checkpoints/unet_best_model.pth"
ATTENTION_MODEL_PATH = "./checkpoints/attention_unet_best_model.pth"
RESNET_MODEL_PATH = "./checkpoints/resnet_unet_best_model.pth"
IMG_SIZE = 256
THRESHOLD = 0.5

def main():
    parser = argparse.ArgumentParser(
        description="Randomly sample one data image and compare all trained models"
    )
    parser.add_argument("--image-dir", default=IMAGE_DIR, help="Directory with input images")
    parser.add_argument("--mask-dir", default=MASK_DIR, help="Directory with GT masks")
    parser.add_argument("--output-path", default=None, help="Optional output image path for visualization")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    dataset = PolypDataset(args.image_dir, args.mask_dir, img_size=IMG_SIZE)
    if len(dataset) == 0:
        raise ValueError("No images available to sample")

    rng = random.Random()
    sample_idx = rng.randrange(len(dataset))

    image_path = dataset.image_files[sample_idx]
    mask_path = Path(args.mask_dir) / image_path.name
    print(f"Sampled image index: {sample_idx}")
    print(f"Sampled image file: {image_path.name}")

    image_tensor, mask_tensor = dataset[sample_idx]
    image_tensor = image_tensor.unsqueeze(0).to(device)
    mask_tensor = mask_tensor.unsqueeze(0).to(device)

    # Create Model Evaluator for each model type
    unet_evaluator = ModelEvaluator(UNET_MODEL_PATH, model_type="unet", device=device, img_size=IMG_SIZE)

    attention_evaluator = ModelEvaluator(
        ATTENTION_MODEL_PATH,
        model_type="attention_unet",
        device=device,
        img_size=IMG_SIZE,
    )

    resnet_evaluator = ModelEvaluator(
        RESNET_MODEL_PATH,
        model_type="resnet_unet",
        device=device,
        img_size=IMG_SIZE,
    )

    # Evaluate performance of each model on the sampled image
    _, unet_hard, unet_dice, unet_iou = unet_evaluator.evaluate_single_image(
        image_tensor, mask_tensor, threshold=THRESHOLD
    )
    _, attn_hard, attn_dice, attn_iou = attention_evaluator.evaluate_single_image(
        image_tensor, mask_tensor, threshold=THRESHOLD
    )
    _, res_hard, res_dice, res_iou = resnet_evaluator.evaluate_single_image(
        image_tensor, mask_tensor, threshold=THRESHOLD
    )

    # Print and plot results for the sampled image
    print("\nPer-image results")
    print(f"U-Net       Dice: {unet_dice:.4f} | IoU: {unet_iou:.4f}")
    print(f"Attention   Dice: {attn_dice:.4f} | IoU: {attn_iou:.4f}")
    print(f"ResNet-UNet Dice: {res_dice:.4f} | IoU: {res_iou:.4f}")

    gt_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if gt_mask is None:
        raise ValueError(f"Could not load ground-truth mask: {mask_path}")
    gt_mask = cv2.resize(gt_mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
    gt_mask = (gt_mask > 127).astype(np.uint8)

    original_image = cv2.imread(str(image_path))
    if original_image is None:
        raise ValueError(f"Could not load sampled image: {image_path}")
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    original_image = cv2.resize(original_image, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    axes[0, 0].imshow(original_image)
    axes[0, 0].set_title("Input Image")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(gt_mask, cmap="gray")
    axes[0, 1].set_title("Ground Truth")
    axes[0, 1].axis("off")

    axes[0, 2].axis("off")
    axes[0, 2].text(
        0.0,
        0.75,
        f"U-Net\nDice: {unet_dice:.4f}\nIoU:  {unet_iou:.4f}\n\n"
        f"Attention U-Net\nDice: {attn_dice:.4f}\nIoU:  {attn_iou:.4f}\n\n"
        f"ResNet-UNet\nDice: {res_dice:.4f}\nIoU:  {res_iou:.4f}",
        fontsize=12,
        va="top",
    )

    axes[1, 0].imshow(unet_hard, cmap="gray")
    axes[1, 0].set_title("U-Net Binary Prediction")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(res_hard, cmap="gray")
    axes[1, 1].set_title("ResNet-UNet Binary Prediction")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(attn_hard, cmap="gray")
    axes[1, 2].set_title("Attention U-Net Binary Prediction")
    axes[1, 2].axis("off")

    fig.suptitle(f"Random Sample: {image_path.name}", fontsize=14)
    plt.tight_layout()

    if args.output_path:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved comparison figure to {output_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
