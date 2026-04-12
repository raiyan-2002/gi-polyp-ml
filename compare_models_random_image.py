"""
Randomly sample one image from the dataset and compare U-Net vs ResNet-UNet.
"""

import argparse
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from models import get_model
from utils import PolypDataset, dice_coefficient, iou_score


def load_checkpoint_model(model_type, model_path, device):
    model = get_model(model_type=model_type, in_channels=3, out_channels=1, device=device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def evaluate_single_image(model, image_tensor, mask_tensor, threshold):
    with torch.no_grad():
        output = model(image_tensor)

    pred_binary = (output > threshold).float()
    dice = dice_coefficient(pred_binary, mask_tensor).item()
    iou = iou_score(pred_binary, mask_tensor).item()

    pred_soft = output.squeeze(0).squeeze(0).cpu().numpy()
    pred_hard = pred_binary.squeeze(0).squeeze(0).cpu().numpy()
    return pred_soft, pred_hard, dice, iou


def main():
    parser = argparse.ArgumentParser(
        description="Randomly sample one data image and compare both trained models"
    )
    parser.add_argument("--image-dir", default="./data_separated/images", help="Directory with input images")
    parser.add_argument("--mask-dir", default="./data_separated/masks", help="Directory with GT masks")
    parser.add_argument("--unet-model-path", default="./checkpoints/unet_best_model.pth", help="Path to U-Net checkpoint")
    parser.add_argument(
        "--resnet-model-path",
        default="./checkpoints/resnet_unet_best_model.pth",
        help="Path to ResNet-UNet checkpoint",
    )
    parser.add_argument("--img-size", type=int, default=256, help="Input size used by the models")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for binary mask")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (optional)")
    parser.add_argument("--output-path", default=None, help="Optional output image path for visualization")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    dataset = PolypDataset(args.image_dir, args.mask_dir, img_size=args.img_size)
    if len(dataset) == 0:
        raise ValueError("No images available to sample")

    rng = random.Random(args.seed)
    sample_idx = rng.randrange(len(dataset))

    image_path = dataset.image_files[sample_idx]
    mask_path = Path(args.mask_dir) / image_path.name
    print(f"Sampled image index: {sample_idx}")
    print(f"Sampled image file: {image_path.name}")

    image_tensor, mask_tensor = dataset[sample_idx]
    image_tensor = image_tensor.unsqueeze(0).to(device)
    mask_tensor = mask_tensor.unsqueeze(0).to(device)

    unet_model = load_checkpoint_model("unet", args.unet_model_path, device)
    resnet_model = load_checkpoint_model("resnet_unet", args.resnet_model_path, device)

    unet_soft, unet_hard, unet_dice, unet_iou = evaluate_single_image(
        unet_model, image_tensor, mask_tensor, args.threshold
    )
    res_soft, res_hard, res_dice, res_iou = evaluate_single_image(
        resnet_model, image_tensor, mask_tensor, args.threshold
    )

    print("\nPer-image results")
    print(f"U-Net       Dice: {unet_dice:.4f} | IoU: {unet_iou:.4f}")
    print(f"ResNet-UNet Dice: {res_dice:.4f} | IoU: {res_iou:.4f}")

    gt_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if gt_mask is None:
        raise ValueError(f"Could not load ground-truth mask: {mask_path}")
    gt_mask = cv2.resize(gt_mask, (args.img_size, args.img_size), interpolation=cv2.INTER_NEAREST)
    gt_mask = (gt_mask > 127).astype(np.uint8)

    original_image = cv2.imread(str(image_path))
    if original_image is None:
        raise ValueError(f"Could not load sampled image: {image_path}")
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    original_image = cv2.resize(original_image, (args.img_size, args.img_size), interpolation=cv2.INTER_LINEAR)

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))

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

    diff = np.abs(res_hard.astype(np.float32) - unet_hard.astype(np.float32))
    axes[1, 2].imshow(diff, cmap="hot")
    axes[1, 2].set_title("Model Disagreement")
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
