import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from evaluate import ModelEvaluator
from utils import PolypDataset


IMAGE_DIR = "./data_separated/images"
MASK_DIR = "./data_separated/masks"
UNET_MODEL_PATH = "./checkpoints/unet_best_model.pth"
ATTENTION_MODEL_PATH = "./checkpoints/attention_unet_best_model.pth"
RESNET_MODEL_PATH = "./checkpoints/resnet_unet_best_model.pth"
OUTPUT_CSV = "./results/model_comparison_metrics.csv"
IMG_SIZE = 256
THRESHOLD = 0.5


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate all trained segmentation models and save per-image CSV metrics"
    )
    parser.add_argument("--image-dir", default=IMAGE_DIR, help="Directory with input images")
    parser.add_argument("--mask-dir", default=MASK_DIR, help="Directory with ground-truth masks")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    dataset = PolypDataset(args.image_dir, args.mask_dir, img_size=IMG_SIZE)
    if len(dataset) == 0:
        raise ValueError(f"No images found in {args.image_dir}.")

    print(f"Found {len(dataset)} images.")

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

    rows = []
    unet_dice_scores = []
    unet_iou_scores = []
    attention_dice_scores = []
    attention_iou_scores = []
    resnet_dice_scores = []
    resnet_iou_scores = []

    # Evaluate each image in the dataset with all models and collect metrics
    for idx in tqdm(range(len(dataset)), desc="Evaluating all images"):
        image_tensor, mask_tensor = dataset[idx]
        filename = dataset.image_files[idx].name

        image_tensor = image_tensor.unsqueeze(0).to(device)
        mask_tensor = mask_tensor.unsqueeze(0).to(device)

        _, _, unet_dice, unet_iou = unet_evaluator.evaluate_single_image(
            image_tensor, mask_tensor, threshold=THRESHOLD
        )
        _, _, attention_dice, attention_iou = attention_evaluator.evaluate_single_image(
            image_tensor, mask_tensor, threshold=THRESHOLD
        )
        _, _, resnet_dice, resnet_iou = resnet_evaluator.evaluate_single_image(
            image_tensor, mask_tensor, threshold=THRESHOLD
        )

        rows.append({
            "filename": filename,
            "unet_dice": unet_dice,
            "unet_iou": unet_iou,
            "attention_unet_dice": attention_dice,
            "attention_unet_iou": attention_iou,
            "resnet_unet_dice": resnet_dice,
            "resnet_unet_iou": resnet_iou,
        })

        unet_dice_scores.append(unet_dice)
        unet_iou_scores.append(unet_iou)
        attention_dice_scores.append(attention_dice)
        attention_iou_scores.append(attention_iou)
        resnet_dice_scores.append(resnet_dice)
        resnet_iou_scores.append(resnet_iou)

    avg_row = {
        "filename": "AVERAGE",
        "unet_dice": float(np.mean(unet_dice_scores)),
        "unet_iou": float(np.mean(unet_iou_scores)),
        "attention_unet_dice": float(np.mean(attention_dice_scores)),
        "attention_unet_iou": float(np.mean(attention_iou_scores)),
        "resnet_unet_dice": float(np.mean(resnet_dice_scores)),
        "resnet_unet_iou": float(np.mean(resnet_iou_scores)),
    }
    rows.append(avg_row)

    output_path = Path(OUTPUT_CSV)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write results to CSV
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "filename",
                "unet_dice",
                "unet_iou",
                "attention_unet_dice",
                "attention_unet_iou",
                "resnet_unet_dice",
                "resnet_unet_iou",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\nDone.")
    print(f"Saved CSV to: {output_path}")
    print("\nAverage metrics:")
    print(f"U-Net Dice:        {avg_row['unet_dice']:.4f}")
    print(f"U-Net IoU:         {avg_row['unet_iou']:.4f}")
    print(f"Attention Dice:    {avg_row['attention_unet_dice']:.4f}")
    print(f"Attention IoU:     {avg_row['attention_unet_iou']:.4f}")
    print(f"ResNet-UNet Dice:  {avg_row['resnet_unet_dice']:.4f}")
    print(f"ResNet-UNet IoU:   {avg_row['resnet_unet_iou']:.4f}")


if __name__ == "__main__":
    main()