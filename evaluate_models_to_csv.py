"""
Evaluate all trained segmentation models on every image in data_separated
and save one CSV with per-image metrics plus an average row.

Usage:
    python evaluate_models_to_csv.py
"""

import csv
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from models import get_model
from utils import PolypDataset, dice_coefficient, iou_score


IMAGE_DIR = "./data_separated/images"
MASK_DIR = "./data_separated/masks"
UNET_MODEL_PATH = "./checkpoints/unet_best_model.pth"
ATTENTION_MODEL_PATH = "./checkpoints/attention_unet_best_model.pth"
RESNET_MODEL_PATH = "./checkpoints/resnet_unet_best_model.pth"
OUTPUT_CSV = "./results/model_comparison_metrics.csv"
IMG_SIZE = 256
THRESHOLD = 0.5


def load_checkpoint_model(model_type: str, model_path: str, device: str):
    model = get_model(
        model_type=model_type,
        in_channels=3,
        out_channels=1,
        device=device,
    )
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def evaluate_one_image(model, image_tensor, mask_tensor, threshold: float):
    with torch.no_grad():
        output = model(image_tensor)

    pred_binary = (output > threshold).float()
    dice = float(dice_coefficient(pred_binary, mask_tensor).item())
    iou = float(iou_score(pred_binary, mask_tensor).item())
    return dice, iou


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    dataset = PolypDataset(IMAGE_DIR, MASK_DIR, img_size=IMG_SIZE)
    if len(dataset) == 0:
        raise ValueError("No images found in data_separated.")

    print(f"Found {len(dataset)} images.")

    unet_model = load_checkpoint_model("unet", UNET_MODEL_PATH, device)
    attention_model = load_checkpoint_model("attention_unet", ATTENTION_MODEL_PATH, device)
    resnet_model = load_checkpoint_model("resnet_unet", RESNET_MODEL_PATH, device)

    rows = []
    unet_dice_scores = []
    unet_iou_scores = []
    attention_dice_scores = []
    attention_iou_scores = []
    resnet_dice_scores = []
    resnet_iou_scores = []

    for idx in tqdm(range(len(dataset)), desc="Evaluating all images"):
        image_tensor, mask_tensor = dataset[idx]
        filename = dataset.image_files[idx].name

        image_tensor = image_tensor.unsqueeze(0).to(device)
        mask_tensor = mask_tensor.unsqueeze(0).to(device)

        unet_dice, unet_iou = evaluate_one_image(
            unet_model, image_tensor, mask_tensor, THRESHOLD
        )
        attention_dice, attention_iou = evaluate_one_image(
            attention_model, image_tensor, mask_tensor, THRESHOLD
        )
        resnet_dice, resnet_iou = evaluate_one_image(
            resnet_model, image_tensor, mask_tensor, THRESHOLD
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