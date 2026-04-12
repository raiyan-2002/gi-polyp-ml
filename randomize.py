"""
Script to randomly select 50 unique images from data/images directory,
copy them and their masks to data_separated/, and delete from originals.
"""

import random
import shutil
from pathlib import Path


def select_and_separate_images(image_dir='./data/images', mask_dir='./data/masks',
                               output_image_dir='./data_separated/images',
                               output_mask_dir='./data_separated/masks',
                               num_images=50, delete_originals=True):
    """
    Randomly select unique images, copy them and masks to new directories,
    and optionally delete from originals.
    
    Args:
        image_dir: Path to original images directory
        mask_dir: Path to original masks directory
        output_image_dir: Path to output images directory
        output_mask_dir: Path to output masks directory
        num_images: Number of images to select (default 50)
        delete_originals: Whether to delete from original directories
        
    Returns:
        List of selected image filenames
    """
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)
    output_image_dir = Path(output_image_dir)
    output_mask_dir = Path(output_mask_dir)
    
    # Get all image files
    image_files = sorted([f.name for f in image_dir.glob('*') 
                         if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']])
    
    if len(image_files) == 0:
        print(f"No images found in {image_dir}")
        return []
    
    print(f"Total images available: {len(image_files)}")
    
    # Check if we have enough images
    if len(image_files) < num_images:
        print(f"Warning: Only {len(image_files)} images available, requested {num_images}")
        num_images = len(image_files)
    
    # Randomly select unique images
    random.seed(42)  # For reproducibility
    selected_images = random.sample(image_files, num_images)
    
    print(f"\nRandomly selected {len(selected_images)} unique images:")
    print("-" * 50)
    
    # Create output directories
    output_image_dir.mkdir(parents=True, exist_ok=True)
    output_mask_dir.mkdir(parents=True, exist_ok=True)
    
    copied_count = 0
    missing_masks = []
    
    # Copy images and masks to data_separated
    for i, img in enumerate(selected_images, 1):
        src_image = image_dir / img
        dst_image = output_image_dir / img
        
        # Find corresponding mask (same filename)
        src_mask = mask_dir / img
        dst_mask = output_mask_dir / img
        
        # Check if mask exists
        if not src_mask.exists():
            missing_masks.append(img)
            print(f"{i:2d}. {img} (WARNING: mask not found)")
            continue
        
        # Copy image
        shutil.copy2(src_image, dst_image)
        # Copy mask
        shutil.copy2(src_mask, dst_mask)
        
        print(f"{i:2d}. {img}")
        copied_count += 1
    
    print("\n" + "-" * 50)
    print(f"Copied {copied_count} images and masks to {output_image_dir.parent}")
    
    if missing_masks:
        print(f"\nWarning: {len(missing_masks)} images had missing masks:")
        for img in missing_masks:
            print(f"  - {img}")
    
    # Delete originals if requested
    if delete_originals and copied_count > 0:
        print("\n" + "-" * 50)
        print(f"Deleting {copied_count} images and masks from originals...")
        
        deleted_count = 0
        for img in selected_images:
            if img not in missing_masks:
                src_image = image_dir / img
                src_mask = mask_dir / img
                
                try:
                    src_image.unlink()
                    src_mask.unlink()
                    deleted_count += 1
                except Exception as e:
                    print(f"Error deleting {img}: {e}")
        
        print(f"Deleted {deleted_count} images and masks from originals")
    
    return selected_images


if __name__ == '__main__':
    selected = select_and_separate_images(
        image_dir='./data/images',
        mask_dir='./data/masks',
        output_image_dir='./data_separated/images',
        output_mask_dir='./data_separated/masks',
        num_images=50,
        delete_originals=True
    )
    print("\n" + "=" * 50)
    print(f"Process complete! {len(selected)} images processed")

