# gi-polyp-ml

download all the folders from this google drive [link](https://drive.google.com/drive/folders/1Bwbz3EF7SkfZXx2IWcrFn4K9Q35p75H4)

to create the environment, run the following command:

```
python -m venv venv

source venv/bin/activate  # on Windows, use `venv\Scripts\activate`

pip install -r requirements.txt
```


to test the models, you can run the following code:

```
python compare_models_random_image.py --output-path ./tests/testx.png
```

this will run the code on a random image from the test set and save the output to `./tests/testx.png`. You can change the output path as needed.

the images and masks are from: [here](https://datasets.simula.no/kvasir-seg/) (kvasir-seg.zip)

# Polyp Segmentation - Medical Image Segmentation with Deep Learning

A complete implementation of medical image segmentation for gastrointestinal polyps using deep learning. This project includes U-Net and ResNet-UNet models trained on the Kvasir-SEG dataset.

## Project Overview

**Problem:** Medical image segmentation is the task of assigning a class label to every pixel in an image. Accurate pixel-level boundaries for polyps help identify suspicious tissue regions more precisely than image classification, making segmentation useful for computer-aided diagnosis and clinical screening workflows.

**Dataset:** Kvasir-SEG dataset containing 1000 colonoscopy images with corresponding ground-truth segmentation masks (256×256 pixels).

**Metrics:** 
- **Dice Coefficient**: Measures overlap between predicted and ground-truth masks
- **Intersection over Union (IoU)**: Measures the intersection divided by the union of predicted and ground-truth regions

## Project Structure

```
gi-polyp-ml/
├── data/
│   ├── images/          # Input colonoscopy images
│   └── masks/           # Ground-truth segmentation masks
├── checkpoints/         # Saved model weights
├── results/             # Evaluation results and visualizations
├── models.py            # U-Net and ResNet-UNet architectures
├── utils.py             # Data loading, preprocessing, metrics
├── train.py             # Training loop and trainer class
├── evaluate.py          # Model evaluation on dataset
├── inference.py         # Prediction and visualization
├── main.py              # Command-line interface
├── experiment.py        # Full pipeline with both models
├── demo.py              # Quick demo script
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

## Installation

### 1. Clone repository and navigate to project
```bash
cd gi-polyp-ml
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Prepare dataset
- Download Kvasir-SEG dataset from [here](https://datasets.simula.no/kvasir-seg/)
- Extract to `data/` folder so that:
  ```
  data/
  ├── images/    # 1000 colonoscopy images
  └── masks/     # 1000 corresponding masks
  ```

## Usage

### Option 1: Command-line Interface

**Training U-Net model:**
```bash
python main.py --mode train --model-type unet --num-epochs 100 --batch-size 10
```

**Training ResNet-UNet model:**
```bash
python main.py --mode train --model-type resnet_unet --num-epochs 100 --batch-size 10
```

**Evaluate model:**
```bash
python main.py --mode evaluate --model-path ./checkpoints/best_model.pth \
               --model-type unet --output-dir ./results
```

**Make predictions:**
```bash
python main.py --mode predict --model-path ./checkpoints/best_model.pth \
               --image-path ./data/images/sample.jpg --model-type unet \
               --output-dir ./results
```

### Option 2: Python API

**Quick training and evaluation:**
```python
from train import train_segmentation_model
from evaluate import evaluate_model

# Train model
trainer, train_hist, val_hist = train_segmentation_model(
    model_type='unet',
    image_dir='./data/images',
    mask_dir='./data/masks',
    batch_size=10,
    num_epochs=100,
    img_size=256
)

# Evaluate
results = evaluate_model(
    model_path='./checkpoints/best_model.pth',
    image_dir='./data/images',
    mask_dir='./data/masks',
    model_type='unet',
    output_dir='./results'
)
```

**Inference on single image:**
```python
from inference import predict_and_visualize

pred_mask = predict_and_visualize(
    model_path='./checkpoints/best_model.pth',
    image_path='./data/images/sample.jpg',
    model_type='unet',
    output_path='./results/prediction.png',
    threshold=0.5
)
```

### Option 3: Full Experiment Pipeline

Run entire training and evaluation pipeline for both models:
```bash
python experiment.py
```

This script will:
1. Train U-Net model and save results
2. Train ResNet-UNet model and save results
3. Evaluate both models on the full dataset
4. Generate comparison plots
5. Save all results to `./results/`

### Option 4: Quick Demo

```bash
python demo.py
```

## Models

### 1. U-Net (Baseline)
- **Architecture**: Standard U-Net with 4 encoding levels
- **Channels**: 32 → 64 → 128 → 256 → 512 (bottleneck)
- **Decoder**: Symmetric structure with skip connections
- **Parameters**: ~7.8M
- **Advantages**: Lightweight, trains quickly, good baseline

```
Input (3, 256, 256)
  ↓
Encoder (4 levels)
  ↓
Bottleneck (512 channels)
  ↓
Decoder (4 levels) with skip connections
  ↓
Output (1, 256, 256) - Sigmoid activation
```

### 2. ResNet-UNet
- **Backbone**: ResNet50 pretrained on ImageNet
- **Architecture**: U-Net with ResNet50 encoder
- **Transfer Learning**: Benefits from ImageNet pretraining
- **Parameters**: ~25.5M
- **Advantages**: Better performance due to pretrained features, handles single-GPU constraints

## Key Features

### Data Processing
- **Image Resizing**: All images and masks resized to 256×256 pixels
- **Normalization**: Images normalized to [0, 1] range
- **Mask Binarization**: Masks converted to binary (0/1) labels
- **Train/Val Split**: 80% training, 20% validation

### Training
- **Batch Size**: 10 (optimized for single GPU)
- **Loss Function**: Binary Cross-Entropy (BCE)
- **Optimizer**: Adam with learning rate 1e-3
- **Scheduler**: ReduceLROnPlateau
- **Early Stopping**: Patience of 20 epochs

### Evaluation
- **Dice Coefficient**: $\text{Dice} = \frac{2|X \cap Y|}{|X| + |Y|}$
- **IoU Score**: $\text{IoU} = \frac{|X \cap Y|}{|X \cup Y|}$
- **Metrics computed** on full validation set with statistics (mean, std, min, max)

### Inference
- **Threshold**: 0.5 for binary mask conversion
- **Output**: Binary segmentation mask matching original image size
- **Visualization**: Side-by-side comparison of input, soft prediction, and binary mask

## Expected Performance

Based on typical Kvasir-SEG results:

| Model | Dice | IoU |
|-------|------|-----|
| U-Net | 0.75-0.85 | 0.65-0.78 |
| ResNet-UNet | 0.82-0.90 | 0.72-0.85 |

Note: Performance varies based on training duration and hyperparameters.

## Training Tips

1. **GPU Requirements**: ~4-8GB VRAM for batch size 10
2. **Training Time**: 
   - U-Net: ~2-4 hours per 100 epochs
   - ResNet-UNet: ~4-6 hours per 100 epochs
3. **Best Practices**:
   - Use ResNet-UNet if you have sufficient time/compute
   - Monitor validation Dice score for early stopping
   - Adjust learning rate if training becomes unstable
   - Increase batch size if you have more GPU memory

## Output Files

### Checkpoints
- `checkpoints/best_model.pth`: Best model weights and training metadata

### Results
- `results/evaluation_results.png`: Histogram of Dice and IoU scores
- `results/metrics.txt`: Detailed evaluation metrics
- `results/unet_training_history.png`: U-Net training curves
- `results/resnet_unet_training_history.png`: ResNet-UNet training curves
- `results/model_comparison.png`: Side-by-side model comparison
- `results/prediction.png`: Prediction visualization

## Troubleshooting

### Out of Memory Error
- Reduce batch size: `--batch-size 5`
- Reduce image size: `--img-size 128`
- Use U-Net instead of ResNet-UNet

### Slow Training
- Increase batch size if memory allows
- Use ResNet-UNet (better features, may converge faster)
- Check GPU utilization with `nvidia-smi`

### Low Performance
- Train for more epochs (increase `--num-epochs`)
- Try different learning rates (e.g., 1e-4 or 5e-3)
- Ensure data is correctly formatted (matching filenames)
- Check for image quality issues

## References

1. **U-Net**: Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation" (2015)
2. **Kvasir-SEG Dataset**: Available at https://datasets.simula.no/kvasir-seg/
3. **segmentation-models-pytorch**: https://github.com/qubvel/segmentation_models.pytorch

## License

This implementation is for educational purposes.

## Questions and Issues

For questions or issues, please refer to the code documentation and comments in each module.
