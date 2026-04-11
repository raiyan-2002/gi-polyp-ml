"""
U-Net models for polyp segmentation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Double convolution block: Conv -> ReLU -> Conv -> ReLU"""
    
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    Standard U-Net architecture for image segmentation.
    Encoder: 4 levels with 32, 64, 128, 256 channels
    Decoder: 4 levels with skip connections
    """
    
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        
        # Encoder
        self.enc0 = ConvBlock(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.enc1 = ConvBlock(32, 64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.enc2 = ConvBlock(64, 128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.enc3 = ConvBlock(128, 256)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        # Bottleneck
        self.bottleneck = ConvBlock(256, 512)
        
        # Decoder
        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(512, 256)
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(256, 128)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(128, 64)
        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec0 = ConvBlock(64, 32)
        
        # Final output layer
        self.final = nn.Conv2d(32, out_channels, kernel_size=1)
    
    def forward(self, x):
        # Encoder with skip connections
        e0 = self.enc0(x)
        e1 = self.enc1(self.pool1(e0))
        e2 = self.enc2(self.pool2(e1))
        e3 = self.enc3(self.pool3(e2))
        
        # Bottleneck
        b = self.bottleneck(self.pool4(e3))
        
        # Decoder with skip connections
        d3 = self.dec3(torch.cat([self.upconv4(b), e3], 1))
        d2 = self.dec2(torch.cat([self.upconv3(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.upconv2(d2), e1], 1))
        d0 = self.dec0(torch.cat([self.upconv1(d1), e0], 1))
        
        # Output
        out = self.final(d0)
        out = torch.sigmoid(out)  # Sigmoid for binary segmentation
        
        return out


class ResNetUNet(nn.Module):
    """
    U-Net with ResNet50 encoder backbone using segmentation-models-pytorch.
    """
    
    def __init__(self, in_channels=3, out_channels=1):
        super(ResNetUNet, self).__init__()
        
        try:
            import segmentation_models_pytorch as smp
            
            self.model = smp.Unet(
                encoder_name="resnet50",
                encoder_weights="imagenet",
                in_channels=in_channels,
                classes=out_channels,
                activation="sigmoid"  # For binary segmentation
            )
        except ImportError:
            raise ImportError(
                "segmentation-models-pytorch is required for ResNetUNet. "
                "Install it with: pip install segmentation-models-pytorch"
            )
    
    def forward(self, x):
        return self.model(x)


def get_model(model_type='unet', in_channels=3, out_channels=1, device='cuda'):
    """
    Get a segmentation model.
    
    Args:
        model_type: 'unet' or 'resnet_unet'
        in_channels: Number of input channels (default 3 for RGB)
        out_channels: Number of output channels (default 1 for binary segmentation)
        device: Device to place model on
        
    Returns:
        Model on specified device
    """
    if model_type == 'unet':
        model = UNet(in_channels=in_channels, out_channels=out_channels)
    elif model_type == 'resnet_unet':
        model = ResNetUNet(in_channels=in_channels, out_channels=out_channels)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model = model.to(device)
    return model
