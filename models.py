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


class AttentionGate(nn.Module):
    """
    Attention gate for filtering encoder skip features using decoder context.
    Based on Attention U-Net gating: alpha = sigmoid(psi(ReLU(Wx*x + Wg*g)))
    """

    def __init__(self, F_g, F_l, F_int):
        """
        Args:
            F_g: channels of gating signal (decoder feature)
            F_l: channels of skip connection (encoder feature)
            F_int: intermediate channels
        """
        super(AttentionGate, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        alpha = self.psi(psi)
        return x * alpha


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


class AttentionUNet(nn.Module):
    """
    Attention U-Net:
    Same encoder/decoder widths as UNet, but skip connections are gated.
    """

    def __init__(self, in_channels=3, out_channels=1):
        super(AttentionUNet, self).__init__()

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
        self.att4 = AttentionGate(F_g=256, F_l=256, F_int=128)
        self.dec3 = ConvBlock(512, 256)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.att3 = AttentionGate(F_g=128, F_l=128, F_int=64)
        self.dec2 = ConvBlock(256, 128)

        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.att2 = AttentionGate(F_g=64, F_l=64, F_int=32)
        self.dec1 = ConvBlock(128, 64)

        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.att1 = AttentionGate(F_g=32, F_l=32, F_int=16)
        self.dec0 = ConvBlock(64, 32)

        # Final output layer
        self.final = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        e0 = self.enc0(x)
        e1 = self.enc1(self.pool1(e0))
        e2 = self.enc2(self.pool2(e1))
        e3 = self.enc3(self.pool3(e2))

        # Bottleneck
        b = self.bottleneck(self.pool4(e3))

        # Decoder + attention-gated skips
        g4 = self.upconv4(b)
        x4 = self.att4(g=g4, x=e3)
        d3 = self.dec3(torch.cat([g4, x4], dim=1))

        g3 = self.upconv3(d3)
        x3 = self.att3(g=g3, x=e2)
        d2 = self.dec2(torch.cat([g3, x3], dim=1))

        g2 = self.upconv2(d2)
        x2 = self.att2(g=g2, x=e1)
        d1 = self.dec1(torch.cat([g2, x2], dim=1))

        g1 = self.upconv1(d1)
        x1 = self.att1(g=g1, x=e0)
        d0 = self.dec0(torch.cat([g1, x1], dim=1))

        out = self.final(d0)
        out = torch.sigmoid(out)
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
        model_type: 'unet', 'attention_unet', or 'resnet_unet'
        in_channels: Number of input channels (default 3 for RGB)
        out_channels: Number of output channels (default 1 for binary segmentation)
        device: Device to place model on

    Returns:
        Model on specified device
    """
    if model_type == 'unet':
        model = UNet(in_channels=in_channels, out_channels=out_channels)
    elif model_type == 'attention_unet':
        model = AttentionUNet(in_channels=in_channels, out_channels=out_channels)
    elif model_type == 'resnet_unet':
        model = ResNetUNet(in_channels=in_channels, out_channels=out_channels)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model = model.to(device)
    return model