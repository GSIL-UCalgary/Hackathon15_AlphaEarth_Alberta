import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x):
        residual = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += residual
        out = self.relu(out)
        
        return out

class ResNetUNet(nn.Module):
    def __init__(self, config):
        super(ResNetUNet, self).__init__()
        
        # Extract parameters from config
        in_channels = config['model']['in_channels']
        num_classes = config['model']['num_classes']
        stem_dim = config['model']['stem_dim']
        stem_kernel = config['model']['stem_kernel']
        stem_padding = config['model']['stem_padding']
        stem_downsampling = config['model']['stem_downsampling']
        depths = config['model']['depths']
        dims = config['model']['dims']
        
        # Stem layer
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stem_dim, kernel_size=stem_kernel, 
                     stride=2 if stem_downsampling else 1, 
                     padding=stem_padding, bias=False),
            nn.BatchNorm2d(stem_dim),
            nn.ReLU(inplace=True)
        )
        
        # Encoder (Downsampling path)
        self.encoder1 = self._make_layer(stem_dim, dims[0], depths[0], stride=1)
        self.encoder2 = self._make_layer(dims[0], dims[1], depths[1], stride=2)
        self.encoder3 = self._make_layer(dims[1], dims[2], depths[2], stride=2)
        self.encoder4 = self._make_layer(dims[2], dims[3], depths[3], stride=2)
        
        # Bridge
        self.bridge = nn.Sequential(
            nn.Conv2d(dims[3], dims[3] * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dims[3] * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(dims[3] * 2, dims[3], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dims[3]),
            nn.ReLU(inplace=True)
        )
        
        # Decoder (Upsampling path)
        self.upconv4 = nn.ConvTranspose2d(dims[3], dims[2], kernel_size=2, stride=2)
        self.decoder3 = self._make_layer(dims[2] * 2, dims[2], depths[2])
        
        self.upconv3 = nn.ConvTranspose2d(dims[2], dims[1], kernel_size=2, stride=2)
        self.decoder2 = self._make_layer(dims[1] * 2, dims[1], depths[1])
        
        self.upconv2 = nn.ConvTranspose2d(dims[1], dims[0], kernel_size=2, stride=2)
        self.decoder1 = self._make_layer(dims[0] * 2, dims[0], depths[0])
        
        self.upconv1 = nn.ConvTranspose2d(dims[0], stem_dim, kernel_size=2, stride=2)
        
        # Final layers
        self.final_conv = nn.Sequential(
            nn.Conv2d(stem_dim, stem_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(stem_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(stem_dim, num_classes, kernel_size=1)
        )
        
        # Store intermediate features for skip connections
        self.stem_out = None
        self.enc1_out = None
        self.enc2_out = None
        self.enc3_out = None
        
    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Store original input size for final upsampling
        original_size = x.shape[2:]
        
        # Stem
        stem_out = self.stem(x)
        
        # Encoder with skip connections
        enc1 = self.encoder1(stem_out)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)
        
        # Bridge
        bridge = self.bridge(enc4)
        
        # Decoder with skip connections
        dec3 = self.upconv4(bridge)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.decoder3(dec3)
        
        dec2 = self.upconv3(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv2(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.decoder1(dec1)
        
        dec0 = self.upconv1(dec1)
        
        # Final output
        out = self.final_conv(dec0)
        
        # Upsample to original input size if needed
        if out.shape[2:] != original_size:
            out = F.interpolate(out, size=original_size, mode='bilinear', align_corners=True)
        
        return out

# Wrapper class to match your expected interface
class ResNetUNetWrapper(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = ResNetUNet(config)
    
    def forward(self, x):
        return self.model(x)