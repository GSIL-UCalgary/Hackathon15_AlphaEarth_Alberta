import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleUNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        in_channels = config['model']['in_channels']
        num_classes = config['model']['num_classes']
        stem_dim = config['model']['stem_dim']
        
        # Stem (same as your MIMUNet)
        stem_layers = [
            nn.Conv2d(in_channels, stem_dim, 
                     kernel_size=config['model']['stem_kernel'],
                     padding=config['model']['stem_padding']),
            nn.BatchNorm2d(stem_dim),
            nn.ReLU(inplace=True),
        ]
        
        if config['model'].get('stem_downsampling', False):
            stem_layers.append(nn.AvgPool2d(kernel_size=2, stride=2))
            
        self.stem = nn.Sequential(*stem_layers)
        
        # Encoder (Contracting Path) - using dims from config
        dims = config['model']['dims']  # [32, 64, 128, 256]
        depths = config['model']['depths']  # [1, 1, 2, 1]
        
        # Encoder blocks
        self.encoder_blocks = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        
        for i in range(len(dims)):
            # Create multiple blocks for each depth
            blocks = []
            for _ in range(depths[i]):
                blocks.append(self._conv_block(dims[i-1] if i > 0 else stem_dim, dims[i]))
            self.encoder_blocks.append(nn.Sequential(*blocks))
            
            if i < len(dims) - 1:  # No pooling after last encoder block
                self.pool_layers.append(nn.MaxPool2d(2))
        
        # Bridge (bottleneck)
        self.bridge = self._conv_block(dims[-1], dims[-1] * 2)
        
        # Decoder (Expanding Path)
        self.decoder_blocks = nn.ModuleList()
        self.upconv_layers = nn.ModuleList()
        
        for i in range(len(dims) - 1, 0, -1):  # Reverse order: 3, 2, 1
            self.upconv_layers.append(
                nn.ConvTranspose2d(dims[i] * 2 if i == len(dims)-1 else dims[i], 
                                  dims[i-1], kernel_size=2, stride=2)
            )
            self.decoder_blocks.append(
                self._conv_block(dims[i-1] * 2, dims[i-1])  # *2 for skip connection
            )
        
        # Final convolution
        self.final_conv = nn.Conv2d(dims[0], num_classes, kernel_size=1)
        
    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Stem
        x = self.stem(x)
        skip_connections = []
        
        # Encoder
        for i, (encoder_block, pool_layer) in enumerate(zip(self.encoder_blocks, self.pool_layers + [None])):
            x = encoder_block(x)
            skip_connections.append(x)
            if pool_layer is not None:
                x = pool_layer(x)
        
        # Bridge
        x = self.bridge(x)
        
        # Decoder with skip connections
        for i, (upconv, decoder_block) in enumerate(zip(self.upconv_layers, self.decoder_blocks)):
            x = upconv(x)
            # Get corresponding skip connection (reverse order)
            skip = skip_connections[-(i + 2)]  # -2, -3, -4
            # Crop skip connection if needed (due to pooling/upsampling)
            if x.shape != skip.shape:
                # Center crop skip connection to match x
                diff_h = (skip.shape[2] - x.shape[2]) // 2
                diff_w = (skip.shape[3] - x.shape[3]) // 2
                skip = skip[:, :, 
                           diff_h:diff_h + x.shape[2], 
                           diff_w:diff_w + x.shape[3]]
            
            x = torch.cat([x, skip], dim=1)
            x = decoder_block(x)
        
        # Final output
        return self.final_conv(x)

# Alternative even simpler version
class BasicUNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        in_channels = config['model']['in_channels']
        num_classes = config['model']['num_classes']
        stem_dim = config['model']['stem_dim']
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stem_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(stem_dim),
            nn.ReLU(inplace=True)
        )
        
        # Encoder
        self.enc1 = self._block(stem_dim, 64)
        self.enc2 = self._block(64, 128)
        self.enc3 = self._block(128, 256)
        self.enc4 = self._block(256, 512)
        
        self.pool = nn.MaxPool2d(2)
        
        # Bridge
        self.bridge = self._block(512, 1024)
        
        # Decoder
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = self._block(1024, 512)  # 1024 = 512 (skip) + 512 (upconv)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self._block(512, 256)  # 512 = 256 (skip) + 256 (upconv)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self._block(256, 128)  # 256 = 128 (skip) + 128 (upconv)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self._block(128, 64)  # 128 = 64 (skip) + 64 (upconv)
        
        # Final
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        
    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Stem
        x = self.stem(x)
        
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        
        # Bridge
        bridge = self.bridge(self.pool(enc4))
        
        # Decoder with skip connections
        dec4 = self.upconv4(bridge)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)
        
        return self.final_conv(dec1)