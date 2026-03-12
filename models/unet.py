# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class SimpleUNet(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.config = config
        
#         in_channels = config['in_channels']
#         num_classes = config['num_classes']
#         stem_dim = config['stem_dim']
        
#         # Stem (same as your MIMUNet)
#         stem_layers = [
#             nn.Conv2d(in_channels, stem_dim, 
#                      kernel_size=config['stem_kernel'],
#                      padding=config['stem_padding']),
#             nn.BatchNorm2d(stem_dim),
#             nn.ReLU(inplace=True),
#         ]
        
#         if config.get('stem_downsampling', False):
#             stem_layers.append(nn.AvgPool2d(kernel_size=2, stride=2))
            
#         self.stem = nn.Sequential(*stem_layers)
        
#         # Encoder (Contracting Path) - using dims from config
#         dims = config['dims']  # [32, 64, 128, 256]
#         depths = config['depths']  # [1, 1, 2, 1]
        
#         # Encoder blocks
#         self.encoder_blocks = nn.ModuleList()
#         self.pool_layers = nn.ModuleList()
        
#         for i in range(len(dims)):
#             # Create multiple blocks for each depth
#             blocks = []
#             for _ in range(depths[i]):
#                 blocks.append(self._conv_block(dims[i-1] if i > 0 else stem_dim, dims[i]))
#             self.encoder_blocks.append(nn.Sequential(*blocks))
            
#             if i < len(dims) - 1:  # No pooling after last encoder block
#                 self.pool_layers.append(nn.MaxPool2d(2))
        
#         # Bridge (bottleneck)
#         self.bridge = self._conv_block(dims[-1], dims[-1] * 2)
        
#         # Decoder (Expanding Path)
#         self.decoder_blocks = nn.ModuleList()
#         self.upconv_layers = nn.ModuleList()
        
#         for i in range(len(dims) - 1, 0, -1):  # Reverse order: 3, 2, 1
#             self.upconv_layers.append(
#                 nn.ConvTranspose2d(dims[i] * 2 if i == len(dims)-1 else dims[i], 
#                                   dims[i-1], kernel_size=2, stride=2)
#             )
#             self.decoder_blocks.append(
#                 self._conv_block(dims[i-1] * 2, dims[i-1])  # *2 for skip connection
#             )
        
#         # Final convolution
#         self.final_conv = nn.Conv2d(dims[0], num_classes, kernel_size=1)
        
#     def _conv_block(self, in_channels, out_channels):
#         return nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )
    
#     def forward(self, x):
#         # Stem
#         x = self.stem(x)
#         skip_connections = []
        
#         # Encoder
#         for i, (encoder_block, pool_layer) in enumerate(zip(self.encoder_blocks, self.pool_layers + [None])):
#             x = encoder_block(x)
#             skip_connections.append(x)
#             if pool_layer is not None:
#                 x = pool_layer(x)
        
#         # Bridge
#         x = self.bridge(x)
        
#         # Decoder with skip connections
#         for i, (upconv, decoder_block) in enumerate(zip(self.upconv_layers, self.decoder_blocks)):
#             x = upconv(x)
#             # Get corresponding skip connection (reverse order)
#             skip = skip_connections[-(i + 2)]  # -2, -3, -4
#             # Crop skip connection if needed (due to pooling/upsampling)
#             if x.shape != skip.shape:
#                 # Center crop skip connection to match x
#                 diff_h = (skip.shape[2] - x.shape[2]) // 2
#                 diff_w = (skip.shape[3] - x.shape[3]) // 2
#                 skip = skip[:, :, 
#                            diff_h:diff_h + x.shape[2], 
#                            diff_w:diff_w + x.shape[3]]
            
#             x = torch.cat([x, skip], dim=1)
#             x = decoder_block(x)
        
#         # Final output
#         return self.final_conv(x)

# # Alternative even simpler version
# class BasicUNet(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.config = config
        
#         in_channels = config['in_channels']
#         num_classes = config['num_classes']
#         stem_dim = config['stem_dim']
        
#         # Stem
#         self.stem = nn.Sequential(
#             nn.Conv2d(in_channels, stem_dim, kernel_size=3, padding=1),
#             nn.BatchNorm2d(stem_dim),
#             nn.ReLU(inplace=True)
#         )
        
#         # Encoder
#         self.enc1 = self._block(stem_dim, 64)
#         self.enc2 = self._block(64, 128)
#         self.enc3 = self._block(128, 256)
#         self.enc4 = self._block(256, 512)
        
#         self.pool = nn.MaxPool2d(2)
        
#         # Bridge
#         self.bridge = self._block(512, 1024)
        
#         # Decoder
#         self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
#         self.dec4 = self._block(1024, 512)  # 1024 = 512 (skip) + 512 (upconv)
        
#         self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
#         self.dec3 = self._block(512, 256)  # 512 = 256 (skip) + 256 (upconv)
        
#         self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
#         self.dec2 = self._block(256, 128)  # 256 = 128 (skip) + 128 (upconv)
        
#         self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
#         self.dec1 = self._block(128, 64)  # 128 = 64 (skip) + 64 (upconv)
        
#         # Final
#         self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        
#     def _block(self, in_channels, out_channels):
#         return nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )
    
#     def forward(self, x):
#         # Stem
#         x = self.stem(x)
        
#         # Encoder
#         enc1 = self.enc1(x)
#         enc2 = self.enc2(self.pool(enc1))
#         enc3 = self.enc3(self.pool(enc2))
#         enc4 = self.enc4(self.pool(enc3))
        
#         # Bridge
#         bridge = self.bridge(self.pool(enc4))
        
#         # Decoder with skip connections
#         dec4 = self.upconv4(bridge)
#         dec4 = torch.cat((dec4, enc4), dim=1)
#         dec4 = self.dec4(dec4)
        
#         dec3 = self.upconv3(dec4)
#         dec3 = torch.cat((dec3, enc3), dim=1)
#         dec3 = self.dec3(dec3)
        
#         dec2 = self.upconv2(dec3)
#         dec2 = torch.cat((dec2, enc2), dim=1)
#         dec2 = self.dec2(dec2)
        
#         dec1 = self.upconv1(dec2)
#         dec1 = torch.cat((dec1, enc1), dim=1)
#         dec1 = self.dec1(dec1)
        
#         return self.final_conv(dec1)



import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicUNet(nn.Module):
    """
    UNet with ~4.04M parameters, matching ParallelGraphMHCSegNet (4,016,890).

    Channel schedule
    ----------------
    stem    : in_channels -> 32
    enc1    : 32  -> 32
    enc2    : 32  -> 64
    enc3    : 64  -> 128
    enc4    : 128 -> 170
    bridge  : 170 -> 340
    dec4    : 340 up -> 170  + skip enc4 (170) -> cat 340 -> 170
    dec3    : 170 up -> 128  + skip enc3 (128) -> cat 256 -> 128
    dec2    : 128 up -> 64   + skip enc2 (64)  -> cat 128 -> 64
    dec1    : 64  up -> 32   + skip enc1 (32)  -> cat 64  -> 32
    out     : 32  -> num_classes

    Parameter count (in_channels=6)
    --------------------------------
    Total: 4,038,293  (~4.02M, within 0.53% of target 4,016,890)
    Note: varies slightly with in_channels due to the stem conv.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        in_channels = config['in_channels']
        num_classes  = config['num_classes']
        stem_dim     = config.get('stem_dim', 32)

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stem_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(stem_dim),
            nn.ReLU(inplace=True),
        )

        # Encoder
        self.enc1 = self._block(stem_dim, 32)
        self.enc2 = self._block(32,       64)
        self.enc3 = self._block(64,       128)
        self.enc4 = self._block(128,      170)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bridge
        self.bridge = self._block(170, 340)

        # Decoder
        self.upconv4 = nn.ConvTranspose2d(340, 170, kernel_size=2, stride=2)
        self.dec4    = self._block(340, 170)   # 170 up + 170 skip

        self.upconv3 = nn.ConvTranspose2d(170, 128, kernel_size=2, stride=2)
        self.dec3    = self._block(256, 128)   # 128 up + 128 skip

        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2    = self._block(128, 64)    # 64  up + 64  skip

        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1    = self._block(64, 32)     # 32  up + 32  skip

        # Output
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    @staticmethod
    def _block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def _pad_and_cat(upsampled: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if upsampled.shape[2:] != skip.shape[2:]:
            dh = (skip.shape[2] - upsampled.shape[2]) // 2
            dw = (skip.shape[3] - upsampled.shape[3]) // 2
            skip = skip[:, :,
                        dh: dh + upsampled.shape[2],
                        dw: dw + upsampled.shape[3]]
        return torch.cat([upsampled, skip], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stem
        x = self.stem(x)                           # (B,  32, H,    W   )

        # Encoder
        e1 = self.enc1(x)                          # (B,  32, H,    W   )
        e2 = self.enc2(self.pool(e1))              # (B,  64, H/2,  W/2 )
        e3 = self.enc3(self.pool(e2))              # (B, 128, H/4,  W/4 )
        e4 = self.enc4(self.pool(e3))              # (B, 170, H/8,  W/8 )

        # Bridge
        b  = self.bridge(self.pool(e4))            # (B, 340, H/16, W/16)

        # Decoder
        d4 = self._pad_and_cat(self.upconv4(b),  e4)   # (B, 340, H/8,  W/8 )
        d4 = self.dec4(d4)                              # (B, 170, H/8,  W/8 )

        d3 = self._pad_and_cat(self.upconv3(d4), e3)   # (B, 256, H/4,  W/4 )
        d3 = self.dec3(d3)                              # (B, 128, H/4,  W/4 )

        d2 = self._pad_and_cat(self.upconv2(d3), e2)   # (B, 128, H/2,  W/2 )
        d2 = self.dec2(d2)                              # (B,  64, H/2,  W/2 )

        d1 = self._pad_and_cat(self.upconv1(d2), e1)   # (B,  64, H,    W   )
        d1 = self.dec1(d1)                              # (B,  32, H,    W   )

        return self.final_conv(d1)                      # (B, num_classes, H, W)


# -----------------------------------------------------------------------------
# train.py integration  — drop this block into create_model()
# -----------------------------------------------------------------------------
#
# elif model_name == 'BasicUNet':
#     in_channels  = input_channels
#     num_classes  = num_classes
#     stem_dim     = 32
#     enc_dims     = [32, 64, 128, 170]
#     bridge_dim   = 340
#     dec_dims     = [170, 128, 64, 32]
#
#     print(f"BasicUNet config - in_channels: {in_channels}, "
#           f"stem_dim: {stem_dim}, enc_dims: {enc_dims}, bridge_dim: {bridge_dim}")
#
#     model_hyperparameters = {
#         'BasicUNet_in_channels' : in_channels,
#         'BasicUNet_num_classes' : num_classes,
#         'BasicUNet_stem_dim'    : stem_dim,
#         'BasicUNet_enc_dims'    : enc_dims,
#         'BasicUNet_bridge_dim'  : bridge_dim,
#         'BasicUNet_dec_dims'    : dec_dims,
#     }
#
#     model = BasicUNet({
#         'in_channels' : in_channels,
#         'num_classes' : num_classes,
#         'stem_dim'    : stem_dim,
#     })
#     return model, model_hyperparameters


# -----------------------------------------------------------------------------
# Smoke test
# -----------------------------------------------------------------------------
def test_basic_unet():
    print("=" * 60)
    print("TEST: BasicUNet  (~4M parameters)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target = 4_016_890   # ParallelGraphMHCSegNet

    for sensor, in_ch in [("landsat8", 6), ("sentinel2", 10), ("alphaearth", 64)]:
        config = {'in_channels': in_ch, 'num_classes': 13, 'stem_dim': 32}
        model  = BasicUNet(config).to(device)
        total  = sum(p.numel() for p in model.parameters())

        x = torch.randn(4, in_ch, 224, 224, device=device)
        with torch.no_grad():
            logits = model(x)

        assert logits.shape == (4, 13, 224, 224), f"Bad shape: {logits.shape}"

        print(f"\n  Sensor : {sensor}  (in_channels={in_ch})")
        print(f"  Params : {total:,}  (diff from target: {total-target:+,}, "
              f"{abs(total-target)/target*100:.2f}%)")
        print(f"  Output : {tuple(logits.shape)}  ✓")

    print("\n" + "=" * 60)
    print("ALL CHECKS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    test_basic_unet()