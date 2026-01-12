import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from .focal import FocalTransformerBlock

class FocalUNet(nn.Module):
    def __init__(
        self,config):
        super().__init__()
        
        # now pull out everything you need:
        in_ch  = config['model']['in_channels']
        out_ch = config['model']['num_classes']
        stem_dim    = config['model']['stem_dim']
        stem_k      = config['model']['stem_kernel']
        stem_p      = config['model']['stem_padding']
        do_stem_ds  = config['model'].get('stem_downsampling', False)
        depths      = config['model']['depths']
        num_heads    = config['model']['heads']
        dims= config['model']['dims']
        raw_img_size= config['model']['img_size']



        # img_size might be a single int or a 2-list
        m = config['model']['focal']  #  img_size: 128       # your patch height and width
        window_size= m['window_size']

        if isinstance(raw_img_size, int):
            H = W = raw_img_size
        else:
            H, W = raw_img_size
        focal_levels = m['focal_levels']
        
        mlp_ratio    = m['mlp_ratio']
        drop_rate    = m['drop_rate']
        norm_layer   = partial(nn.LayerNorm, eps=1e-6)

        # store for forward
        self.img_size   = (H, W)
        self.num_layers = len(depths)
        self.dims       = dims

        # figure out your base_h/base_w
        base_h = H // 4
        base_w = W // 4
        # ─── Stem ───────────────────────────────────────────────────────
        stem_layers = [
            nn.Conv2d(in_ch, stem_dim, kernel_size=stem_k, padding=stem_p, bias=False),
            nn.BatchNorm2d(stem_dim),
            nn.GELU(),
        ]
        if do_stem_ds:
            stem_layers.append(nn.AvgPool2d(2,2))  # 128→64
        self.stem = nn.Sequential(*stem_layers)

        # compute base H/W *after* the stem
        if do_stem_ds:
            # stem did a 2× AvgPool, so we’re at H/2
            base_h, base_w = H // 2, W // 2
        else:
            # stem kept full resolution
            base_h, base_w = H, W

        # ─── Encoder ───────────────────────────────────────────────────
        self.encoder = nn.ModuleList()
        for i in range(self.num_layers):
            # focal blocks
            res_h = base_h // (2**i)
            res_w = base_w // (2**i)
            blocks = nn.ModuleList([
                FocalTransformerBlock(
                    dim=dims[i],
                    input_resolution=(res_h, res_w),
                    num_heads=  num_heads , #dims[i] // 32,
                    window_size=window_size,
                    focal_level=focal_levels[i],
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    norm_layer=norm_layer,
                ) for _ in range(depths[i])
            ])
            self.encoder.append(blocks)
            # downsample conv
            if i < self.num_layers - 1:
                self.encoder.append(
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2)
                )

        # ─── Bottleneck ────────────────────────────────────────────────
        deep_h = base_h // (2**(self.num_layers - 1))
        deep_w = base_w // (2**(self.num_layers - 1))
        self.bottleneck = nn.ModuleList([
            FocalTransformerBlock(
                dim=dims[-1],
                input_resolution=(deep_h, deep_w),
                num_heads= num_heads, #dims[-1] // 32,
                window_size=window_size,
                focal_level=focal_levels[-1],
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                norm_layer=norm_layer,
            ) for _ in range(2)
        ])

        # ─── Decoder ───────────────────────────────────────────────────
        self.decoder = nn.ModuleList()
        for i in reversed(range(self.num_layers - 1)):
            # 1) upsample
            up = nn.ConvTranspose2d(dims[i+1], dims[i], kernel_size=2, stride=2)
            self.decoder.append(up)
            # 2) focal blocks on concat(skip, x)
            res_h = base_h // (2**i)
            res_w = base_w // (2**i)
            blocks = nn.ModuleList([
                FocalTransformerBlock(
                    dim=dims[i]*2,
                    input_resolution=(res_h, res_w),
                    num_heads= num_heads, #dims[i] // 32,
                    window_size=window_size,
                    focal_level=focal_levels[i],
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    norm_layer=norm_layer,
                ) for _ in range(depths[i])
            ])
            self.decoder.append(blocks)
            # 3) project back to dims[i]
            proj = nn.Conv2d(dims[i]*2, dims[i], kernel_size=1, bias=False)
            self.decoder.append(proj)

        # ─── Final conv ────────────────────────────────────────────────
        # always have a 1×1 projection to get the right number of channels
        self.final_conv = nn.Conv2d(dims[0], out_ch, kernel_size=1, bias=False)
        # then conditionally define the upsample
        if do_stem_ds:
            # stem pooled 2×, so decoder output is half spatial → need 2× learnable upsample
            self.final_upsample = nn.ConvTranspose2d(
                out_ch,  # projecting channels is already done by final_conv
                out_ch,
                kernel_size=2,
                stride=2,
                bias=False
            )
        else:
            # no spatial change needed
            self.final_upsample = nn.Identity()

    def forward(self, x):
        x = self.stem(x)          # (B, dims[0], H, W)
        skips = []

        # Encoder
        for layer in self.encoder:
            if isinstance(layer, nn.ModuleList):
                B, C, H, W = x.shape
                seq = x.permute(0,2,3,1).reshape(B, H*W, C)
                for blk in layer:
                    seq = blk(seq)
                x = seq.reshape(B, H, W, C).permute(0,3,1,2)
            else:
                skips.append(x)
                x = layer(x)

        # Bottleneck
        for blk in self.bottleneck:
            B, C, H, W = x.shape
            seq = x.permute(0,2,3,1).reshape(B, H*W, C)
            seq = blk(seq)
            x = seq.reshape(B, H, W, C).permute(0,3,1,2)

        # Decoder
        for layer in self.decoder:
            if isinstance(layer, nn.ConvTranspose2d):
                x = layer(x)
                skip = skips.pop()
                x = torch.cat([x, skip], dim=1)
            elif isinstance(layer, nn.ModuleList):
                B, C, H, W = x.shape
                seq = x.permute(0,2,3,1).reshape(B, H*W, C)
                for blk in layer:
                    seq = blk(seq)
                x = seq.reshape(B, H, W, C).permute(0,3,1,2)
            else:  # projection Conv2d
                x = layer(x)
        
        # project to num_classes
        x = self.final_conv(x)
        # upsample back if needed (else a no-op)
        x = self.final_upsample(x)
        return x

if __name__ == "__main__":
    from torchinfo import summary
    import yaml
    # Load YAML config
    config = yaml.safe_load(open("./config/config.yaml"))    
    model = FocalUNet( config)
    dummy = torch.randn(1, 10, 128, 128)
    out = model(dummy)
    print("Output shape:", out.shape)  # (1, 10, 128, 128)
    # Display summary 
    summary(model, input_size=(1, config['model']['in_channels'], config['model']['img_size'], config['model']['img_size']))
