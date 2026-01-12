# model/sepvit_unet.py

import torch
import torch.nn as nn
from functools import partial
from .SepVIT import OverlappingPatchEmbed, PEG, Transformer

class SepViTUNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        m  = config['model']['sepvit']

        # Basic parameters
        in_ch       = config['model']['in_channels']
        out_ch      = config['model']['num_classes']
        stem_dim    = config['model']['stem_dim']
        stem_k      = config['model']['stem_kernel']
        stem_p      = config['model']['stem_padding']
        do_stem_ds  = config['model'].get('stem_downsampling', False)
        depths      = config['model']['depths']
        num_heads    = config['model']['heads']
        dims= config['model']['dims']
        img_size= config['model']['img_size']
        ws= config['model']['local_window_size']

        # SepViT hyperparams
        H = W = img_size if isinstance(img_size, int) else tuple(img_size)
        ff_mult     = config['model'].get('ff_mult', m.get('mlp_ratio', 4.0))
        drop_rate   = m['dropout']

        # ─── Stem ───────────────────────────────────────────────────────
        stem_layers = [
            nn.Conv2d(in_ch, stem_dim, kernel_size=stem_k, padding=stem_p, bias=False),
            nn.BatchNorm2d(stem_dim),
            nn.GELU(),
        ]
        if do_stem_ds:
            stem_layers.append(nn.AvgPool2d(2,2))  # 128→64
        self.stem = nn.Sequential(*stem_layers)

        # ─── Encoder ───────────────────────────────────────────────────
        self.encoder = nn.ModuleList()
        for i, depth in enumerate(depths):
            # 1) Downsample spatially + channel
            if i>0:
                self.encoder.append(
                    nn.Conv2d(dims[i-1], dims[i], kernel_size=2, stride=2)
                )
            # 2) Transformer stack at this res
            blocks = nn.ModuleList()
            for _ in range(depth):
                blocks.append(nn.Sequential(
                    OverlappingPatchEmbed(dim_in=dims[i], dim_out=dims[i], stride=1),
                    PEG(dims[i]),
                    Transformer(
                        dim=dims[i], depth=1, heads=num_heads,
                        window_size=ws, ff_mult=ff_mult,
                        dropout=drop_rate, norm_output=False
                    )
                ))
            self.encoder.append(blocks)

        # ─── Bottleneck ────────────────────────────────────────────────
        self.bottleneck = nn.ModuleList()
        for _ in range(2):
            self.bottleneck.append(nn.Sequential(
                OverlappingPatchEmbed(dim_in=dims[-1], dim_out=dims[-1], stride=1),
                PEG(dims[-1]),
                Transformer(
                    dim=dims[-1], depth=1, heads=num_heads,
                    window_size=ws, ff_mult=ff_mult,
                    dropout=drop_rate, norm_output=False
                )
            ))

        # ─── Decoder ───────────────────────────────────────────────────
        self.decoder = nn.ModuleList()
        for i in reversed(range(len(depths)-1)):
            # 1) Upsample spatially & channel dims
            self.decoder.append(
                nn.ConvTranspose2d(dims[i+1], dims[i], kernel_size=2, stride=2)
            )
            # 2) Project 2× channels → dims[i], then run depth[i] blocks
            blocks = nn.ModuleList()
            # first block: OverlappingPatchEmbed on concat
            blocks.append(nn.Sequential(
                OverlappingPatchEmbed(dim_in=dims[i]*2, dim_out=dims[i], stride=1),
                PEG(dims[i]),
                Transformer(
                    dim=dims[i], depth=1, heads=num_heads,
                    window_size=ws, ff_mult=ff_mult,
                    dropout=drop_rate, norm_output=False
                )
            ))
            # remaining blocks: pure Transformer (no extra OPE)
            for _ in range(depths[i]-1):
                blocks.append(Transformer(
                    dim=dims[i], depth=1, heads=num_heads,
                    window_size=ws, ff_mult=ff_mult,
                    dropout=drop_rate, norm_output=False
                ))
            self.decoder.append(blocks)

        # ─── Final Upsample to original H×W ────────────────────────────
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
        # Stem
        x = self.stem(x)
        skips = []

        # Encoder
        for layer in self.encoder:
            if isinstance(layer, nn.ModuleList):
                for blk in layer:
                    x = blk(x)
            else:
                skips.append(x)
                x = layer(x)

        # Bottleneck
        for blk in self.bottleneck:
            x = blk(x)

        # Decoder
        for layer in self.decoder:
            if isinstance(layer, nn.ConvTranspose2d):
                x = layer(x)
                skip = skips.pop()
                x = torch.cat([x, skip], dim=1)
            else:
                for blk in layer:
                    x = blk(x)

        # project to num_classes
        x = self.final_conv(x)
        # upsample back if needed (else a no-op)
        x = self.final_upsample(x)
        return x


if __name__ == "__main__":
    from torchinfo import summary
    import yaml
    config = yaml.safe_load(open("./config/config.yaml"))
    model = SepViTUNet(config)
    dummy = torch.randn(1,
        config['model']['in_channels'],
        *(config['model']['img_size'],)*2
    )
    out = model(dummy)
    print("Output shape:", out.shape)  # (1, num_classes, H, W)

    # Display summary 
    summary(model, input_size=(1, config['model']['in_channels'], config['model']['img_size'], config['model']['img_size']))
