import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from .cat import CATBlock

class CATUNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        # basic params
        in_ch = config['model']['in_channels']
        out_ch = config['model']['num_classes']
        stem_dim = config['model']['stem_dim']
        stem_k = config['model']['stem_kernel']
        stem_p = config['model']['stem_padding']
        do_stem_ds = config['model'].get('stem_downsampling', False)
        depths = config['model']['depths']
        num_heads = config['model']['heads']
        dims = config['model']['dims']
        raw_img_size = config['model']['img_size']
        # CAT-specific
        cat_cfg = config['model']['cat']
        patch_size = cat_cfg['window_size']
        mlp_ratio = cat_cfg['mlp_ratio']
        drop_rate = cat_cfg['drop_rate']
        ipsa_drop = cat_cfg['ipsa_attn_drop']
        cpsa_drop = cat_cfg['cpsa_attn_drop']
        drop_path_rate = cat_cfg['drop_path_rate']
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        # image dims
        if isinstance(raw_img_size, int):
            H = W = raw_img_size
        else:
            H, W = raw_img_size
        base_h = H // 2 if do_stem_ds else H
        base_w = W // 2 if do_stem_ds else W

        # drop_path schedule
        encoder_blocks = sum(depths) + len(depths)
        bottleneck_blocks = 2
        decoder_blocks = sum(depths[:-1])
        total_blocks = encoder_blocks + bottleneck_blocks + decoder_blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]
        dp_idx = 0

        # Stem
        stem_layers = [
            nn.Conv2d(in_ch, stem_dim, kernel_size=stem_k, padding=stem_p, bias=False),
            nn.BatchNorm2d(stem_dim),
            nn.GELU(),
        ]
        if do_stem_ds:
            stem_layers.append(nn.AvgPool2d(2, 2))
        self.stem = nn.Sequential(*stem_layers)

        # Encoder
        self.encoder = nn.ModuleList()
        for i_layer in range(len(depths)):
            res_h = base_h // (2 ** i_layer)
            res_w = base_w // (2 ** i_layer)
            blocks = []
            # IPSA blocks
            for _ in range(depths[i_layer]):
                blocks.append(
                    CATBlock(
                        dim=dims[i_layer],
                        input_resolution=(res_h, res_w),
                        num_heads=num_heads,
                        patch_size=patch_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=True,
                        qk_scale=None,
                        drop=drop_rate,
                        attn_drop=ipsa_drop,
                        drop_path=dpr[dp_idx],
                        act_layer=nn.GELU,
                        norm_layer=nn.LayerNorm,
                        attn_type="ipsa",
                        rpe=True
                    )
                )
                dp_idx += 1
            # CPSA block
            blocks.append(
                CATBlock(
                    dim=dims[i_layer],
                    input_resolution=(res_h, res_w),
                    num_heads=1,
                    patch_size=patch_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    qk_scale=None,
                    drop=drop_rate,
                    attn_drop=cpsa_drop,
                    drop_path=dpr[dp_idx],
                    act_layer=nn.GELU,
                    norm_layer=nn.LayerNorm,
                    attn_type="cpsa",
                    rpe=False
                )
            )
            dp_idx += 1
            self.encoder.append(nn.Sequential(*blocks))
            # downsample conv
            if i_layer < len(depths) - 1:
                self.encoder.append(nn.Conv2d(dims[i_layer], dims[i_layer + 1], kernel_size=2, stride=2))

        # Bottleneck
        self.bottleneck = nn.ModuleList()
        deep_h = base_h // (2 ** (len(depths) - 1))
        deep_w = base_w // (2 ** (len(depths) - 1))
        for _ in range(bottleneck_blocks):
            self.bottleneck.append(
                CATBlock(
                    dim=dims[-1],
                    input_resolution=(deep_h, deep_w),
                    num_heads=num_heads,
                    patch_size=patch_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    qk_scale=None,
                    drop=drop_rate,
                    attn_drop=ipsa_drop,
                    drop_path=dpr[dp_idx],
                    act_layer=nn.GELU,
                    norm_layer=nn.LayerNorm,
                    attn_type="ipsa",
                    rpe=True
                )
            )
            dp_idx += 1

        # Decoder
        self.decoder = nn.ModuleList()
        for i_layer in reversed(range(len(depths) - 1)):
            # upsample
            self.decoder.append(nn.ConvTranspose2d(dims[i_layer + 1], dims[i_layer], kernel_size=2, stride=2))
            res_h = base_h // (2 ** i_layer)
            res_w = base_w // (2 ** i_layer)
            blocks = []
            for _ in range(depths[i_layer]):
                blocks.append(
                    CATBlock(
                        dim=dims[i_layer] * 2,
                        input_resolution=(res_h, res_w),
                        num_heads=num_heads,
                        patch_size=patch_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=True,
                        qk_scale=None,
                        drop=drop_rate,
                        attn_drop=ipsa_drop,
                        drop_path=dpr[dp_idx],
                        act_layer=nn.GELU,
                        norm_layer=nn.LayerNorm,
                        attn_type="ipsa",
                        rpe=True
                    )
                )
                dp_idx += 1
            self.decoder.append(nn.Sequential(*blocks))
            self.decoder.append(nn.Conv2d(dims[i_layer] * 2, dims[i_layer], kernel_size=1, bias=False))

        # Final conv
        self.final_conv = nn.Conv2d(dims[0], out_ch, kernel_size=1, bias=False)
        self.final_upsample = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=2, stride=2, bias=False)
        if not do_stem_ds:
            self.final_upsample = nn.Identity()

    def forward(self, x):
        x = self.stem(x)
        skips = []
        # encoder
        for layer in self.encoder:
            if isinstance(layer, nn.Sequential):
                B, C, H, W = x.shape
                seq = x.permute(0, 2, 3, 1).reshape(B, H*W, C)
                seq = layer(seq)
                x = seq.reshape(B, H, W, C).permute(0, 3, 1, 2)
            else:
                skips.append(x)
                x = layer(x)
        # bottleneck
        for blk in self.bottleneck:
            B, C, H, W = x.shape
            seq = x.permute(0, 2, 3, 1).reshape(B, H*W, C)
            seq = blk(seq)
            x = seq.reshape(B, H, W, C).permute(0, 3, 1, 2)
        # decoder
        for layer in self.decoder:
            if isinstance(layer, nn.ConvTranspose2d):
                x = layer(x)
                skip = skips.pop()
                x = torch.cat([x, skip], dim=1)
            elif isinstance(layer, nn.Sequential):
                B, C, H, W = x.shape
                seq = x.permute(0, 2, 3, 1).reshape(B, H*W, C)
                seq = layer(seq)
                x = seq.reshape(B, H, W, C).permute(0, 3, 1, 2)
            else:
                x = layer(x)
        x = self.final_conv(x)
        x = self.final_upsample(x)
        return x

if __name__ == "__main__":
    from torchinfo import summary
    import yaml
    config = yaml.safe_load(open("./config/config.yaml"))
    model = CATUNet(config)
    img_size = config['model']['img_size']
    if isinstance(img_size, int):
        shape = (1, config['model']['in_channels'], img_size, img_size)
    else:
        shape = (1, config['model']['in_channels'], *img_size)
    dummy = torch.randn(*shape)
    out = model(dummy)
    print("Output shape:", out.shape)
    summary(model, input_size=shape)
