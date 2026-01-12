import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from .twins import CPVTV2

class TwinsUNet(nn.Module):
    """
    UNet-style segmentation using a configurable Twins (CPVTV2) encoder.

    Configurable parameters (in config['model']):
      - in_channels, num_classes, img_size
      - twins: dict containing all CPVTV2 kwargs:
          * patch_size, embed_dims, depths, num_heads, mlp_ratios,
            qkv_bias, qk_scale, drop_rate, attn_drop_rate, drop_path_rate,
            sr_ratios, block_cls, F4, extra_norm
    """
    def __init__(self, config):
        super().__init__()
        m = config['model']
        in_ch = m['in_channels']
        out_ch = m['num_classes']
        img_size = m['img_size']

        # --- Twins encoder parameters ---
        twins_cfg = m['twins']
        # ensure required keys exist or provide defaults
        tw_params = {
            'img_size': img_size,
            'patch_size': twins_cfg.get('patch_size', 4),
            'in_chans': in_ch,
            'num_classes': out_ch,
            'embed_dims': twins_cfg.get('embed_dims', [64,128,256,512]),
            'num_heads': twins_cfg.get('num_heads', [1,2,4,8]),
            'mlp_ratios': twins_cfg.get('mlp_ratios', [4]*len(twins_cfg.get('depths', []))),
            'qkv_bias': twins_cfg.get('qkv_bias', True),
            'qk_scale': twins_cfg.get('qk_scale', None),
            'drop_rate': twins_cfg.get('drop_rate', 0.0),
            'attn_drop_rate': twins_cfg.get('attn_drop_rate', 0.0),
            'drop_path_rate': twins_cfg.get('drop_path_rate', 0.0),
            'norm_layer': partial(nn.LayerNorm, eps=1e-6),
            'depths': twins_cfg.get('depths', []),
            'sr_ratios': twins_cfg.get('sr_ratios', [1]*len(twins_cfg.get('depths', []))),
            'block_cls': twins_cfg.get('block_cls', None),
            'F4': twins_cfg.get('F4', False),
            'extra_norm': twins_cfg.get('extra_norm', False)
        }
        # create encoder
        # remove block_cls if not explicitly set so default Block is used
        if tw_params.get('block_cls') is None:
            tw_params.pop('block_cls')
        self.encoder = CPVTV2(**tw_params)
        embed_dims = tw_params['embed_dims']
        depths = tw_params['depths']

        # --- Decoder: mirror encoder stages ---
        self.ups = nn.ModuleList()
        self.decoder_convs = nn.ModuleList()
        # for each upsampling from stage i -> i-1
        for i in range(len(embed_dims)-1, 0, -1):
            # learnable upsample
            self.ups.append(
                nn.ConvTranspose2d(embed_dims[i], embed_dims[i-1], kernel_size=2, stride=2)
            )
            # build convs to mirror encoder depth at stage i-1
            conv_blocks = []
            # both upsampled and skip have embed_dims[i-1] channels
            in_ch_dec = embed_dims[i-1] * 2
            out_ch_dec = embed_dims[i-1]
            # one conv-block per Transformer block depth
            for _ in range(depths[i-1] if depths else 1):
                conv_blocks.extend([
                    nn.Conv2d(in_ch_dec, out_ch_dec, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(out_ch_dec),
                    nn.ReLU(inplace=True)
                ])
                # after first conv, input channels reduce
                in_ch_dec = out_ch_dec
            self.decoder_convs.append(nn.Sequential(*conv_blocks))

        # segmentation head
        self.seg_head = nn.Conv2d(embed_dims[0], out_ch, kernel_size=1)

    def forward(self, x):
        # get multi-scale features from Twins encoder
        feats = self.encoder.forward_features(x)
        x = feats[-1]  # deepest features
        # decode
        for up, conv, skip in zip(self.ups, self.decoder_convs, reversed(feats[:-1])):
            x = up(x)
            x = torch.cat([x, skip], dim=1)
            x = conv(x)
        # final logits
        return self.seg_head(x)

if __name__ == "__main__":
    from torchinfo import summary
    import yaml
    config = yaml.safe_load(open("./config/config.yaml"))
    model = TwinsUNet(config)
    # dummy input
    img_size = config['model']['img_size']
    batch = (1, config['model']['in_channels'], img_size, img_size) if isinstance(img_size, int) else (1, config['model']['in_channels'], *img_size)
    dummy = torch.randn(*batch)
    seg = model(dummy)
    print("Seg output shape:", seg.shape)
    summary(model, input_size=batch)
