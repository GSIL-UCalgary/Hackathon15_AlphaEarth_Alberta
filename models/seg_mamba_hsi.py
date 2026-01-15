"""
MambaHSI for Semantic Segmentation
Architecture: Stem (4x downsample) → Encoder → Segmentation Head (4x upsample)
Spatial progression: 224×224 → 56×56 → 28×28 → 14×14 → 7×7 → 224×224
"""

import math
import torch
from torch import nn
from mamba_ssm import Mamba


class SpeMamba(nn.Module):
    def __init__(self, channels, token_num=8, use_residual=True, group_num=4):
        super(SpeMamba, self).__init__()
        self.token_num = token_num
        self.use_residual = use_residual

        self.group_channel_num = math.ceil(channels/token_num)
        self.channel_num = self.token_num * self.group_channel_num

        self.mamba = Mamba(
            d_model=self.group_channel_num,
            d_state=16,
            d_conv=4,
            expand=2,
        )

        self.proj = nn.Sequential(
            nn.GroupNorm(group_num, self.channel_num),
            nn.SiLU()
        )

    def padding_feature(self, x):
        B, C, H, W = x.shape
        if C < self.channel_num:
            pad_c = self.channel_num - C
            pad_features = torch.zeros((B, pad_c, H, W)).to(x.device)
            cat_features = torch.cat([x, pad_features], dim=1)
            return cat_features
        else:
            return x

    def forward(self, x):
        x_pad = self.padding_feature(x)
        x_pad = x_pad.permute(0, 2, 3, 1).contiguous()
        B, H, W, C_pad = x_pad.shape
        x_flat = x_pad.view(B * H * W, self.token_num, self.group_channel_num)
        x_flat = self.mamba(x_flat)
        x_recon = x_flat.view(B, H, W, C_pad)
        x_recon = x_recon.permute(0, 3, 1, 2).contiguous()
        x_proj = self.proj(x_recon)
        if self.use_residual:
            return x + x_proj
        else:
            return x_proj


class SpaMamba(nn.Module):
    def __init__(self, channels, use_residual=True, group_num=4, use_proj=True):
        super(SpaMamba, self).__init__()
        self.use_residual = use_residual
        self.use_proj = use_proj
        self.mamba = Mamba(
            d_model=channels,
            d_state=16,
            d_conv=4,
            expand=2,
        )
        if self.use_proj:
            self.proj = nn.Sequential(
                nn.GroupNorm(group_num, channels),
                nn.SiLU()
            )

    def forward(self, x):
        x_re = x.permute(0, 2, 3, 1).contiguous()
        B, H, W, C = x_re.shape
        x_flat = x_re.view(1, -1, C)
        x_flat = self.mamba(x_flat)

        x_recon = x_flat.view(B, H, W, C)
        x_recon = x_recon.permute(0, 3, 1, 2).contiguous()
        if self.use_proj:
            x_recon = self.proj(x_recon)
        if self.use_residual:
            return x_recon + x
        else:
            return x_recon


class BothMamba(nn.Module):
    def __init__(self, channels, token_num, use_residual, group_num=4, use_att=True):
        super(BothMamba, self).__init__()
        self.use_att = use_att
        self.use_residual = use_residual
        if self.use_att:
            self.weights = nn.Parameter(torch.ones(2) / 2)
            self.softmax = nn.Softmax(dim=0)

        self.spa_mamba = SpaMamba(channels, use_residual=use_residual, group_num=group_num)
        self.spe_mamba = SpeMamba(channels, token_num=token_num, use_residual=use_residual, group_num=group_num)

    def forward(self, x):
        spa_x = self.spa_mamba(x)
        spe_x = self.spe_mamba(x)
        if self.use_att:
            weights = self.softmax(self.weights)
            fusion_x = spa_x * weights[0] + spe_x * weights[1]
        else:
            fusion_x = spa_x + spe_x
        if self.use_residual:
            return fusion_x + x
        else:
            return fusion_x


class MambaHSISeg(nn.Module):
    """
    MambaHSI for Semantic Segmentation
    
    Architecture (with stem):
    Input: (B, in_channels, 224, 224)
    ↓ Stem (Conv + MaxPool) - optional, controlled by use_stem
    (B, c1, 56, 56)
    ↓ Stage 1 + Downsample
    (B, 2*c1, 28, 28)
    ↓ Stage 2 + Downsample
    (B, 4*c1, 14, 14)
    ↓ Stage 3 + Downsample
    (B, 8*c1, 7, 7)
    ↓ Segmentation Head (Upsample 32x or 8x)
    Output: (B, num_classes, 224, 224)
    
    Architecture (without stem):
    Input: (B, in_channels, 224, 224)
    ↓ 1x1 Conv to base_dim (no downsampling)
    (B, c1, 224, 224)
    ↓ Stage 1 + Downsample
    (B, 2*c1, 112, 112)
    ↓ Stage 2 + Downsample
    (B, 4*c1, 56, 56)
    ↓ Stage 3 + Downsample
    (B, 8*c1, 28, 28)
    ↓ Segmentation Head (Upsample 8x)
    Output: (B, num_classes, 224, 224)
    """
    
    def __init__(self, 
                 in_channels=10, 
                 base_dim=32,  
                 num_classes=13, 
                 use_residual=True, 
                 mamba_type='both', 
                 token_num=4, 
                 group_num=4, 
                 use_att=True,
                 use_stem=True):  
        super(MambaHSISeg, self).__init__()
        
        self.mamba_type = mamba_type
        self.use_stem = use_stem
        self.dims = [base_dim, base_dim*2, base_dim*4, base_dim*8]  # [c1, 2c1, 4c1, 8c1]
        
        # ==================== STEM (Optional) ====================
        if use_stem:
            # Downsample 224x224 → 56x56 (4x reduction)
            self.stem = nn.Sequential(
                nn.Conv2d(in_channels, base_dim, kernel_size=7, stride=2, padding=3),  # 224→112
                nn.GroupNorm(group_num, base_dim),
                nn.SiLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)  # 112→56
            )
        else:
            # No downsampling, just project to base_dim
            # 224x224 → 224x224
            self.stem = nn.Sequential(
                nn.Conv2d(in_channels, base_dim, kernel_size=1),
                nn.GroupNorm(group_num, base_dim),
                nn.SiLU()
            )
        
        # ==================== ENCODER ====================
        # Build encoder stages with channel doubling
        self.encoder = nn.ModuleList()
        self.downsample = nn.ModuleList()
        
        for i in range(len(self.dims)):
            # Current stage dimension
            dim = self.dims[i]
            
            # Mamba block
            if mamba_type == 'spa':
                block = SpaMamba(dim, use_residual=use_residual, group_num=group_num)
            elif mamba_type == 'spe':
                block = SpeMamba(dim, token_num=token_num, use_residual=use_residual, group_num=group_num)
            elif mamba_type == 'both':
                block = BothMamba(dim, token_num=token_num, use_residual=use_residual, 
                                group_num=group_num, use_att=use_att)
            
            self.encoder.append(block)
            
            # Downsample (except last stage)
            if i < len(self.dims) - 1:
                # Downsample by 2x and double channels
                downsample = nn.Sequential(
                    nn.Conv2d(dim, self.dims[i+1], kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(group_num, self.dims[i+1]),
                    nn.SiLU()
                )
                self.downsample.append(downsample)
        
        # ==================== SEGMENTATION HEAD ====================
        if use_stem:
            # With stem: Upsample 7x7 → 224x224 (32x upsampling)
            # Progressive upsampling: 7→14→28→56→112→224
            self.seg_head = nn.Sequential(
                # 7x7 → 14x14
                nn.ConvTranspose2d(self.dims[-1], self.dims[-2], kernel_size=2, stride=2),
                nn.GroupNorm(group_num, self.dims[-2]),
                nn.SiLU(),
                
                # 14x14 → 28x28
                nn.ConvTranspose2d(self.dims[-2], self.dims[-3], kernel_size=2, stride=2),
                nn.GroupNorm(group_num, self.dims[-3]),
                nn.SiLU(),
                
                # 28x28 → 56x56
                nn.ConvTranspose2d(self.dims[-3], self.dims[-4], kernel_size=2, stride=2),
                nn.GroupNorm(group_num, self.dims[-4]),
                nn.SiLU(),
                
                # 56x56 → 112x112
                nn.ConvTranspose2d(self.dims[-4], self.dims[-4], kernel_size=2, stride=2),
                nn.GroupNorm(group_num, self.dims[-4]),
                nn.SiLU(),
                
                # 112x112 → 224x224
                nn.ConvTranspose2d(self.dims[-4], self.dims[-4], kernel_size=2, stride=2),
                nn.GroupNorm(group_num, self.dims[-4]),
                nn.SiLU(),
                
                # Final 1x1 conv to get num_classes
                nn.Conv2d(self.dims[-4], num_classes, kernel_size=1)
            )
        else:
            # Without stem: Upsample 28x28 → 224x224 (8x upsampling)
            # Progressive upsampling: 28→56→112→224
            self.seg_head = nn.Sequential(
                # 28x28 → 56x56
                nn.ConvTranspose2d(self.dims[-1], self.dims[-2], kernel_size=2, stride=2),
                nn.GroupNorm(group_num, self.dims[-2]),
                nn.SiLU(),
                
                # 56x56 → 112x112
                nn.ConvTranspose2d(self.dims[-2], self.dims[-3], kernel_size=2, stride=2),
                nn.GroupNorm(group_num, self.dims[-3]),
                nn.SiLU(),
                
                # 112x112 → 224x224
                nn.ConvTranspose2d(self.dims[-3], self.dims[-4], kernel_size=2, stride=2),
                nn.GroupNorm(group_num, self.dims[-4]),
                nn.SiLU(),
                
                # Final 1x1 conv to get num_classes
                nn.Conv2d(self.dims[-4], num_classes, kernel_size=1)
            )

    def forward(self, x):
        # Stem: 224×224 → 56×56 (if use_stem=True) or 224×224 → 224×224 (if use_stem=False)
        x = self.stem(x)
        
        # Encoder with downsampling
        if self.use_stem:
            # With stem:
            # Stage 0: 56×56, c1
            # Stage 1: 28×28, 2c1  
            # Stage 2: 14×14, 4c1
            # Stage 3: 7×7, 8c1
            pass
        else:
            # Without stem:
            # Stage 0: 224×224, c1
            # Stage 1: 112×112, 2c1
            # Stage 2: 56×56, 4c1
            # Stage 3: 28×28, 8c1
            pass
        
        for i in range(len(self.encoder)):
            x = self.encoder[i](x)
            if i < len(self.downsample):
                x = self.downsample[i](x)
        
        # Segmentation head: upsample to 224×224
        logits = self.seg_head(x)
        
        return logits


# ==================== TEST CODE ====================
def test_mambahsi_seg():
    """Test the segmentation model"""
    print("="*80)
    print("Testing MambaHSI Segmentation Model")
    print("="*80)
    
    # Check for CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    if not torch.cuda.is_available():
        print("  WARNING: CUDA not available. Mamba requires GPU!")
        return
    
    # Configuration
    batch_size = 2
    in_channels = 10
    num_classes = 13
    base_dim = 32
    
    # Test both with and without stem
    for use_stem in [True, False]:
        print(f"\n{'='*80}")
        print(f"Testing with use_stem={use_stem}")
        print(f"{'='*80}")
        
        print(f"\nConfiguration:")
        print(f"  Input channels: {in_channels}")
        print(f"  Num classes: {num_classes}")
        print(f"  Base dimension (c1): {base_dim}")
        print(f"  Use stem (4x downsample): {use_stem}")
        
        # Create model
        model = MambaHSISeg(
            in_channels=in_channels,
            base_dim=base_dim,
            num_classes=num_classes,
            mamba_type='both',
            token_num=4,
            use_residual=True,
            use_att=True,
            use_stem=use_stem
        ).to(device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {total_params:,} ({total_params/1e6:.2f}M)")
        
        # Test forward pass
        x = torch.randn(batch_size, in_channels, 224, 224).to(device)
        
        model.eval()
        with torch.no_grad():
            output = model(x)
        
        print(f"\n✓ Forward pass successful!")
        print(f"  Input:  {tuple(x.shape)}")
        print(f"  Output: {tuple(output.shape)}")
        
        assert output.shape == (batch_size, num_classes, 224, 224), \
            f"Expected output shape ({batch_size}, {num_classes}, 224, 224), got {output.shape}"
        print(f"  ✓ Output shape correct!")


def analyze_dimensions():
    """Analyze spatial dimensions through the network"""
    print("\n" + "="*80)
    print("Spatial Dimension Analysis")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        print("⚠️  Skipping - CUDA required")
        return
    
    # Test both configurations
    for use_stem in [True, False]:
        print(f"\n{'='*80}")
        print(f"Configuration: use_stem={use_stem}")
        print(f"{'='*80}")
        
        model = MambaHSISeg(
            in_channels=10,
            base_dim=32,
            num_classes=13,
            mamba_type='both',
            use_stem=use_stem
        ).to(device)
        
        x = torch.randn(1, 10, 224, 224).to(device)
        
        print(f"\nInput: {tuple(x.shape)}")
        
        # Stem
        x = model.stem(x)
        print(f"After stem: {tuple(x.shape)}")
        
        # Encoder stages
        for i in range(len(model.encoder)):
            x = model.encoder[i](x)
            print(f"After encoder[{i}]: {tuple(x.shape)}", end="")
            if i < len(model.downsample):
                x = model.downsample[i](x)
                print(f" → after downsample: {tuple(x.shape)}")
            else:
                print()
        
        # Segmentation head
        x = model.seg_head(x)
        print(f"After seg_head: {tuple(x.shape)} [224×224, {x.shape[1]} classes]")
        
        if use_stem:
            print(f"\n✓ With stem: 224×224 → 56×56 → 28×28 → 14×14 → 7×7 → 224×224")
        else:
            print(f"\n✓ Without stem: 224×224 → 224×224 → 112×112 → 56×56 → 28×28 → 224×224")


# ==================== WRAPPER ====================
class MambaHSISegWrapper(nn.Module):
    """Simple wrapper for MambaHSISeg - matches HRNetWrapper style"""
    
    def __init__(self, config):
        super().__init__()
        
        # config should have 'in_channels' and 'num_classes' keys
        # Optional: 'base_dim', 'mamba_type', 'token_num', 'use_stem', etc.
        in_channels = config['in_channels']
        num_classes = config['num_classes']
        base_dim = config.get('base_dim', 32)
        mamba_type = config.get('mamba_type', 'both')
        token_num = config.get('token_num', 4)
        use_residual = config.get('use_residual', True)
        group_num = config.get('group_num', 4)
        use_att = config.get('use_att', True)
        use_stem = config.get('use_stem', True)  # NEW: control 4x downsampling
        
        # Create MambaHSISeg model
        self.model = MambaHSISeg(
            in_channels=in_channels,
            base_dim=base_dim,
            num_classes=num_classes,
            use_residual=use_residual,
            mamba_type=mamba_type,
            token_num=token_num,
            group_num=group_num,
            use_att=use_att,
            use_stem=use_stem
        )
    
    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    test_mambahsi_seg()
    analyze_dimensions()
    
    print("\n" + "="*80)
    print("All tests completed successfully! ✓")
    print("="*80)
    print("\n📊 ARCHITECTURE SUMMARY:")
    print("\nWith stem (use_stem=True):")
    print("  • Input:  (B, 10, 224, 224)")
    print("  • Stem:   224→56 (4x downsample)")
    print("  • Stage1: 56→28 with c1→2c1")
    print("  • Stage2: 28→14 with 2c1→4c1")
    print("  • Stage3: 14→7 with 4c1→8c1")
    print("  • Head:   7→224 (32x upsample)")
    print("  • Output: (B, 13, 224, 224)")
    
    print("\nWithout stem (use_stem=False):")
    print("  • Input:  (B, 10, 224, 224)")
    print("  • Stem:   224→224 (no downsample, just 1x1 conv)")
    print("  • Stage1: 224→112 with c1→2c1")
    print("  • Stage2: 112→56 with 2c1→4c1")
    print("  • Stage3: 56→28 with 4c1→8c1")
    print("  • Head:   28→224 (8x upsample)")
    print("  • Output: (B, 13, 224, 224)")
    
    # Test wrapper
    print("\n" + "="*80)
    print("Testing Wrapper")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        # Test with stem
        wrapper = MambaHSISegWrapper({
            'in_channels': 10,
            'num_classes': 13,
            'base_dim': 32,
            'mamba_type': 'both',
            'use_stem': True
        }).to(device)
        
        x = torch.randn(2, 10, 224, 224).to(device)
        output = wrapper(x)
        
        print(f"✓ Wrapper works correctly with stem!")
        print(f"  Output shape: {tuple(output.shape)}")
        
        # Test without stem
        wrapper_no_stem = MambaHSISegWrapper({
            'in_channels': 10,
            'num_classes': 13,
            'base_dim': 32,
            'mamba_type': 'both',
            'use_stem': False
        }).to(device)
        
        output_no_stem = wrapper_no_stem(x)
        
        print(f"✓ Wrapper works correctly without stem!")
        print(f"  Output shape: {tuple(output_no_stem.shape)}")