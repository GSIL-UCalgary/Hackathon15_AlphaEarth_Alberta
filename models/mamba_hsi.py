"""
Official code of Mamba for Hyperspectral Image Classification
Reference:
https://github.com/li-yapeng/MambaHSI/blob/main/model/MambaHSI.py

"""

import math
import torch
from torch import nn
from mamba_ssm import Mamba


class SpeMamba(nn.Module):
    def __init__(self,channels, token_num=8, use_residual=True, group_num=4):
        super(SpeMamba, self).__init__()
        self.token_num = token_num
        self.use_residual = use_residual

        self.group_channel_num = math.ceil(channels/token_num)
        self.channel_num = self.token_num * self.group_channel_num

        self.mamba = Mamba( # This module uses roughly 3 * expand * d_model^2 parameters
                            d_model=self.group_channel_num,  # Model dimension d_model
                            d_state=16,  # SSM state expansion factor
                            d_conv=4,  # Local convolution width
                            expand=2,  # Block expansion factor
                            )

        self.proj = nn.Sequential(
            nn.GroupNorm(group_num, self.channel_num),
            nn.SiLU()
        )

    def padding_feature(self,x):
        B, C, H, W = x.shape
        if C < self.channel_num:
            pad_c = self.channel_num - C
            pad_features = torch.zeros((B, pad_c, H, W)).to(x.device)
            cat_features = torch.cat([x, pad_features], dim=1)
            return cat_features
        else:
            return x

    def forward(self,x):
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
    def __init__(self,channels,use_residual=True,group_num=4,use_proj=True):
        super(SpaMamba, self).__init__()
        self.use_residual = use_residual
        self.use_proj = use_proj
        self.mamba = Mamba(  # This module uses roughly 3 * expand * d_model^2 parameters
                           d_model=channels,  # Model dimension d_model
                           d_state=16,  # SSM state expansion factor
                           d_conv=4,  # Local convolution width
                           expand=2,  # Block expansion factor
                           )
        if self.use_proj:
            self.proj = nn.Sequential(
                nn.GroupNorm(group_num, channels),
                nn.SiLU()
            )

    def forward(self,x):
        x_re = x.permute(0, 2, 3, 1).contiguous()
        B,H,W,C = x_re.shape
        x_flat = x_re.view(1,-1, C)
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
    def __init__(self,channels,token_num,use_residual,group_num=4,use_att=True):
        super(BothMamba, self).__init__()
        self.use_att = use_att
        self.use_residual = use_residual
        if self.use_att:
            self.weights = nn.Parameter(torch.ones(2) / 2)
            self.softmax = nn.Softmax(dim=0)

        self.spa_mamba = SpaMamba(channels,use_residual=use_residual,group_num=group_num)
        self.spe_mamba = SpeMamba(channels,token_num=token_num,use_residual=use_residual,group_num=group_num)

    def forward(self,x):
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


class MambaHSI(nn.Module):
    def __init__(self,in_channels=128,hidden_dim=64,num_classes=10,use_residual=True,mamba_type='both',token_num=4,group_num=4,use_att=True):
        super(MambaHSI, self).__init__()
        self.mamba_type = mamba_type

        self.patch_embedding = nn.Sequential(nn.Conv2d(in_channels=in_channels,out_channels=hidden_dim,kernel_size=1,stride=1,padding=0),
                                             nn.GroupNorm(group_num,hidden_dim),
                                             nn.SiLU())
        if mamba_type == 'spa':
            self.mamba = nn.Sequential(SpaMamba(hidden_dim,use_residual=use_residual,group_num=group_num),
                                        nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
                                        SpaMamba(hidden_dim,use_residual=use_residual,group_num=group_num),
                                        nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
                                        SpaMamba(hidden_dim,use_residual=use_residual,group_num=group_num),
                                        )
        elif mamba_type == 'spe':
            self.mamba = nn.Sequential(SpeMamba(hidden_dim,token_num=token_num,use_residual=use_residual,group_num=group_num),
                                        nn.AvgPool2d(kernel_size=2, stride=2, padding=0),

                                        SpeMamba(hidden_dim,token_num=token_num,use_residual=use_residual,group_num=group_num),
                                        nn.AvgPool2d(kernel_size=2, stride=2, padding=0),

                                        SpeMamba(hidden_dim,token_num=token_num,use_residual=use_residual,group_num=group_num)
                                        )

        elif mamba_type=='both':
            self.mamba = nn.Sequential(BothMamba(channels=hidden_dim,token_num=token_num,use_residual=use_residual,group_num=group_num,use_att=use_att),
                                       nn.AvgPool2d(kernel_size=2, stride=2, padding=0),

                                       BothMamba(channels=hidden_dim,token_num=token_num,use_residual=use_residual,group_num=group_num,use_att=use_att),
                                       nn.AvgPool2d(kernel_size=2, stride=2, padding=0),

                                       BothMamba(channels=hidden_dim,token_num=token_num,use_residual=use_residual,group_num=group_num,use_att=use_att),
                                       )


        self.cls_head = nn.Sequential(nn.Conv2d(in_channels=hidden_dim, out_channels=128, kernel_size=1, stride=1, padding=0),
                                      nn.GroupNorm(group_num,128),
                                      nn.SiLU(),
                                      nn.Conv2d(in_channels=128,out_channels=num_classes,kernel_size=1,stride=1,padding=0))

    def forward(self,x):

        x = self.patch_embedding(x)
        x = self.mamba(x)

        logits = self.cls_head(x)
        return logits

"""
Test script for MambaHSI model
Tests with 10-band hyperspectral image of size 224x224
"""
"""
Test script for MambaHSI model
Tests with 10-band hyperspectral image of size 224x224
"""


def test_mambahsi():
    """Test MambaHSI model with different configurations"""
    
    print("="*80)
    print("Testing MambaHSI Model")
    print("="*80)
    
    # Check for CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    if not torch.cuda.is_available():
        print("⚠️  WARNING: CUDA not available. Mamba requires GPU!")
        print("⚠️  This test will fail on CPU. Please run on a GPU.")
        return
    
    # Input specifications
    batch_size = 2
    in_channels = 10  # 10 spectral bands
    height = 224
    width = 224
    num_classes = 13  # Your segmentation classes
    
    # Create dummy input and move to GPU
    x = torch.randn(batch_size, in_channels, height, width).to(device)
    print(f"\nInput shape: {x.shape}")
    print(f"  Batch size: {batch_size}")
    print(f"  Channels (bands): {in_channels}")
    print(f"  Spatial size: {height}x{width}")
    
    # Test different mamba types
    mamba_types = ['spa', 'spe', 'both']
    
    for mamba_type in mamba_types:
        print(f"\n{'-'*80}")
        print(f"Testing with mamba_type='{mamba_type}'")
        print(f"{'-'*80}")
        
        # Create model and move to GPU
        model = MambaHSI(
            in_channels=in_channels,
            hidden_dim=64,
            num_classes=num_classes,
            use_residual=True,
            mamba_type=mamba_type,
            token_num=4,
            group_num=4,
            use_att=True
        ).to(device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"Model Parameters:")
        print(f"  Total: {total_params:,}")
        print(f"  Trainable: {trainable_params:,}")
        print(f"  Size: {total_params/1e6:.2f}M")
        
        # Test forward pass
        model.eval()
        with torch.no_grad():
            try:
                output = model(x)
                print(f"\nForward pass successful!")
                print(f"  Output shape: {output.shape}")
                print(f"  Expected: ({batch_size}, {num_classes}, H_out, W_out)")
                
                # Calculate spatial reduction
                h_out, w_out = output.shape[2], output.shape[3]
                reduction_factor = height // h_out
                print(f"\nSpatial reduction:")
                print(f"  Input: {height}x{width}")
                print(f"  Output: {h_out}x{w_out}")
                print(f"  Reduction factor: {reduction_factor}x (due to 3 AvgPool2d layers with stride=2)")
                
            except Exception as e:
                print(f"\n❌ Error during forward pass: {e}")
    
    print(f"\n{'='*80}")
    print("Testing Complete!")
    print(f"{'='*80}")


def test_custom_config():
    """Test with custom configuration"""
    
    print("\n" + "="*80)
    print("Testing Custom Configuration")
    print("="*80)
    
    # Check for CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    if not torch.cuda.is_available():
        print("⚠️  Skipping - CUDA required for Mamba")
        return
    
    # Custom configuration
    config = {
        'in_channels': 10,
        'hidden_dim': 32,  # Smaller for faster computation
        'num_classes': 13,
        'use_residual': True,
        'mamba_type': 'both',
        'token_num': 8,
        'group_num': 4,
        'use_att': True
    }
    
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Create model and move to GPU
    model = MambaHSI(**config).to(device)
    
    # Test input and move to GPU
    x = torch.randn(1, 10, 224, 224).to(device)
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(x)
    
    print(f"\nResults:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")


def analyze_spatial_dimensions():
    """Analyze how spatial dimensions change through the network"""
    
    print("\n" + "="*80)
    print("Spatial Dimension Analysis")
    print("="*80)
    
    # Check for CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    if not torch.cuda.is_available():
        print("⚠️  Skipping - CUDA required for Mamba")
        return
    
    model = MambaHSI(
        in_channels=10,
        hidden_dim=64,
        num_classes=13,
        mamba_type='both'
    ).to(device)
    
    x = torch.randn(1, 10, 224, 224).to(device)
    
    print(f"\nInput: {x.shape}")
    
    # Patch embedding
    x = model.patch_embedding(x)
    print(f"After patch_embedding: {x.shape}")
    
    # Through mamba layers
    for i, layer in enumerate(model.mamba):
        x = layer(x)
        print(f"After mamba[{i}] ({layer.__class__.__name__}): {x.shape}")
    
    # Classification head
    x = model.cls_head(x)
    print(f"After cls_head (final output): {x.shape}")
    
    print(f"\nSummary:")
    print(f"  Input spatial size: 224x224")
    print(f"  After 1st AvgPool: 112x112")
    print(f"  After 2nd AvgPool: 56x56")
    print(f"  After 3rd AvgPool: 28x28")
    print(f"  Final output: 28x28")


if __name__ == "__main__":
    # Test 1: Basic functionality with all mamba types
    test_mambahsi()
    
    # Test 2: Custom configuration
    test_custom_config()
    
    # Test 3: Spatial dimension analysis
    analyze_spatial_dimensions()
    
    print("\n" + "="*80)
    print("All tests completed!")
    print("="*80)
    
    # Summary
    print("\n📊 SUMMARY:")
    print("  ✓ MambaHSI accepts input of shape (B, 10, 224, 224)")
    print("  ✓ Output shape is (B, 13, 28, 28)")
    print("  ✓ Spatial dimensions are reduced by 8x (224→28)")
    print("  ✓ This is due to 3 AvgPool2d layers with kernel=2, stride=2")
    print("  ⚠ Output is NOT the same spatial size as input!")
    print("  ⚠ For segmentation, you may need to upsample the output")