import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm2d(nn.Module):
    """针对 2D 特征的 LayerNorm"""

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x):
        # 输入格式: [B, C, H, W]
        x = x.permute(0, 2, 3, 1)  # [B, H, W, C]
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)  # 恢复原始维度
        return x


class ConvNeXtBlock(nn.Module):
    """ConvNeXt 基础块 (无下采样)"""

    def __init__(self, dim, expansion_ratio=4):
        super().__init__()
        hidden_dim = dim * expansion_ratio
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm2d(dim)
        self.pwconv1 = nn.Conv2d(dim, hidden_dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(hidden_dim, dim, kernel_size=1)

    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        return residual + x


class ConvNeXtForSegmentation(nn.Module):
    def __init__(
            self,
            in_chans=10,
            depths=[3, 3],
            dims=[96, 192],
            num_classes=13,
            patch_size=1
    ):
        super().__init__()

        # 初始投影
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=3, stride=patch_size, padding=1),
            LayerNorm2d(dims[0])
        )

        # 仅构建不降分辨率的阶段
        self.stages = nn.ModuleList()
        for i in range(len(depths)):
            stage = nn.Sequential(
                *[ConvNeXtBlock(dims[i]) for _ in range(depths[i])]
            )
            self.stages.append(stage)

        # 分割头
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(dims[-1], dims[-1] // 2, 3, padding=1),
            LayerNorm2d(dims[-1] // 2),
            nn.GELU(),
            nn.Conv2d(dims[-1] // 2, dims[-1] // 4, 3, padding=1),
            LayerNorm2d(dims[-1] // 4),
            nn.GELU(),
            nn.Conv2d(dims[-1] // 4, num_classes, 1)
        )

    def forward(self, x):
        x = self.stem(x)  # [B, in_chans, H, W] -> [B, dims[0], H, W]
        for stage in self.stages:
            x = stage(x)  # 保持尺度不变
        seg_map = self.segmentation_head(x)  # [B, num_classes, H, W]
        return seg_map


def convnext_segmentation(in_chans= 10, num_classes=13):
                                                        
    """专为语义分割设计的轻量版"""
    return ConvNeXtForSegmentation(
        in_chans=in_chans,
        depths=[2, 2],
        dims=[128, 128],
        num_classes=num_classes,
        patch_size=1
    )

def count_parameters(model):
    """Count parameters by layer type"""
    conv_params = 0
    norm_params = 0
    total_params = 0
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'conv' in name or 'dwconv' in name or 'pwconv' in name:
                conv_params += param.numel()
            elif 'norm' in name:
                norm_params += param.numel()
            total_params += param.numel()
    
    return conv_params, norm_params, total_params


def main():
    """Test function to verify the segmentation model works correctly"""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dummy input (simulating typical remote sensing/medical image)
    # Parameters: [batch_size, channels, height, width]
    batch_size = 2
    channels = 10  # matches in_chans=10
    height = 224
    width = 224


    # Create model instance
    model = convnext_segmentation(in_chans=channels, num_classes=13)
    model = model.to(device)
    model.eval()
    
    # Print model summary
    print("\n" + "="*60)
    print("MODEL SUMMARY")
    print("="*60)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    
    
    dummy_input = torch.randn(batch_size, channels, height, width).to(device)
    
    print("\n" + "="*60)
    print("DUMMY INPUT SPECIFICATIONS")
    print("="*60)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Input type: {dummy_input.dtype}")
    print(f"Input range: [{dummy_input.min():.3f}, {dummy_input.max():.3f}]")
    print(f"Input mean: {dummy_input.mean():.3f}, std: {dummy_input.std():.3f}")
    
    # Forward pass
    print("\n" + "="*60)
    print("FORWARD PASS")
    print("="*60)
    
    with torch.no_grad():
        # Track intermediate shapes
        print("Processing through model layers:")
        
        # Stem
        x = model.stem(dummy_input)
        print(f"  After stem: {x.shape}")
        
        # Stages
        for i, stage in enumerate(model.stages):
            x_before = x.shape
            x = stage(x)
            print(f"  After stage {i}: {x_before} -> {x.shape}")
        
        # Segmentation head
        output = model.segmentation_head(x)
        print(f"  After segmentation head: {output.shape}")
    
    # Verify output
    print("\n" + "="*60)
    print("OUTPUT VERIFICATION")
    print("="*60)
    print(f"Output shape: {output.shape}")
    print(f"Expected shape: [batch_size={batch_size}, num_classes=13, height={height}, width={width}]")
    
    # Check output properties
    if output.shape == (batch_size, 13, height, width):
        print("✓ Output shape is CORRECT!")
    else:
        print("✗ Output shape is INCORRECT!")
    
    print(f"\nOutput statistics:")
    print(f"  Min value: {output.min():.4f}")
    print(f"  Max value: {output.max():.4f}")
    print(f"  Mean value: {output.mean():.4f}")
    
    # Add parameter breakdown
    print("\n" + "="*60)
    print("PARAMETER BREAKDOWN")
    print("="*60)
    
    conv_params, norm_params, total_params = count_parameters(model)
    print(f"Convolutional parameters: {conv_params:,} ({conv_params/total_params*100:.1f}%)")
    print(f"Normalization parameters: {norm_params:,} ({norm_params/total_params*100:.1f}%)")
    print(f"Total parameters: {total_params:,}")
    
    # Count blocks
    num_blocks = sum(len(stage) for stage in model.stages)
    print(f"\nNumber of ConvNeXt blocks: {num_blocks}")
    
    # Print detailed layer info
    print("\n" + "="*60)
    print("LAYER DETAILS")
    print("="*60)
    
    print("\nStem:")
    print(f"  Conv2d: {model.stem[0].in_channels}→{model.stem[0].out_channels}")
    
    for i, stage in enumerate(model.stages):
        print(f"\nStage {i+1} ({len(stage)} blocks):")
        for j, block in enumerate(stage):
            print(f"  Block {j+1}:")
            print(f"    Depthwise Conv: {block.dwconv.in_channels} channels, 7x7 kernel")
            print(f"    Pointwise Conv1: {block.pwconv1.in_channels}→{block.pwconv1.out_channels}")
            print(f"    Pointwise Conv2: {block.pwconv2.in_channels}→{block.pwconv2.out_channels}")
    
    print("\nSegmentation Head:")
    print(f"  Conv1: {model.segmentation_head[0].in_channels}→{model.segmentation_head[0].out_channels}")
    print(f"  Conv2: {model.segmentation_head[3].in_channels}→{model.segmentation_head[3].out_channels}")
    print(f"  Conv3: {model.segmentation_head[6].in_channels}→{model.segmentation_head[6].out_channels}")

    # Test with different input sizes (if needed)
    print("\n" + "="*60)
    print("ADDITIONAL TESTS")
    print("="*60)
    
    # Test 1: Different batch size
    test_input = torch.randn(4, channels, 128, 128).to(device)
    with torch.no_grad():
        test_output = model(test_input)
    print(f"Test 1 - Different batch/size: {test_input.shape} -> {test_output.shape}")
    
    # Test 2: Odd dimensions
    test_input = torch.randn(1, channels, 101, 101).to(device)
    with torch.no_grad():
        test_output = model(test_input)
    print(f"Test 2 - Odd dimensions: {test_input.shape} -> {test_output.shape}")
    
    # Memory usage
    print("\n" + "="*60)
    print("MEMORY USAGE")
    print("="*60)
    if torch.cuda.is_available():
        print(f"GPU Memory allocated: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")
        print(f"GPU Memory cached: {torch.cuda.memory_reserved(device) / 1024**2:.2f} MB")
    
    return model, dummy_input, output


if __name__ == "__main__":
    model, dummy_input, output = main()
    
    print("\n" + "="*60)
    print("TEST COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nModel is ready for training/inference.")
    print(f"Input: {dummy_input.shape}")
    print(f"Output: {output.shape}")