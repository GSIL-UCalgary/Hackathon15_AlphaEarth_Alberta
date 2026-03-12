import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class DownsampleBlock(nn.Module):
    """Downsampling block for initial feature extraction"""
    def __init__(self, in_chans, out_chans, kernel_size=3, stride=2, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_chans, out_chans, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_chans)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class PatchEmbedding(nn.Module):
    """Standard ViT Patch Embedding: Converts image to patches and projects to embeddings"""
    def __init__(self, img_size=56, patch_size=7, in_chans=128, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Standard ViT patch embedding using Conv2d
        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        # x: [B, C, H, W]
        x = self.proj(x)  # [B, embed_dim, H/patch, W/patch]
        x = rearrange(x, 'b c h w -> b (h w) c')  # [B, num_patches, embed_dim]
        return x

class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention"""
    def __init__(self, embed_dim=768, num_heads=12, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MLP(nn.Module):
    """MLP with GELU activation"""
    def __init__(self, in_features, hidden_features=None, out_features=None, dropout=0.1):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class TransformerBlock(nn.Module):
    """Standard Transformer encoder block"""
    def __init__(self, embed_dim=768, num_heads=12, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(
            in_features=embed_dim,
            hidden_features=int(embed_dim * mlp_ratio),
            dropout=dropout
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class SimpleViTSegmentation(nn.Module):
    """
    Simple ViT for Segmentation with:
    - Factor 4 downsampling to 56x56
    - Standard ViT with patch embedding (patch_size=7 → 8x8 patches)
    - 3 identical ViT layers with same embed_dim
    - Single factor 4 upsampling to 224x224
    """
    
    def __init__(
            self,
            img_size=224,
            in_chans=68,
            embed_dim=384,           # ViT embedding dimension
            patch_size=7,             # 56/7 = 8 patches per side
            depth=3,                   # Number of transformer blocks
            num_heads=6,               # Number of attention heads
            mlp_ratio=4.0,
            dropout=0.1,
            num_classes=13
    ):
        super().__init__()
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        
        # === FACTOR 4 DOWNSAMPLING: 224x224 -> 56x56 ===
        self.downsample_stem = nn.Sequential(
            DownsampleBlock(in_chans, embed_dim // 4, stride=2),  # 224 -> 112
            DownsampleBlock(embed_dim // 4, embed_dim // 2, stride=2),  # 112 -> 56
        )
        
        # After downsampling, we have 56x56 feature maps with embed_dim//2 channels
        self.downsampled_size = img_size // 4  # 56
        
        # === STANDARD VIT PATCH EMBEDDING ===
        # Convert 56x56 feature maps to patches
        # With patch_size=7, we get 8x8 = 64 patches
        self.patch_embed = PatchEmbedding(
            img_size=self.downsampled_size,
            patch_size=patch_size,
            in_chans=embed_dim // 2,  # Input channels from downsampling
            embed_dim=embed_dim
        )
        
        # Calculate number of patches
        self.num_patches = (self.downsampled_size // patch_size) ** 2  # (56/7)^2 = 64
        self.patch_h = self.downsampled_size // patch_size  # 8
        self.patch_w = self.downsampled_size // patch_size  # 8
        
        # === STANDARD VIT COMPONENTS ===
        # Class token (optional for segmentation, but keeping for consistency)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Position embedding - standard ViT has one position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        
        # Transformer encoder - identical blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # === SEGMENTATION HEAD ===
        # Project back to spatial and segment
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 2, 3, padding=1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim // 2, embed_dim // 4, 3, padding=1),
            nn.BatchNorm2d(embed_dim // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim // 4, num_classes, 1)
        )
        
        # Simple factor 4 upsampling
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        
        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B, C, H, W = x.shape
        
        # === DOWNSAMPLING: 224x224 -> 56x56 ===
        x_down = self.downsample_stem(x)  # [B, embed_dim//2, 56, 56]
        
        # === PATCH EMBEDDING: Convert to patches ===
        x_patches = self.patch_embed(x_down)  # [B, num_patches, embed_dim]
        
        # === ADD CLASS TOKEN ===
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x_patches = torch.cat((cls_tokens, x_patches), dim=1)  # [B, num_patches+1, embed_dim]
        
        # === ADD POSITION EMBEDDING ===
        x_patches = x_patches + self.pos_embed
        
        # === TRANSFORMER ENCODER ===
        for block in self.blocks:
            x_patches = block(x_patches)
        
        x_patches = self.norm(x_patches)
        
        # === REMOVE CLASS TOKEN ===
        patch_tokens = x_patches[:, 1:]  # [B, num_patches, embed_dim]
        
        # === RESHAPE TO SPATIAL ===
        patch_tokens = patch_tokens.transpose(1, 2).reshape(
            B, self.embed_dim, self.patch_h, self.patch_w
        )  # [B, embed_dim, 8, 8]
        
        # === SEGMENTATION HEAD ===
        seg_map = self.segmentation_head(patch_tokens)  # [B, num_classes, 8, 8]
        
        # === UPSAMPLE TO ORIGINAL SIZE ===
        seg_map = self.upsample(seg_map)  # [B, num_classes, 32, 32] -> Wait, this is wrong!
        
        # Fix: We need to upsample from 8x8 to 224x224 (factor 28, not 4)
        # Let's use a proper upsampling layer
        self.upsample_final = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)
        seg_map = self.upsample_final(patch_tokens)  # [B, num_classes, 224, 224]
        
        return seg_map

# Let me provide the corrected version:
class SimpleViTSegmentation(nn.Module):
    """
    Simple ViT for Segmentation with:
    - Factor 4 downsampling to 56x56
    - Standard ViT with patch embedding (patch_size=7 → 8x8 patches)
    - 3 identical ViT layers with same embed_dim
    - Single upsampling to 224x224
    """
    
    def __init__(
            self,
            img_size=224,
            in_chans=68,
            embed_dim=384,           # ViT embedding dimension
            patch_size=7,             # 56/7 = 8 patches per side
            depth=3,                   # Number of transformer blocks
            num_heads=6,               # Number of attention heads
            mlp_ratio=4.0,
            dropout=0.1,
            num_classes=13
    ):
        super().__init__()
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        
        # === FACTOR 4 DOWNSAMPLING: 224x224 -> 56x56 ===
        self.downsample_stem = nn.Sequential(
            DownsampleBlock(in_chans, embed_dim // 4, stride=2),  # 224 -> 112
            DownsampleBlock(embed_dim // 4, embed_dim // 2, stride=2),  # 112 -> 56
        )
        
        # After downsampling, we have 56x56 feature maps with embed_dim//2 channels
        self.downsampled_size = img_size // 4  # 56
        
        # === STANDARD VIT PATCH EMBEDDING ===
        # Convert 56x56 feature maps to patches
        # With patch_size=7, we get 8x8 = 64 patches
        self.patch_embed = PatchEmbedding(
            img_size=self.downsampled_size,
            patch_size=patch_size,
            in_chans=embed_dim // 2,  # Input channels from downsampling
            embed_dim=embed_dim
        )
        
        # Calculate number of patches
        self.num_patches = (self.downsampled_size // patch_size) ** 2  # (56/7)^2 = 64
        self.patch_h = self.downsampled_size // patch_size  # 8
        self.patch_w = self.downsampled_size // patch_size  # 8
        
        # === STANDARD VIT COMPONENTS ===
        # Class token (optional)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Position embedding - one for all patches
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        
        # Transformer encoder - identical blocks (depth=3)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # === SEGMENTATION HEAD ===
        # Work on 8x8 feature maps
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 2, 3, padding=1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim // 2, embed_dim // 4, 3, padding=1),
            nn.BatchNorm2d(embed_dim // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim // 4, num_classes, 1)
        )
        
        # Final upsampling from 8x8 to 224x224 (factor 28)
        self.upsample = nn.Upsample(size=(img_size, img_size), mode='bilinear', align_corners=False)
        
        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        
        # === DOWNSAMPLING: 224x224 -> 56x56 ===
        x_down = self.downsample_stem(x)  # [B, embed_dim//2, 56, 56]
        
        # === PATCH EMBEDDING: 56x56 -> 8x8 patches -> [B, 64, embed_dim] ===
        x_patches = self.patch_embed(x_down)  # [B, num_patches, embed_dim]
        
        # === ADD CLASS TOKEN ===
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x_patches = torch.cat((cls_tokens, x_patches), dim=1)  # [B, 65, embed_dim]
        
        # === ADD POSITION EMBEDDING (ONE FOR ALL LAYERS) ===
        x_patches = x_patches + self.pos_embed
        
        # === TRANSFORMER ENCODER (3 IDENTICAL LAYERS) ===
        for block in self.blocks:
            x_patches = block(x_patches)
        
        x_patches = self.norm(x_patches)
        
        # === REMOVE CLASS TOKEN ===
        patch_tokens = x_patches[:, 1:]  # [B, 64, embed_dim]
        
        # === RESHAPE TO SPATIAL: [B, embed_dim, 8, 8] ===
        patch_tokens = patch_tokens.transpose(1, 2).reshape(
            B, self.embed_dim, self.patch_h, self.patch_w
        )
        
        # === SEGMENTATION HEAD (works on 8x8) ===
        seg_map = self.segmentation_head(patch_tokens)  # [B, num_classes, 8, 8]
        
        # === UPSAMPLE TO 224x224 ===
        seg_map = self.upsample(seg_map)  # [B, num_classes, 224, 224]
        
        return seg_map

# Example usage:
if __name__ == "__main__":
    model = SimpleViTSegmentation(
        img_size=224,
        in_chans=68,
        embed_dim=384,     # ViT embedding dimension
        patch_size=7,       # 56/7 = 8 patches
        depth=3,            # 3 identical transformer layers
        num_heads=6,
        num_classes=13
    )
    
    # Test
    x = torch.randn(2, 68, 224, 224)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")