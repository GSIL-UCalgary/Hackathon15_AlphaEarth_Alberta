import torch
import torch.nn as nn
import torch.nn.functional as F
class PatchEmbedding(nn.Module):
    """将图像分割为 Patch 并嵌入到向量"""

    def __init__(self, img_size=128, patch_size=16, in_chans=10, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # 使用卷积层实现 Patch Embedding
        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x)  # [B, C, H, W] -> [B, embed_dim, H/patch, W/patch]
        x = rearrange(x, 'b c h w -> b (h w) c')  # 展平为序列
        return x


class MultiHeadSelfAttention(nn.Module):
    """多头自注意力机制"""

    def __init__(self, embed_dim=768, num_heads=12, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)  # Q, K, V 投影
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, N, head_dim]

        # 计算注意力分数
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # 加权求和
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    """多层感知机"""

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
    """Transformer 编码块"""

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
        # 残差连接 + LayerNorm
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformerForSegmentation(nn.Module):
    """用于语义分割的 ViT 模型"""

    def __init__(
            self,
            img_size=128,
            patch_size=16,
            in_chans=10,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.0,
            dropout=0.1,
            num_classes=13
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_features = embed_dim
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.patch_h = img_size // patch_size
        self.patch_w = img_size // patch_size

        # 位置编码和分类令牌
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Transformer 编码层
        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        
        # 分割头 - 上采样到原始分辨率
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 2, 3, padding=1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim // 2, embed_dim // 4, 3, padding=1),
            nn.BatchNorm2d(embed_dim // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim // 4, num_classes, 1)
        )
        
        # 上采样层
        self.upsample = nn.Upsample(size=img_size, mode='bilinear', align_corners=False)

        # 初始化权重
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B, C, H, W = x.shape
        
        # Patch embedding
        x = self.patch_embed(x)  # [B, num_patches, embed_dim]

        # 添加分类令牌
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed

        # 通过 Transformer 编码层
        x = self.blocks(x)
        x = self.norm(x)

        # 移除分类令牌，只保留patch tokens
        patch_tokens = x[:, 1:]  # [B, num_patches, embed_dim]
        
        # 重塑为空间格式 [B, embed_dim, patch_h, patch_w]
        patch_tokens = patch_tokens.transpose(1, 2).reshape(B, -1, self.patch_h, self.patch_w)
        
        # 通过分割头
        seg_map = self.segmentation_head(patch_tokens)
        
        # 上采样到原始分辨率
        seg_map = self.upsample(seg_map)
        
        return seg_map