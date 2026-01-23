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


def convnext_segmentation(num_classes=13):
    """专为语义分割设计的轻量版"""
    return ConvNeXtForSegmentation(
        in_chans=10,
        depths=[2, 2],
        dims=[128, 128],
        num_classes=num_classes,
        patch_size=1
    )