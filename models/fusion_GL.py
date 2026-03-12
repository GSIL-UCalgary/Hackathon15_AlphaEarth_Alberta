# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
import pdb

import numpy as np
from functools import partial
#from HR_Mamba import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import PatchEmbed, Block
#from ResNet import resnet18, resnet34, resnet50
from .branch1 import DipResNet2Layers
from .feature_fusion import FeatureFusion
# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------


# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------

# --- Semantic Segmentation Decoder ---
# Helper Block for the Decoder

class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization (RMSNorm) layer.
    
    RMSNorm is a simplified normalization technique that normalizes inputs using 
    the root mean square statistic instead of mean and variance like LayerNorm.
    
    The normalization is computed as:
        y = (x / RMS(x)) * weight
    where RMS(x) = sqrt(mean(x^2) + epsilon)
    
    Advantages over LayerNorm:
    - Computationally cheaper (no mean subtraction)
    - More stable for certain activations (e.g., ReLU)
    - Often performs comparably to LayerNorm in practice
    
    Args:
        dim (int): Dimensionality of the input features to normalize.
        eps (float, optional): Small epsilon value for numerical stability.
                               Prevents division by zero. Default: 1e-6.
    
    Attributes:
        eps (float): Epsilon value for numerical stability.
        weight (nn.Parameter): Learnable scaling parameter of shape (dim,).
    
    Note:
        - Unlike LayerNorm, RMSNorm does not have a bias term.
        - Normalization is applied along the last dimension.
        - The weight parameter is initialized to ones.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        """
        Forward pass for RMS normalization.
        
        Args:
            x (torch.Tensor): Input tensor of shape (*, dim).
        
        Returns:
            torch.Tensor: Normalized tensor of same shape as input.
            
        Note:
            - Computes RMS along the last dimension.
            - Applies learnable scaling after normalization.
            - The operation is differentiable.
        """
        # Calculate the root mean square for each sample
        # rsqrt computes 1/sqrt(value) for efficiency
        norm = (x.pow(2).mean(-1, keepdim=True) + self.eps).rsqrt()
        return x * (self.weight * norm)
    

def batched_index_select(input, dim, index):
    for ii in range(1, len(input.shape)):
        if ii != dim:
            index = index.unsqueeze(ii)
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(input, dim, index)

class SparseDeformableMambaBlock(nn.Module):
    def __init__(self, dim, d_state=32, d_conv=8, expand=2, sparsity_ratio=0.3):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.expand = expand
        self.expanded_dim = dim * expand
        self.sparsity_ratio = sparsity_ratio

        self.norm = RMSNorm(dim)
        self.proj_in = nn.Linear(dim, self.expanded_dim)
        self.proj_out = nn.Linear(self.expanded_dim, dim)

        # self.A = nn.Parameter(self._build_controllable_matrix(d_state))
        self.A = nn.Parameter(torch.zeros(d_state, d_state))

        self.B = nn.Parameter(torch.zeros(1, 1, d_state))
        self.C = nn.Parameter(torch.zeros(self.expanded_dim, d_state))

        self.conv = nn.Conv1d(
            in_channels=self.expanded_dim,
            out_channels=self.expanded_dim,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.expanded_dim,
            bias=False
        )

    def _build_controllable_matrix(self, n):
        A = torch.zeros(n, n)
        for i in range(n - 1):
            A[i, i + 1] = 1.0
        A[-1, :] = torch.randn(n) * 0.02
        return A

    def forward(self, x):
        B, L, C = x.shape
        # L = H * W
        residual = x

        # Flatten spatial dimensions
        x_flat = x  # x.reshape(B, L, C)

        # Normalize and project
        x_norm = self.norm(x_flat)
        x_proj = self.proj_in(x_norm)  # [B, L, expanded_dim]

        # Token selection
        center_idx = L // 2
        center = x_proj[:, center_idx:center_idx + 1, :]

        x_proj_norm = F.normalize(x_proj, p=2, dim=-1)  # [B, L, D]
        center_norm = F.normalize(center, p=2, dim=-1)  # [B, 1, D]
        sim = torch.matmul(x_proj_norm, center_norm.transpose(-1, -2)).squeeze(-1)
        # im = torch.matmul(x_proj, center.transpose(-1, -2)).squeeze(-1)  # [B, L]
        sim = torch.softmax(sim, dim=-1)  # Normalized probabilities

        k = max(1, int(L * self.sparsity_ratio))
        _, topk_idx = torch.topk(sim, k=k, dim=-1)

        x_sparse = batched_index_select(x_proj, 1, topk_idx)  # [B, k, expanded_dim]

        # Conv processing
        x_conv = x_sparse.transpose(1, 2)
        x_conv = self.conv(x_conv)[..., :k]
        x_conv = x_conv.transpose(1, 2)

        # SSM processing
        h = torch.zeros(B, self.expanded_dim, self.d_state, device=x.device)
        outputs = []

        for t in range(k):
            x_t = x_conv[:, t].unsqueeze(-1)
            Bx = torch.sigmoid(self.B.to(x.device)) * x_t
            h = torch.matmul(h, self.A.to(x.device).T) + Bx
            out_t = (h * torch.sigmoid(self.C.to(x.device).unsqueeze(0))).sum(-1)
            outputs.append(out_t)

        x_processed = torch.stack(outputs, dim=1)
        x_processed = self.proj_out(x_processed)

        # Combine with residual
        # x_processed = x_processed + batched_index_select(residual.reshape(B, L, C), 1, topk_idx)

        # Scatter back to original positions
        output = torch.zeros(B, L, C, device=x.device)
        output.scatter_(1, topk_idx.unsqueeze(-1).expand(-1, -1, C), x_processed)

        # return output.reshape(B, H, W, C) + x

        return output + x

class SimplifiedMambaBlock(nn.Module):
    def __init__(self, dim: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.expand = expand
        self.expanded_dim = dim * expand

        #self.norm = DyT(dim)
        self.norm = RMSNorm(dim)
        self.proj_in = nn.Linear(dim, self.expanded_dim)
        self.proj_out = nn.Linear(self.expanded_dim, dim)

        # SSM parameters
        self.A = nn.Parameter(torch.zeros(self.expanded_dim, d_state))
        self.B = nn.Parameter(torch.zeros(self.expanded_dim, d_state))
        self.C = nn.Parameter(torch.zeros(self.expanded_dim, d_state))

        # Convolution layer
        self.conv = nn.Conv1d(
            in_channels=self.expanded_dim,
            out_channels=self.expanded_dim,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.expanded_dim,
            bias=False
        )

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.proj_in(x)

        # Conv branch
        x_conv = x.transpose(1, 2)
        x_conv = self.conv(x_conv)[..., :x.shape[1]]
        x_conv = x_conv.transpose(1, 2)

        # SSM branch
        batch_size, seq_len, _ = x.shape
        h = torch.zeros(batch_size, self.expanded_dim, self.d_state, device=x.device)
        outputs = []

        for t in range(seq_len):
            x_t = x_conv[:, t].unsqueeze(-1)
            Bx = torch.sigmoid(self.B) * x_t
            h = torch.sigmoid(self.A.unsqueeze(0)) * h + Bx
            out_t = (h * torch.sigmoid(self.C.unsqueeze(0))).sum(-1)
            outputs.append(out_t)

        x = torch.stack(outputs, dim=1)
        x = self.proj_out(x)
        return x + residual

class Global_superxiel_model(nn.Module):
    def __init__(self, num_classes, num_superpixel, dim, d_conv, in_channel, total_pixel=128*128):
        super().__init__()

        self.in_channel = in_channel
        self.num_classes = num_classes

        self.num_superpixel = num_superpixel

        self.dim = dim
        self.d_conv = d_conv
        self.total_pixel = total_pixel

        self.stem = DipResNet2Layers(
        num_input_channels=self.in_channel,
        num_output_channels=self.num_classes,
        num_channels=self.dim,
        act_fun="LeakyReLU",
        pad="reflection"
            )

        self.stem2 = DipResNet2Layers(
        num_input_channels=self.dim,
        num_output_channels=self.num_classes,
        num_channels=self.dim,
        act_fun="LeakyReLU",
        pad="reflection"
            )

        self.global_mamba = \
            nn.ModuleList([SimplifiedMambaBlock(dim=self.dim, d_conv=self.d_conv) for i in range(4)])


        self.fuse_GL = FeatureFusion(
            p=self.dim,
            spatial_dim=self.num_superpixel,
            spectral_dim=self.total_pixel,
            d_model=256,
            nhead=8,
            num_layers=2,
            dropout=0.1
        )

        self.global_seg = nn.Linear(self.dim, self.num_classes)

        self.Gl_seg = DipResNet2Layers(
        num_input_channels=self.dim,
        num_output_channels=self.num_classes,
        num_channels=self.dim,
        act_fun="LeakyReLU",
        pad="reflection"
            )
        #
        self.Lg_seg = DipResNet2Layers(
        num_input_channels=self.dim,
        num_output_channels=self.num_classes,
        num_channels=self.dim,
        act_fun="LeakyReLU",
        pad="reflection"
            )

    def forward(self, x, segments, assignment_matrix_, target_num=600): #target_num= num of superpixels
        
        B, C, H, W = x.shape
        features, _ = self.stem(x)
        Local_feat, local_map = self.stem2(features)
        num_channels = x.shape[1]# B, 256, 128, 128

        batch_size, num_channels, h, w = features.shape

        # 展平特征和标签
        features_flat = features.permute(0, 2, 3, 1).reshape(batch_size, -1, num_channels)  # [b, h*w, 256]
        segments_flat = segments.reshape(batch_size, -1)  # [b, h*w]

        assert segments.max() <= target_num
        padded_results = torch.zeros(batch_size, target_num, num_channels,
                                     device=features.device, dtype=features.dtype)

        for b in range(batch_size):
            # 获取当前batch的超像素ID和数量
            unique_ids = torch.unique(segments_flat[b])
            actual_num = len(unique_ids)

            # 计算实际存在的超像素特征
            masks = (segments_flat[b].unsqueeze(1) == unique_ids.unsqueeze(0))  # [h*w, actual_num]
            sum_features = torch.matmul(masks.T.float(), features_flat[b])  # [actual_num, 256]
            pixel_counts = masks.sum(dim=0).float()  # [actual_num]
            mean_features = sum_features / (pixel_counts.unsqueeze(1) + 1e-7)  # [actual_num, 256]
            padded_results[b, :actual_num] = mean_features
            # # 填充到target_num
            # if actual_num >= target_num:
            #     padded_results[b] = mean_features[:target_num]  # 截断超出的部分
            # else:
            #     padded_results[b, :actual_num] = mean_features  # 不足部分保持为0

        for blk in self.global_mamba:
            padded_results = blk(padded_results)

        mamba_output = padded_results
        batch_size, num_sp, feat_dim = mamba_output.shape
        device = mamba_output.device

        # 获取每个batch的实际超像素数量
        actual_nums = [len(torch.unique(segments[b])) for b in range(batch_size)]

        # 创建扩展后的segments用于索引 [batch, 128, 128, 1]
        segments_expanded = segments.unsqueeze(-1)  # [batch, 128, 128, 1]

        # 创建输出张量

        #
        global_map = self.global_seg(mamba_output)

        remap_output = torch.zeros(batch_size, self.num_classes, *segments.shape[1:], device=device)

        for b in range(batch_size):
            # 获取有效特征 [actual_num, 256]
            valid_features = global_map[b, :actual_nums[b]]

            # 创建查找表 [max_sp_id + 1, 256]
            max_id = segments[b].max().item()
            lookup_table = torch.zeros(max_id+1, self.num_classes, device=device)
            unique_ids = torch.unique(segments[b])
            lookup_table[unique_ids] = valid_features[:len(unique_ids)]

            # 向量化映射
            remap_output[b] = lookup_table[segments[b]].permute(2, 0, 1)  # [256, 128, 128]



        G = mamba_output   # [batchsize, num_superpixel, self.dim]
        L = Local_feat.permute(0, 2, 3, 1).reshape(batch_size, -1, self.dim)
        Gl, Lg = self.fuse_GL(G, L, assignment_matrix_)

        # 创建输出张量
        Gloutput = torch.zeros(batch_size, feat_dim, *segments.shape[1:], device=device)

        for b in range(batch_size):
            # 获取有效特征 [actual_num, 256]
            valid_features = Gl[b, :actual_nums[b]]

            # 创建查找表 [max_sp_id + 1, 256]
            max_id = segments[b].max().item()
            lookup_table = torch.zeros(max_id+1, feat_dim, device=device)
            unique_ids = torch.unique(segments[b])
            lookup_table[unique_ids] = valid_features[:len(unique_ids)]

            # 向量化映射
            Gloutput[b] = lookup_table[segments[b]].permute(2, 0, 1)  # [256, 128, 128]

        Lg = Lg.permute(0, 2, 1).reshape(batch_size, self.dim, H, W)
        Gl_map = self.Gl_seg(Gloutput)[1]
        Lg_map = self.Lg_seg(Lg)[1]

        voting_map = local_map+remap_output#+Gl_map+Lg_map

        '''
            G = torch.randn(b, config['total_pixel'], p)  # num super pixels
            L = torch.randn(b, config['num_superpixel'], p) # total pixels
        '''


        # return local_map, global_map, Gl_map, Lg_map, voting_map # Original
        return voting_map

