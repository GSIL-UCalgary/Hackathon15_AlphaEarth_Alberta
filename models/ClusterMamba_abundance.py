import math
import pdb
import warnings

import matplotlib.pyplot as plt
import torch
from torch import nn
from mamba_ssm import Mamba
import torch.nn.functional as F
import numpy as np

def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = x.norm(2, dim=-1, keepdim=True)
        rms_x = norm_x * (x.shape[-1] ** -0.5)
        return self.weight * (x / (rms_x + self.eps))


def batched_index_select(input, dim, index):
    """Select indices along specified dimension in batched manner"""
    for ii in range(1, len(input.shape)):
        if ii != dim:
            index = index.unsqueeze(ii)
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(input, dim, index)


class AttentionSelector(nn.Module):
    """Attention-based importance scoring for pixel selection"""
    
    def __init__(self, feature_dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        assert feature_dim % num_heads == 0, "feature_dim must be divisible by num_heads"
        
        # Multi-head attention components
        self.q_proj = nn.Linear(feature_dim, feature_dim)
        self.k_proj = nn.Linear(feature_dim, feature_dim)
        self.v_proj = nn.Linear(feature_dim, feature_dim)
        self.out_proj = nn.Linear(feature_dim, 1)  # Project to importance score
        
        self.scale = self.head_dim ** -0.5
        
    def forward(self, x):
        """
        Args:
            x: [B, L, D] features
        Returns:
            importance_scores: [B, L] attention-based importance scores
        """
        B, L, D = x.shape
        
        # Multi-head projections
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, L, d]
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, L, d]
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, L, d]
        
        # Compute attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, L, L]
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attn, v)  # [B, H, L, d]
        out = out.transpose(1, 2).contiguous().view(B, L, D)  # [B, L, D]
        
        # Project to importance scores
        importance_scores = self.out_proj(out).squeeze(-1)  # [B, L]
        
        return importance_scores


class ClusterHead(nn.Module):
    """K-means clustering head for semantic segmentation"""

    def __init__(self, feature_dim, num_clusters, temperature=1.0, ema_momentum=0.9,  learn_var=True):
        super().__init__()
        self.num_clusters = num_clusters
        self.temperature = temperature
        self.ema_momentum = ema_momentum
        #self.cluster_centers = nn.Parameter(torch.randn(num_clusters, feature_dim))
        # self.register_buffer(
        #     "cluster_centers",
        #     torch.randn(num_clusters, feature_dim)
        # )
        #self.center_labels = torch.arange(9, device=self.cluster_centers.device).repeat_interleave(3)

        #self.means = nn.Parameter(torch.randn(num_clusters, feature_dim))

        #if learn_var:
        #    self.log_vars = nn.Parameter(torch.zeros(num_clusters))  # log variance
        #else:
        #    self.register_buffer("log_vars", torch.zeros(num_clusters))

    def forward(self, features, cluster_centers, log_vars, alpha=1.0):
    #     # # features: [B, L, D], centers: [K, D]
    #     # features_norm = F.normalize(features, p=2, dim=-1)
    #     # centers_norm = F.normalize(self.cluster_centers, p=2, dim=-1)
    #     #
    #     # # squared distance
    #     # dist_sq = torch.cdist(features_norm, centers_norm, p=2) ** 2  # [B, L, K]
    #     #
    #     # # Student's t-distribution
    #     # q = 1.0 / (1.0 + dist_sq / alpha)
    #     # q = q ** ((alpha + 1.0) / 2.0)
    #     #
    #     # # Normalize to probabilities
    #     # q = q / q.sum(dim=-1, keepdim=True)
    #     # return
        """
        features: [B, L, D]
        returns: soft assignments [B, L, K]
        # """
        B, L, D = features.shape
        # [B, L, 1, D] - [1, 1, K, D] = [B, L, K, D]
        diff = features.unsqueeze(2) - cluster_centers.unsqueeze(0)
        dist2 = diff.pow(2).sum(-1)  # [B, L, K]

        # variance per cluster
        var = torch.exp(log_vars).permute(0, 2, 1) # [1,1,K]

        # log likelihood (ignoring constants)
        logits = - dist2 / (2 * var)

        # soft assignments
        probs = F.softmax(logits, dim=-1)
        return probs, logits
    #
    # def forward(self, features):
    #     """
    #     Args:
    #         features: [B, L, D] where L = H * W (spatial positions), D = feature dim
    #     Returns:
    #         cluster_assignments: [B, L, K] soft cluster membership probabilities
    #     """
    #     B, L, D = features.shape
    #
    #     # Normalize features and cluster centers
    #     features_norm = F.normalize(features, p=2, dim=-1)  # [B, L, D]
    #     centers_norm = F.normalize(self.cluster_centers, p=2, dim=-1)  # [K, D]
    #
    #     # Compute similarity between features and cluster centers
    #     similarity = torch.matmul(features_norm, centers_norm.t())  # [B, L, K]
    #
    #     # Convert to soft cluster assignments using temperature scaling
    #     cluster_assignments = F.softmax(similarity / self.temperature, dim=-1)
    #
    #     return cluster_assignments

    def update_centers(self, features, labels):
        """
        features: (1, C, H, W)
        labels:   (1, H, W), values in {-1,...,8}
        """
        B, C, H, W = features.shape
        features = features.permute(0, 2, 3, 1).reshape(-1, C)  # (N, C)
        labels = labels.view(-1)  # (N,)

        class_means = []
        for cls in range(self.num_clusters):
            mask = labels == cls
            if mask.any():
                mean_feat = features[mask].mean(dim=0)
            else:
                mean_feat = torch.zeros(C, device=features.device)
            class_means.append(mean_feat)
        class_means = torch.stack(class_means, dim=0)

        # EMA update
        with torch.no_grad():
            self.cluster_centers.data = (
                self.ema_momentum * self.cluster_centers.data
                + (1 - self.ema_momentum) * class_means
            )

    def contrastive_center_loss(self, margin=0.2):
        """
        Encourage cluster centers to be far apart.
        Args:
            margin: minimal cosine distance between centers
        Returns:
            loss (scalar tensor)
        """
        # Normalize centers
        centers = self.cluster_centers  # [K, D]
        labels = self.center_labels  # [K]

        # Normalize centers for cosine similarity
        centers_norm = F.normalize(centers, p=2, dim=-1)  # [K, D]

        # Pairwise cosine similarity [K, K]
        sim_matrix = torch.matmul(centers_norm, centers_norm.t())  # [K, K]

        # Build label mask
        same_class_mask = labels.unsqueeze(0) == labels.unsqueeze(1)  # [K, K]
        same_class_mask = same_class_mask.to(centers.device)
        diff_class_mask = ~same_class_mask

        # Exclude self-similarity
        eye_mask = torch.eye(len(labels), device=centers.device).bool()
        same_class_mask = same_class_mask & ~eye_mask
        diff_class_mask = diff_class_mask & ~eye_mask

        # ----- Positive term: same-class centers should be close -----
        # We penalize (1 - sim), i.e., encourage high cosine similarity
        if same_class_mask.sum() > 0:
            pos_loss = (1 - sim_matrix[same_class_mask]).mean()
        else:
            pos_loss = torch.tensor(0.0, device=centers.device)

        # ----- Negative term: different-class centers should be apart -----
        # Penalize if similarity > (1 - margin)
        if diff_class_mask.sum() > 0:
            neg_loss = F.relu(sim_matrix[diff_class_mask] - (1 - margin)).mean()
        else:
            neg_loss = torch.tensor(0.0, device=centers.device)

        # Combine with equal weight (can be tuned)
        loss = pos_loss + neg_loss

        return loss


class SimplifiedAttentionSelector(nn.Module):
    """Simplified attention without full multi-head - much lower memory"""

    def __init__(self, feature_dim, reduction_ratio=4):
        super().__init__()
        reduced_dim = feature_dim // reduction_ratio

        # Simple attention scoring
        self.query = nn.Linear(feature_dim, reduced_dim)
        self.key = nn.Linear(feature_dim, reduced_dim)
        self.scale = reduced_dim ** -0.5

    def forward(self, x):
        """
        Args:
            x: [B, L, D]
        Returns:
            importance_scores: [B, L]
        """
        B, L, D = x.shape

        q = self.query(x)  # [B, L, reduced_dim]
        k = self.key(x)  # [B, L, reduced_dim]

        # Compute self-attention scores (how much each position attends to itself)
        # This is much simpler than full attention
        importance_scores = torch.sum(q * k, dim=-1) * self.scale  # [B, L]

        return importance_scores



class AttentionSparseDeformableMambaBlock(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, num_clusters=100,
                 sparsity_ratio=0.5, use_attention=True, num_heads=4, 
                 selection_mode='hybrid'):
        """
        Args:
            selection_mode: 'attention', 'cluster', or 'hybrid'
                - 'attention': pure attention-based selection
                - 'cluster': original cluster-based selection
                - 'hybrid': combine both for diversity + importance
        """
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.expand = expand
        self.expanded_dim = dim * expand
        self.num_clusters = num_clusters
        self.sparsity_ratio = sparsity_ratio
        self.use_attention = use_attention
        self.selection_mode = selection_mode

        self.norm = RMSNorm(dim)
        self.proj_in = nn.Linear(dim, self.expanded_dim)

        self.norm2 = RMSNorm(dim)
        self.proj_in1 = nn.Linear(dim, self.expanded_dim)
        self.proj_out = nn.Linear(self.expanded_dim, dim)

        self.proj_out_blocks = nn.ModuleList([
            nn.Linear(self.expanded_dim, dim)
            for _ in range(num_clusters)
        ])

        self.mu_proj = nn.Linear(self.expanded_dim, self.expanded_dim)
        self.logvar_proj = nn.Linear(self.expanded_dim, 1)

        # Clustering for semantic segmentation
        self.cluster_head = ClusterHead(self.expanded_dim, num_clusters)
        
        # Attention-based importance scoring
        if use_attention:
            self.attention_selector = AttentionSelector(self.expanded_dim, num_heads)
            
            # Learnable weight for combining attention and clustering
            if selection_mode == 'hybrid':
                self.selection_weight = nn.Parameter(torch.tensor(0.5))

        # # Convolution layer
        # self.conv = nn.Conv1d(
        #     in_channels=self.expanded_dim,
        #     out_channels=self.expanded_dim,
        #     kernel_size=d_conv,
        #     padding=d_conv - 1,
        #     groups=self.expanded_dim,
        #     bias=False
        # )
        self.conv_blocks = nn.ModuleList([
            nn.Conv1d(
                in_channels=self.expanded_dim,
                out_channels=self.expanded_dim,
                kernel_size=d_conv,
                padding=d_conv - 1,
                groups=self.expanded_dim,
                bias=False
            ) for _ in range(num_clusters)
        ])
        # # Mamba SSM
        self.global_mamba = Mamba(
            d_model=dim*2,
            d_state=16,
            d_conv=4,
            expand=2,
        )
        self.attention_selector = SimplifiedAttentionSelector(
            self.expanded_dim,
            reduction_ratio=4
        )
        self.mamba_blocks = nn.ModuleList([
            Mamba(
                d_model=dim * 2,
                d_state=16,
                d_conv=4,
                expand=2,
            ) for _ in range(num_clusters)
        ])

    def compute_cluster_loss(self, features, cluster_assignments):
        """Compute clustering loss for semantic segmentation"""
        B, L, D = features.shape
        K = self.num_clusters

        # Normalize features
        features_norm = F.normalize(features, p=2, dim=-1)

        # Compute cluster centers from assignments
        cluster_weights = cluster_assignments.sum(dim=1)  # [B, K]
        weighted_features = torch.matmul(cluster_assignments.transpose(1, 2), features_norm)  # [B, K, D]

        # Avoid division by zero
        cluster_weights = cluster_weights + 1e-8
        cluster_centers = weighted_features / cluster_weights.unsqueeze(-1)  # [B, K, D]
        #cluster_centers = self.cluster_head.cluster_centers.unsqueeze(0)

        # Within-cluster variance
        expanded_centers = cluster_centers.unsqueeze(1)  # [B, 1, K, D]
        expanded_features = features_norm.unsqueeze(2)  # [B, L, 1, D]

        # Compute distances to cluster centers
        distances = torch.sum((expanded_features - expanded_centers) ** 2, dim=-1)  # [B, L, K]

        # Weighted within-cluster variance
        within_cluster_var = torch.sum(cluster_assignments * distances) / (B * L)

        # Between-cluster variance
        centers_norm = F.normalize(cluster_centers, p=2, dim=-1)
        centers_similarity = torch.matmul(centers_norm, centers_norm.transpose(1, 2))  # [B, K, K]

        # Maximize distance between different clusters
        mask = 1 - torch.eye(K, device=features.device).unsqueeze(0)  # [1, K, K]
        between_cluster_sim = torch.sum(centers_similarity * mask) / (B * K * (K - 1))

        # Total clustering loss
        cluster_loss = within_cluster_var + between_cluster_sim

        return cluster_loss

    def select_pixels(self, x_proj, cluster_assignments, k_total, per_cluster):
        """Select top-k pixels based on selection mode"""
        B, L, C = x_proj.shape
        CC = cluster_assignments.shape[1]
        cluster_assignments = cluster_assignments.permute(0, 2, 3, 1).reshape(B, L, CC)  # [B, K, L]
        #attention_scores = self.attention_selector(x_proj)  # Compute here
        if self.selection_mode == 'cluster':
            # Original cluster-based selection
            #k_per_cluster = max(1, int(k_total / self.num_clusters * self.sparsity_ratio))
            selected_indices = []
            for cluster_idx in range(len(per_cluster)):
                cluster_scores = cluster_assignments[:, :, cluster_idx]# 141 141 100
                # cluster_scores = cluster_scores + attention_scores
                # 13 * 128
                # 13 * 2
                topk_scores, topk_indices = torch.topk(cluster_scores,k=per_cluster[cluster_idx],dim=-1,largest=True,sorted=True)
                selected_indices.append(topk_indices)
            #pdb.set_trace()
            #pdb.set_trace()ß
            #_, topk_indices = torch.topk(cluster_assignments, k=self.num_clusters, dim=-1)
            #selected_indices = topk_indices.reshape(1, -1)
            # selected_indices = torch.cat(selected_indices, dim=0)
            
            # Trim if necessary
            # if selected_indices.size(-1) > k_total:
            #     importance_scores = torch.gather(
            #         cluster_assignments.max(dim=-1)[0],
            #         1, selected_indices
            #     )
            #     _, top_importance_indices = torch.topk(importance_scores, k=k_total, dim=-1)
            #     selected_indices = torch.gather(selected_indices, 1, top_importance_indices)
                
        elif self.selection_mode == 'attention':
            # Pure attention-based selection
            attention_scores = self.attention_selector(x_proj)  # [B, L]
            _, selected_indices = torch.topk(attention_scores, k=k_total, dim=-1)
            
        elif self.selection_mode == 'hybrid':
            # Hybrid: combine clustering diversity with attention importance
            attention_scores = self.attention_selector(x_proj)  # [B, L]
            cluster_scores = cluster_assignments.max(dim=-1)[0]  # [B, L]
            
            # Normalize both scores
            attention_scores = F.softmax(attention_scores, dim=-1)
            cluster_scores = F.softmax(cluster_scores, dim=-1)
            
            # Combine with learned weight
            alpha = torch.sigmoid(self.selection_weight)
            combined_scores = alpha * attention_scores + (1 - alpha) * cluster_scores
            
            _, selected_indices = torch.topk(combined_scores, k=k_total, dim=-1)
        
        return selected_indices

    def compute_cluster_centers(self, x, labels, num_clusters, eps=1e-6):
        """
        x      : (B, C, H, W)
        labels : (H, W)
        return : (B, K, C)
        """
        B = x.shape[0]
        C = x.shape[-1]
        HW = x.shape[1]

        x_flat = x  # (B, HW, C)
        labels_flat = labels.view(HW).long()  # (HW,)
        labels_flat = labels_flat.unsqueeze(0).expand(B, HW)

        centers = torch.zeros(B, len(num_clusters), C, device=x.device)
        counts = torch.zeros(B, len(num_clusters), 1, device=x.device)

        centers.scatter_add_(
            1,
            labels_flat.unsqueeze(-1).expand(-1, -1, C),
            x_flat
        )

        counts.scatter_add_(
            1,
            labels_flat.unsqueeze(-1),
            torch.ones(B, HW, 1, device=x.device)
        )

        centers = centers / (counts + eps)
        return centers


    def forward(self, x, per_cluster, cluster_assignments, return_cluster_assignments=False):
        B, L, C = x.shape
        residual = x
        # Normalize and project
        x_norm = self.norm(x)
        x_proj = self.proj_in(x_norm)  # [B, L, expanded_dim]
        feats_tokens = x_proj  # (B, HW, 256)
        # Get cluster assignments
         # [B, L, K]
        # Determine number of pixels to select
        k_total = max(1, int(L * self.sparsity_ratio))
        # if k_total >= L:
        #     # Process all pixels (no sparsity)
        #     selected_indices = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)
        #     x_sparse = x_proj
        # else:
            # Select pixels based on selection mode
        output = torch.zeros(B, L, C, device=x.device)
        selected_indices = self.select_pixels(x_proj, cluster_assignments, k_total, per_cluster)
        # for i in range(self.num_clusters):
        #     x_sparse = batched_index_select(x_proj, 1, selected_indices[i].unsqueeze(dim=0))
        #
        # # Convolution processing
        #     x_conv = x_sparse.transpose(1, 2)
        #     x_conv = self.conv_blocks[i](x_conv)[..., :x_sparse.size(1)]
        #     x_conv = x_conv.transpose(1, 2)
        #
        # # Mamba processing
        #     x_processed = self.mamba_blocks[i](x_conv)
        #     x_processed = self.proj_out_blocks[i](x_processed)
        #
        # # Scatter back to original positions
        # # if k_total >= L:
        # #     output = x_processed
        # # else:
        #     output += output.scatter_(1, selected_indices[i].unsqueeze(dim=0).unsqueeze(-1).expand(-1, -1, C), x_processed)

        # Gather features for all clusters in parallel
        # Shape: [num_clusters, B, K, C]

        Nc = self.num_clusters
        batched_sparse = []

        # Gather all cluster features in one tensor
        for i in range(len(selected_indices)):
            # [B, k_per_cluster, C]
            xi = batched_index_select(x_proj, 1, selected_indices[i][0].unsqueeze(dim=0))
            batched_sparse.append(xi)

        # Stack → [Nc, B, K, C]

        #x_sparse = torch.stack(batched_sparse, dim=0)

        # Convolution: flatten Nc*B → feed in parallel
        #Nc, B, K, C_exp = x_sparse.shape
        #x_conv = x_sparse.view(Nc * B, K, C_exp).transpose(1, 2)  # [Nc*B, C_exp, K]
        # Apply cluster-specific convs in one pass using cat/split

        x_conv_out = []
        x_conv_out.extend(
            self.conv_blocks[i](batched_sparse[i].transpose(1, 2)).transpose(1, 2)[:, :batched_sparse[i].size(1), :]
            for i in range(len(selected_indices))
        )  # [Nc*B, C_exp, K]
        #x_conv_out = x_conv_out.transpose(1, 2)[:, :x_sparse.size(2), :]

        # Mamba + proj in parallel (still cluster-specific)
        x_processed = []
        x_processed.extend(
            self.proj_out_blocks[i](self.mamba_blocks[i](x_conv_out[i]))
            for i in range(len(selected_indices))
        )
        # Scatter results back into original positions
        for i in range(len(selected_indices)):
            output.scatter_add_(1,selected_indices[i][0].unsqueeze(dim=0).unsqueeze(-1).expand(-1, -1, C),x_processed[i])
        output = output + residual
        output = self.proj_in1(self.norm2(output))
        output = self.global_mamba(output)
        output = self.proj_out(output)
        if return_cluster_assignments:
            #cluster_loss = self.compute_cluster_loss(x_proj, cluster_assignments)
            #cluster_loss = self.cluster_head.contrastive_center_loss()
            return output+residual
        else:
            return output + residual


# class SpeMamba(nn.Module):
#     def __init__(self, channels, token_num=8, use_residual=True, group_num=4):
#         super(SpeMamba, self).__init__()
#         self.token_num = token_num
#         self.use_residual = use_residual

#         self.group_channel_num = math.ceil(channels/token_num)
#         self.channel_num = self.token_num * self.group_channel_num

#         self.mamba = Mamba(
#             d_model=self.group_channel_num,
#             d_state=16,
#             d_conv=4,
#             expand=2,
#         )

#         self.proj = nn.Sequential(
#             nn.GroupNorm(group_num, self.channel_num),
#             nn.SiLU()
#         )

#     def build_cluster_tokens(self, x, cluster_map, K):
#         """
#         x: (B, C, H, W)
#         cluster_map: (B, H, W) with labels in [0, K-1]
#         """
#         B, C, H, W = x.shape
#         x_flat = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
#         c_flat = cluster_map.reshape(B, H * W)

#         all_tokens = []

#         for b in range(B):
#             tokens_b = []
#             for k in range(K):
#                 mask = (c_flat[b] == k)
#                 pixels = x_flat[b][mask]  # (Nk, C)
#                 tokens_b.append(pixels)
#             all_tokens.append(tokens_b)

#         return all_tokens  # List[B][K] of (Nk, C)

#     def padding_feature(self, x):
#         B, C, H, W = x.shape
#         if C < self.channel_num:
#             pad_c = self.channel_num - C
#             pad_features = torch.zeros((B, pad_c, H, W)).to(x.device)
#             cat_features = torch.cat([x, pad_features], dim=1)
#             return cat_features
#         else:
#             return x

#     def forward(self, x, cluster_map):
#         """
#         x           : (B, C, H, W)
#         cluster_map : (H, W)   # 2D spatial cluster labels (shared across batch)
#         """

#         # -------------------------------------------------
#         # 1. Prepare features
#         # -------------------------------------------------
#         x_pad = self.padding_feature(x)  # (B, C_pad, H, W)
#         x_pad = x_pad.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C_pad)

#         B, H, W, C_pad = x_pad.shape
#         T = self.token_num
#         G = self.group_channel_num
#         assert C_pad == T * G

#         # -------------------------------------------------
#         # 2. Flatten pixels
#         # -------------------------------------------------
#         HW = H * W
#         x_flat = x_pad.view(B * HW, C_pad)  # (BHW, C_pad)

#         # IMPORTANT: cluster_map is (H, W), no batch
#         cluster_flat = cluster_map.view(HW).long()  # (HW,)
#         cluster_flat = cluster_flat.repeat(B)  # (BHW,)

#         # -------------------------------------------------
#         # 3. Sort by cluster id (FAST)
#         # -------------------------------------------------
#         sorted_cluster, perm = torch.sort(cluster_flat)
#         x_sorted = x_flat[perm]

#         # -------------------------------------------------
#         # 4. Split into cluster blocks
#         # -------------------------------------------------
#         _, counts = torch.unique_consecutive(
#             sorted_cluster, return_counts=True
#         )
#         chunks = torch.split(x_sorted, counts.tolist())

#         # -------------------------------------------------
#         # 5. Reshape for Mamba
#         # -------------------------------------------------
#         batched_sparse = torch.cat(
#             [c.view(-1, T, G) for c in chunks],
#             dim=0
#         )  # (BHW, T, G)

#         # -------------------------------------------------
#         # 6. Mamba
#         # -------------------------------------------------
#         batched_out = self.mamba(batched_sparse)  # (BHW, T, G)
#         batched_out = batched_out.view(B * HW, C_pad)

#         # -------------------------------------------------
#         # 7. Scatter back
#         # -------------------------------------------------
#         out_flat = torch.zeros_like(x_flat)
#         out_flat[perm] = batched_out

#         # -------------------------------------------------
#         # 8. Recover spatial shape
#         # -------------------------------------------------
#         out = out_flat.view(B, H, W, C_pad)
#         out = out.permute(0, 3, 1, 2).contiguous()  # (B, C_pad, H, W)

#         out = self.proj(out)
#         return x + out if self.use_residual else out


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
    def __init__(self, channels, use_residual=True, group_num=4, use_proj=True,
                 num_clusters=19, sparsity_ratio=1.0, use_attention=True,
                 num_heads=4, selection_mode='hybrid'):
        super(SpaMamba, self).__init__()
        self.use_residual = use_residual
        self.use_proj = use_proj

        self.mamba = AttentionSparseDeformableMambaBlock(
            dim=channels,
            num_clusters=num_clusters,
            sparsity_ratio=sparsity_ratio,
            use_attention=use_attention,
            num_heads=num_heads,
            selection_mode=selection_mode
        )
        
        if self.use_proj:
            self.proj = nn.Sequential(
                nn.GroupNorm(group_num, channels),
                nn.SiLU()
            )

    def forward(self, x, per_cluster, cluster_map):
        x_re = x.permute(0, 2, 3, 1).contiguous()
        B, H, W, C = x_re.shape
        x_flat = x_re.view(1, -1, C)
        x_flat = self.mamba(x_flat, per_cluster, cluster_map, return_cluster_assignments=True)

        x_recon = x_flat.view(B, H, W, C)
        x_recon = x_recon.permute(0, 3, 1, 2).contiguous()
        if self.use_proj:
            x_recon = self.proj(x_recon)
        if self.use_residual:
            return x_recon + x
        else:
            return x_recon


class BothMamba(nn.Module):
    def __init__(self, channels, token_num, use_residual, group_num=4, use_att=True,
                 num_clusters=19, sparsity_ratio=1.0, attention_heads=4, original_image_bands = 10,
                 selection_mode='hybrid'):
        # channels is hidden_dim
        super(BothMamba, self).__init__()
        self.use_att = use_att
        self.use_residual = use_residual
        self.original_image_bands = original_image_bands
        self.num_queries_times = 30
        self.num_clusters = num_clusters
        if self.use_att:
            self.fusion_weights = nn.Parameter(torch.ones(2) / 2)
            self.softmax = nn.Softmax(dim=0)
            

        self.spa_mamba = SpaMamba(
            channels, 
            use_residual=use_residual, 
            group_num=group_num,
            num_clusters=num_clusters,
            sparsity_ratio=sparsity_ratio,
            use_attention=True,
            num_heads=attention_heads,
            selection_mode=selection_mode
        )
        self.cls_head = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=128, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(group_num, 128),
            nn.SiLU(),
            nn.Conv2d(in_channels=128, out_channels=num_clusters, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_clusters),
            SumToOne(scale=3.5),
        )
        self.spe_mamba = SpeMamba(channels, token_num=token_num, use_residual=use_residual, group_num=group_num)
        self.num_queries = self.num_queries_times * num_clusters
        self.query_embed = nn.Embedding(self.num_queries, self.original_image_bands)
        self.weights = nn.Parameter(torch.ones((num_clusters, self.num_queries_times)))
        

    def forward(self, x, per_cluster, cluster_map):
        spa_x = self.spa_mamba(x, per_cluster, cluster_map)
        spa_x = self.spa_mamba(spa_x, per_cluster, cluster_map)
        spe_x = self.spe_mamba(x)
        spe_x = self.spe_mamba(spe_x)
        weights = self.softmax(self.fusion_weights)
        fusion_x = spa_x * weights[0] + spe_x * weights[1]
    
        # x1 = self.spa_mamba(x, per_cluster, cluster_map)
        
        # x1 = self.spa_mamba(x1, per_cluster, cluster_map)
        
        # x1 = self.spe_mamba(x1)
        # x1 = self.spe_mamba(x1)
        fusion_x = fusion_x + x        
        abu = self.cls_head(fusion_x)
        endm_get = self.get_endmember()
        recon_linear = torch.einsum('brhw,rl->blhw', [abu, endm_get])
        
        return recon_linear, abu, endm_get, fusion_x

    def get_endmember(self):
        query_embed_weight_split = torch.chunk(self.query_embed.weight, self.num_clusters, dim=0)
        query_embed_weight_split = torch.stack(query_embed_weight_split)
        endmember_get = self.weights.unsqueeze(-1).repeat(1, 1, self.original_image_bands) * query_embed_weight_split
        endmember_get = torch.mean(endmember_get, dim=1)
        return endmember_get


class SpaMamba_init(nn.Module):
    def __init__(self, channels, use_residual=True, use_proj=True):
        super(SpaMamba_init, self).__init__()
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
                nn.Linear(channels, channels),
                nn.LayerNorm(channels),
                nn.SiLU()
            )

    def forward(self, x):
        x_re = x.permute(0, 2, 3, 1).contiguous()
        B, H, W, C = x_re.shape
        x_flat = x_re.view(1, -1, C)
        x_flat = self.mamba(x_flat)

        if self.use_proj:
            x_flat = self.proj(x_flat)

        x_recon = x_flat.view(B, H, W, C)
        x_recon = x_recon.permute(0, 3, 1, 2).contiguous()

        if self.use_residual:
            return x_recon + x
        else:
            return x_recon

class SumToOne(nn.Module):
    def __init__(self, scale=3.5):
        super(SumToOne, self).__init__()
        self.scale = scale

    def forward(self, x):
        x = torch.softmax(self.scale * x, dim=1)
        return x
    
class Initblock(nn.Module):
    def __init__(self, channels, token_num, use_residual, group_num=4, use_att=True,
                 num_clusters=19, sparsity_ratio=1.0, attention_heads=4, original_image_bands= 10,
                 selection_mode='hybrid'):
        # channels is hidden_dim 
        super(Initblock, self).__init__()
        self.use_att = use_att
        self.use_residual = use_residual
        self.num_queries_times = 30
        self.num_clusters = num_clusters    
        self.original_image_bands = original_image_bands    

        if self.use_att:
            self.fusion_weights = nn.Parameter(torch.ones(2) / 2)
            self.softmax = nn.Softmax(dim=0)

        self.spa_mamba = SpaMamba_init(channels, use_residual=use_residual)
        self.spe_mamba = SpeMamba(channels, token_num=token_num, use_residual=use_residual, group_num=group_num)
        self.cls_head = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=128, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(group_num, 128),
            nn.SiLU(),
            nn.Conv2d(in_channels=128, out_channels=num_clusters, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_clusters),
            SumToOne(scale=3.5),
        )
        self.num_queries = self.num_queries_times * num_clusters
        self.query_embed = nn.Embedding(self.num_queries, self.original_image_bands)
        self.weights = nn.Parameter(torch.ones((num_clusters, self.num_queries_times)))

    def forward(self, x):
        
        spa_x = self.spa_mamba(x)
        spa_x = self.spa_mamba(spa_x)
        spe_x = self.spe_mamba(x)
        spe_x = self.spe_mamba(spe_x)
        weights = self.softmax(self.fusion_weights)
        fusion_x = spa_x * weights[0] + spe_x * weights[1]
        fusion_x = fusion_x + x
        abu = self.cls_head(fusion_x)
        endm_get = self.get_endmember()
        recon_linear = torch.einsum('brhw,rl->blhw', [abu, endm_get])
        return recon_linear, abu, endm_get, fusion_x
        
    def get_endmember(self):
        query_embed_weight_split = torch.chunk(self.query_embed.weight, self.num_clusters, dim=0)
        query_embed_weight_split = torch.stack(query_embed_weight_split)
        endmember_get = self.weights.unsqueeze(-1).repeat(1, 1, self.original_image_bands) * query_embed_weight_split
        endmember_get = torch.mean(endmember_get, dim=1)
        return endmember_get
        
        

class MambaHSI(nn.Module):
    def __init__(self, in_channels=128, hidden_dim=64, num_classes=10, 
                 use_residual=True, mamba_type='both', token_num=4, group_num=4,
                 use_att=True, num_clusters=19, sparsity_ratio=1.0,
                 attention_heads=4, selection_mode='cluster'):
        """
        Args:
            selection_mode: 'attention', 'cluster', or 'hybrid' for pixel selection strategy
            attention_heads: number of attention heads for importance scoring
            sparsity_ratio: ratio of pixels to select (1.0 = all pixels)
        """
        super(MambaHSI, self).__init__()
        self.mamba_type = mamba_type
        self.in_channels = in_channels
        self.patch_embedding = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=hidden_dim, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(group_num, hidden_dim),
            nn.SiLU()
        )
        
        self.first_block = Initblock(
                channels=hidden_dim, 
                token_num=token_num, 
                use_residual=use_residual, 
                group_num=group_num, 
                use_att=use_att,
                num_clusters=20,
                sparsity_ratio=sparsity_ratio,
                attention_heads=attention_heads,
                selection_mode=selection_mode,
                original_image_bands = in_channels
            )        
        
        if mamba_type == 'both':
            self.mamba1 = BothMamba(
                channels=hidden_dim, 
                token_num=token_num, 
                use_residual=use_residual, 
                group_num=group_num, 
                use_att=use_att,
                num_clusters=30,
                sparsity_ratio=sparsity_ratio,
                attention_heads=attention_heads,
                original_image_bands = in_channels, 
                selection_mode=selection_mode,
            )
            
            self.mamba2 = BothMamba(
                channels=hidden_dim, 
                token_num=token_num, 
                use_residual=use_residual, 
                group_num=group_num,
                use_att=use_att,
                num_clusters=40,
                sparsity_ratio=sparsity_ratio,
                attention_heads=attention_heads,
                original_image_bands = in_channels,
                selection_mode=selection_mode
            )
            
            self.mamba3 = BothMamba(
                channels=hidden_dim, 
                token_num=token_num, 
                use_residual=use_residual, 
                group_num=group_num, 
                use_att=use_att,
                num_clusters=50,
                sparsity_ratio=sparsity_ratio,
                attention_heads=attention_heads,
                original_image_bands = in_channels,
                selection_mode=selection_mode
            )
            
            self.proj_layer = nn.Conv2d(hidden_dim, hidden_dim * 2, 1)
        self.cls_head = nn.Sequential(
            nn.Conv2d(in_channels=50, out_channels=128, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(group_num, 128),
            nn.SiLU(),
            nn.Conv2d(in_channels=128, out_channels=num_classes, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        H, W = x.shape[2], x.shape[3]   # [145 145 200]
        hsi = x
        if self.training:
            x = self.patch_embedding(x)
            recon_linear1, abu1, endm_get1, feat1 = self.first_block(x)
            #pdb.set_trace()
            
            loss1 = F.mse_loss(recon_linear1, hsi, reduction='mean')
            
            per_cluster_num1 = torch.unique(torch.argmax(abu1, dim=1), return_counts=True)[1].cpu().numpy()
            recon_linear2, abu2, endm_get2, x = self.mamba1(feat1, per_cluster_num1, abu1)

            per_cluster_num2 = torch.unique(torch.argmax(abu2, dim=1), return_counts=True)[1].cpu().numpy()
            loss2 = F.mse_loss(recon_linear2, hsi, reduction='mean')
            
            recon_linear3, abu3, endm_get3, x = self.mamba2(x, per_cluster_num2, abu2)
            per_cluster_num3 = torch.unique(torch.argmax(abu3, dim=1), return_counts=True)[1].cpu().numpy()
            loss3 = F.mse_loss(recon_linear3, hsi, reduction='mean')
            
            recon_linear4, abu4, endm_get4, x = self.mamba3(x, per_cluster_num3, abu3)
            loss4 = F.mse_loss(recon_linear4, hsi, reduction='mean')
            logits = self.cls_head(abu4)
            return logits, 0.1 * (loss1 + loss2 + loss3 + loss4) / 4
            
        else:   
            x = self.patch_embedding(x)
            recon_linear1, abu1, endm_get1, feat1 = self.first_block(x)
            
            loss1 = F.mse_loss(recon_linear1, hsi, reduction='mean')
            
            per_cluster_num1 = torch.unique(torch.argmax(abu1, dim=1), return_counts=True)[1].cpu().numpy()
            
            recon_linear2, abu2, endm_get2, x = self.mamba1(feat1, per_cluster_num1, abu1)
            per_cluster_num2 = torch.unique(torch.argmax(abu2, dim=1), return_counts=True)[1].cpu().numpy()
            loss2 = F.mse_loss(recon_linear2, hsi, reduction='mean')
            
            recon_linear3, abu3, endm_get3, x = self.mamba2(x, per_cluster_num2, abu2)
            per_cluster_num3 = torch.unique(torch.argmax(abu3, dim=1), return_counts=True)[1].cpu().numpy()
            loss3 = F.mse_loss(recon_linear3, hsi, reduction='mean')
            
            recon_linear4, abu4, endm_get4, x = self.mamba3(x, per_cluster_num3, abu3)
            loss4 = F.mse_loss(recon_linear4, hsi, reduction='mean')
            logits = self.cls_head(abu4)
            # show_assignment(abu1, 0)
            # show_assignment(abu2, 1)
            # show_assignment(abu3, 2)
            # show_assignment(abu4, 3)
            return logits





def show_assignment(data, index):

    data = data[0].detach().cpu().numpy()  # (K, H, W)
    fig, axes = plt.subplots(10, 10, figsize=(20, 20))
    for i in range(data.shape[0]):
        ax = axes[i // 10, i % 10]
        ax.imshow(data[i, :, :], cmap='jet')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(f"soft_assignment_{index}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
