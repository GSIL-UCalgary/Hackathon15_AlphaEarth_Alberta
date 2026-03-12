import math
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

# Update the ClusterHead.forward to handle this properly:
class ClusterHead(nn.Module):
    """K-means clustering head for semantic segmentation"""

    def __init__(self, feature_dim, num_clusters, temperature=1.0, ema_momentum=0.9, learn_var=True):
        super().__init__()
        self.num_clusters = num_clusters
        self.temperature = temperature
        self.ema_momentum = ema_momentum

    def forward(self, features, cluster_centers, log_vars, alpha=1.0):
        """
        features: [B, L, D]
        cluster_centers: [B, K, D]
        log_vars: [B, K, 1] or [B, K]
        returns: soft assignments [B, L, K]
        """
        B, L, D = features.shape
        
        # features: [B, L, D] -> [B, L, 1, D]
        # cluster_centers: [B, K, D] -> [B, 1, K, D]
        diff = features.unsqueeze(2) - cluster_centers.unsqueeze(1)
        dist2 = diff.pow(2).sum(-1)  # [B, L, K]

        # Ensure log_vars has correct shape: [B, K, 1] -> [B, 1, K]
        if log_vars.dim() == 3:  # [B, K, 1]
            var = torch.exp(log_vars).permute(0, 2, 1)  # [B, 1, K]
        elif log_vars.dim() == 2:  # [B, K]
            var = torch.exp(log_vars).unsqueeze(1)  # [B, 1, K]
        else:
            raise ValueError(f"log_vars has unexpected shape: {log_vars.shape}")
        
        # log likelihood (ignoring constants)
        logits = - dist2 / (2 * var)

        # soft assignments
        probs = F.softmax(logits, dim=-1)
        return probs, logits

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
                 selection_mode='cluster'):
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

        # Mamba SSM
        self.global_mamba = Mamba(
            d_model=dim * 2,
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

    def select_pixels(self, x_proj, cluster_assignments, k_total, per_cluster):
        """Select top-k pixels based on cluster-based selection with empty cluster handling"""
        B, L, C = x_proj.shape
        selected_indices = []
        
        for cluster_idx in range(self.num_clusters):
            # Get number of pixels to select for this cluster
            k = per_cluster[cluster_idx]
            
            # Handle empty clusters (k == 0)
            if k == 0:
                # Return empty tensor for empty clusters
                selected_indices.append(torch.empty((B, 0), device=x_proj.device, dtype=torch.long))
                continue
            
            # Safety check: ensure k doesn't exceed available pixels
            k = min(k, L)
            
            # Handle edge case where k might still be 0 after min(L, 0)
            if k == 0:
                selected_indices.append(torch.empty((B, 0), device=x_proj.device, dtype=torch.long))
                continue
            
            # Get assignment scores for this cluster
            cluster_scores = cluster_assignments[:, :, cluster_idx]
            
            # Select top-k pixels for this cluster
            try:
                topk_scores, topk_indices = torch.topk(cluster_scores, k=k, dim=-1)
                
                # Sort indices by score (optional, but keeps your original logic)
                sorted_scores, sorted_idx = torch.sort(topk_scores, dim=-1, descending=False)
                topk_indices = torch.gather(topk_indices, -1, sorted_idx)
                
                selected_indices.append(topk_indices)
                
            except RuntimeError as e:
                # Handle any unexpected errors in topk
                print(f"Warning: torch.topk failed for cluster {cluster_idx} with k={k}, L={L}: {e}")
                print(f"  cluster_scores shape: {cluster_scores.shape}")
                print(f"  cluster_scores min/max: {cluster_scores.min().item()}/{cluster_scores.max().item()}")
                
                # Fallback: select at least 1 random pixel
                k_fallback = max(1, min(k, L))
                random_indices = torch.randint(0, L, (B, k_fallback), device=x_proj.device)
                selected_indices.append(random_indices)
        
        return selected_indices

    def compute_cluster_centers(self, x, labels, num_clusters, eps=1e-6):
        """
        x: (B, HW, C)
        labels: (B, H, W) -> reshape to (B, HW)
        num_clusters: int - actual number of clusters K (e.g., 50, 30, 20)
        return: (B, K, C)
        """
        B, HW, C = x.shape
        labels_flat = labels.view(B, HW).long()  # (B, HW)
        
        # Validate labels
        valid_labels = labels_flat.clamp(0, num_clusters - 1)
        
        centers = torch.zeros(B, num_clusters, C, device=x.device)
        counts = torch.zeros(B, num_clusters, 1, device=x.device)
        
        # Use one-hot encoding for efficiency
        labels_onehot = F.one_hot(valid_labels, num_classes=num_clusters).float()  # [B, HW, K]
        
        # Compute centers using matrix multiplication
        centers = torch.bmm(labels_onehot.transpose(1, 2), x)  # [B, K, C]
        counts = labels_onehot.sum(dim=1, keepdim=True).transpose(1, 2)  # [B, K, 1]
        
        centers = centers / (counts + eps)
        return centers

    def forward(self, x, per_cluster, cluster_map, return_cluster_assignments=False):
        B, L, C = x.shape
        residual = x
        
        # Normalize and project
        x_norm = self.norm(x)
        x_proj = self.proj_in(x_norm)  # [B, L, expanded_dim]
        
        # Get actual number of clusters from per_cluster length
        num_clusters = len(per_cluster)
        
        # Compute cluster centers
        center = self.compute_cluster_centers(x_proj, cluster_map, num_clusters)

        mu = self.mu_proj(center)  # (B, K, C)
        logvar = self.logvar_proj(center)  # (B, K, C)
        logvar = torch.clamp(logvar, -10.0, 5.0)

        cluster_assignments, logits = self.cluster_head(x_proj, mu, logvar)
        
        # Process all clusters efficiently in parallel
        output = torch.zeros(B, L, C, device=x.device)
        
        for cluster_idx in range(num_clusters):
            k = per_cluster[cluster_idx]
            if k == 0:
                continue
                
            # Ensure k is valid
            k = min(k, L)
            
            # Get scores for this cluster
            cluster_scores = cluster_assignments[:, :, cluster_idx]  # [B, L]
            
            # Select top-k pixels
            topk_scores, topk_indices = torch.topk(cluster_scores, k=k, dim=-1)
            
            # Gather features
            cluster_features = batched_index_select(x_proj, 1, topk_indices)
            
            # Process through conv and mamba
            conv_input = cluster_features.transpose(1, 2)
            conv_output = self.conv_blocks[cluster_idx](conv_input)
            conv_output = conv_output[:, :, :k]  # Trim padding
            conv_output = conv_output.transpose(1, 2)
            
            mamba_output = self.mamba_blocks[cluster_idx](conv_output)
            processed = self.proj_out_blocks[cluster_idx](mamba_output)
            
            # Scatter back
            output.scatter_add_(
                1, 
                topk_indices.unsqueeze(-1).expand(-1, -1, C), 
                processed
            )
        
        # Global processing
        output = output + residual
        output_norm = self.norm2(output)
        output_proj = self.proj_in1(output_norm)
        mamba_out = self.global_mamba(output_proj)
        final_out = self.proj_out(mamba_out)
    
        if return_cluster_assignments:
            return final_out + residual, cluster_assignments, logits
        return final_out + residual

class SpeMamba(nn.Module):
    def __init__(self, channels, token_num=8, use_residual=True, group_num=4):
        super(SpeMamba, self).__init__()
        self.token_num = token_num
        self.use_residual = use_residual

        self.group_channel_num = math.ceil(channels / token_num)
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

    def forward(self, x, cluster_map): # this is for batch with size > 1
        """
        x           : (B, C, H, W)
        cluster_map : (B, H, W)   # Now batched cluster labels
        """
        # Prepare features
        x_pad = self.padding_feature(x)  # (B, C_pad, H, W)
        x_pad = x_pad.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C_pad)

        B, H, W, C_pad = x_pad.shape
        T = self.token_num
        G = self.group_channel_num
        assert C_pad == T * G

        # Flatten pixels per batch
        HW = H * W
        x_flat = x_pad.view(B * HW, C_pad)  # (B*HW, C_pad)
        
        # Cluster map is now (B, H, W), reshape to (B*HW,)
        cluster_flat = cluster_map.view(B, HW).long()  # (B, HW)
        cluster_flat = cluster_flat.view(-1)  # (B*HW,)

        #  Sort by cluster id
        sorted_cluster, perm = torch.sort(cluster_flat)
        x_sorted = x_flat[perm]

        # Split into cluster blocks
        _, counts = torch.unique_consecutive(
            sorted_cluster, return_counts=True
        )
        chunks = torch.split(x_sorted, counts.tolist())

        # Reshape for Mamba
        batched_sparse = torch.cat(
            [c.view(-1, T, G) for c in chunks],
            dim=0
        )  # (BHW, T, G)

        # Mamba
        batched_out = self.mamba(batched_sparse)  # (BHW, T, G)
        batched_out = batched_out.view(B * HW, C_pad)

        # Scatter back
        out_flat = torch.zeros_like(x_flat)
        out_flat[perm] = batched_out

        # Recover spatial shape
        out = out_flat.view(B, H, W, C_pad)
        out = out.permute(0, 3, 1, 2).contiguous()  # (B, C_pad, H, W)

        out = self.proj(out)
        return x + out if self.use_residual else out

class SpaMamba(nn.Module):
    def __init__(self, channels, use_residual=True, group_num=4, use_proj=True,
                 num_clusters=19, sparsity_ratio=1.0, use_attention=True,
                 num_heads=4, selection_mode='cluster'):
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
        
        # FIXED: Remove the "1" and use B instead
        x_flat = x_re.view(B, -1, C)  # Changed from view(1, -1, C) to view(B, -1, C)
        
        x_flat, cluster_logits, distance = self.mamba(x_flat, per_cluster, cluster_map, return_cluster_assignments=True)

        x_recon = x_flat.view(B, H, W, C)
        x_recon = x_recon.permute(0, 3, 1, 2).contiguous()
        if self.use_proj:
            x_recon = self.proj(x_recon)
        if self.use_residual:
            return x_recon + x, cluster_logits, distance
        else:
            return x_recon

class BothMamba(nn.Module):
    def __init__(self, channels, token_num, use_residual, group_num=4, use_att=True,
                 num_clusters=19, sparsity_ratio=1.0, attention_heads=4,
                 selection_mode='cluster'):
        super(BothMamba, self).__init__()
        self.use_att = use_att
        self.use_residual = use_residual
        
        if self.use_att:
            self.weights = nn.Parameter(torch.ones(2) / 2)
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
        self.spe_mamba = SpeMamba(channels, token_num=token_num, use_residual=use_residual, group_num=group_num)

    def forward(self, x, per_cluster, cluster_map):
        spa_x, cluster_logits, distance = self.spa_mamba(x, per_cluster, cluster_map)
        spe_x = self.spe_mamba(x, cluster_map)
        
        if self.use_att:
            weights = self.softmax(self.weights)
            fusion_x = spa_x * weights[0] + spe_x * weights[1]
        else:
            fusion_x = spa_x + spe_x
            
        if self.use_residual:
            return fusion_x + x, cluster_logits, distance
        else:
            return fusion_x


class cluster_MambaHSI(nn.Module):
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
        super(cluster_MambaHSI, self).__init__()
        self.mamba_type = mamba_type

        self.patch_embedding = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=hidden_dim, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(group_num, hidden_dim),
            nn.SiLU()
        )

        if mamba_type == 'both':
            self.mamba1 = BothMamba(
                channels=hidden_dim,
                token_num=token_num,
                use_residual=use_residual,
                group_num=group_num,
                use_att=use_att,
                num_clusters=50,
                sparsity_ratio=sparsity_ratio,
                attention_heads=attention_heads,
                selection_mode=selection_mode
            )

            self.mamba2 = BothMamba(
                channels=hidden_dim,
                token_num=token_num,
                use_residual=use_residual,
                group_num=group_num,
                use_att=use_att,
                num_clusters=30,
                sparsity_ratio=sparsity_ratio,
                attention_heads=attention_heads,
                selection_mode=selection_mode
            )

            self.mamba3 = BothMamba(
                channels=hidden_dim,
                token_num=token_num,
                use_residual=use_residual,
                group_num=group_num,
                use_att=use_att,
                num_clusters=20,
                sparsity_ratio=sparsity_ratio,
                attention_heads=attention_heads,
                selection_mode=selection_mode
            )

        self.cls_head = nn.Sequential(
            nn.Conv2d(in_channels=hidden_dim, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(group_num, 128),
            nn.SiLU(),
            nn.Conv2d(in_channels=128, out_channels=num_classes, kernel_size=1, stride=1, padding=0)
        )
    
    def forward(self, x, per_cluster_num, labels50, labels30, labels20):
        """
        Modified for batch_size > 1 support
        
        Args:
            x: (B, C, H, W) - Input image batch (B can be > 1)
            per_cluster_num: List of lists - [[per_cluster_100], [per_cluster_50], [per_cluster_30]]
            labels50: (B, H, W) - Cluster labels for 50 clusters (must be batched)
            labels30: (B, H, W) - Cluster labels for 30 clusters (must be batched)
            labels20: (B, H, W) - Cluster labels for 20 clusters (must be batched)
        """
        # Get batch size and dimensions
        B, C, H, W = x.shape
        
        # Extract per cluster numbers (same for all batches)
        cluster100 = per_cluster_num[0]  # List of 50 numbers
        cluster50 = per_cluster_num[1]   # List of 30 numbers
        cluster30 = per_cluster_num[2]   # List of 20 numbers

        if self.training:
            # Step 1: Initial projection
            x = self.patch_embedding(x)  # (B, hidden_dim, H, W)

            # Step 2: First Mamba block (50 clusters)
            x, cluster_logits1, _ = self.mamba1(x, cluster100, labels50)
            B_current, D, H_out, W_out = x.shape
            k1 = cluster_logits1.shape[2]  # Should be 50
            
            # Reshape cluster logits from (B, L, 50) to (B, 50, H, W)
            cluster_logits1 = cluster_logits1.reshape(B_current, H_out, W_out, k1)
            cluster_logits1 = cluster_logits1.permute(0, 3, 1, 2)  # (B, 50, H_out, W_out)
            
            # Ensure labels50 has batch dimension and correct size
            if labels50.dim() == 2:  # Single 2D map (H, W) - legacy support
                labels50 = labels50.unsqueeze(0).expand(B_current, H_out, W_out)
            elif labels50.dim() == 3 and labels50.shape[0] == 1:  # (1, H, W)
                labels50 = labels50.expand(B_current, H_out, W_out)
            elif labels50.shape[1:] != (H_out, W_out):
                # Resize if needed
                labels50 = F.interpolate(
                    labels50.float().unsqueeze(1), 
                    size=(H_out, W_out), 
                    mode='nearest'
                ).squeeze(1).long()
            
            # Compute cluster loss 1
            loss1 = F.cross_entropy(cluster_logits1, labels50, reduction='mean')

            # Step 3: Second Mamba block (30 clusters)
            x, cluster_logits2, _ = self.mamba2(x, cluster50, labels30)
            B_current, D, H_out, W_out = x.shape
            k2 = cluster_logits2.shape[2]  # Should be 30
            
            # Reshape cluster logits
            cluster_logits2 = cluster_logits2.reshape(B_current, H_out, W_out, k2)
            cluster_logits2 = cluster_logits2.permute(0, 3, 1, 2)  # (B, 30, H_out, W_out)
            
            # Ensure labels30 has correct size
            if labels30.dim() == 2:
                labels30 = labels30.unsqueeze(0).expand(B_current, H_out, W_out)
            elif labels30.dim() == 3 and labels30.shape[0] == 1:
                labels30 = labels30.expand(B_current, H_out, W_out)
            elif labels30.shape[1:] != (H_out, W_out):
                labels30 = F.interpolate(
                    labels30.float().unsqueeze(1), 
                    size=(H_out, W_out), 
                    mode='nearest'
                ).squeeze(1).long()
                
            loss2 = F.cross_entropy(cluster_logits2, labels30, reduction='mean')

            # # Step 4: Third Mamba block (20 clusters)
            # x, cluster_logits3, _ = self.mamba3(x, cluster30, labels20)
            # B_current, D, H_out, W_out = x.shape
            # k3 = cluster_logits3.shape[2]  # Should be 20
            
            # cluster_logits3 = cluster_logits3.reshape(B_current, H_out, W_out, k3)
            # cluster_logits3 = cluster_logits3.permute(0, 3, 1, 2)  # (B, 20, H_out, W_out)
            
            # # Ensure labels20 has correct size
            # if labels20.dim() == 2:
            #     labels20 = labels20.unsqueeze(0).expand(B_current, H_out, W_out)
            # elif labels20.dim() == 3 and labels20.shape[0] == 1:
            #     labels20 = labels20.expand(B_current, H_out, W_out)
            # elif labels20.shape[1:] != (H_out, W_out):
            #     labels20 = F.interpolate(
            #         labels20.float().unsqueeze(1), 
            #         size=(H_out, W_out), 
            #         mode='nearest'
            #     ).squeeze(1).long()
                
            # loss3 = F.cross_entropy(cluster_logits3, labels20, reduction='mean')

            # Step 5: Final classification head
            logits = self.cls_head(x)  # (B, num_classes, H_out, W_out)
            return logits, 0.1 * (loss1  + loss2)
            #return logits, 0.1 * (loss1 + loss2 + loss3)
        
        else:
            # Evaluation mode
            x = self.patch_embedding(x)
            
            # Process through mamba blocks
            x, cluster_logits1, distance1 = self.mamba1(x, cluster100, labels50)
            B_current, D, H_out, W_out = x.shape
            
            x, cluster_logits2, distance2 = self.mamba2(x, cluster50, labels30)
            x, cluster_logits3, distance3 = self.mamba3(x, cluster30, labels20)
            
            # Get final logits
            logits = self.cls_head(x)
            
            return logits

def show_assignment(data, index):
    data = data[0].detach().cpu().numpy()  # (K, H, W)
    fig, axes = plt.subplots(5, 10, figsize=(20, 20))
    for i in range(data.shape[0]):
        ax = axes[i // 10, i % 10]
        ax.imshow(data[i, :, :], cmap='jet')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(f"soft_assignment_{index}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
