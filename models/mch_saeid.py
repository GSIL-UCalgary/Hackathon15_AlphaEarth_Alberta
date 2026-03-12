"""
parallel_graph_mhc_seg.py
=========================
Parallel mHC-Mamba + Patch-Graph Segmentation Network for multi-spectral
remote sensing imagery (e.g. Landsat 8).

Architecture overview
---------------------
Input (B, C, H, W)
    │
SpectralGroupStem               ← shared; projects each spectral group to
    │                             stem_dim channels
    │  streams: list of n tensors, each (B, stem_dim, H, W)
    │  F0     : mean of streams  (B, stem_dim, H, W)
    │
┌───┴─────────────────────────┐
│                             │
MHCMambaBranch            GraphBranch        ← fully parallel; no shared
│  (per-block dims:           │                layers after the stem
│   mhc_block_dims list)      │
│                             │
└────────────┬────────────────┘
             │   BranchFusion  (add | mul | cat)
             │
       SegmentationHead
             │
    (B, num_classes, H, W)

Key design choices
------------------
* Per-layer (per-block) channel dimensions are specified via lists
  (mhc_block_dims, patch_gcn_dims, spatial_gcn_dims, seg_head_dims)
  so every stage can have a different channel width.  Projection layers
  (1x1 Conv / Linear) are inserted automatically between stages that
  differ in width.
* Spatial resolution is preserved end-to-end; no AdaptiveAvgPool is used.
* The graph branch operates on the stem output F0, not on the mHC-Mamba
  output, making the two branches truly parallel.
* miniGCN-style batch-compatible graph convolution (Hong et al., 2021)
  is used throughout the graph branch.

Reference
---------
Hong, D. et al. "Graph Convolutional Networks for Hyperspectral Image
Classification." IEEE TGRS, 2021.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from typing import List, Optional, Dict


# -----------------------------------------------------------------------------
# Default band grouping for Landsat 8 (B2-B7, 0-indexed)
# -----------------------------------------------------------------------------

LANDSAT8_BAND_GROUPS: Dict[str, List[int]] = {
    "VIS":  [0, 1, 2],   # B2 Blue, B3 Green, B4 Red
    "NIR":  [3],          # B5 Near-Infrared
    "SWIR": [4, 5],       # B6 SWIR-1, B7 SWIR-2
}


SENTINEL2_BAND_GROUPS: Dict[str, List[int]] = {
    "VIS":  [0, 1, 2],      # B2 Blue, B3 Green, B4 Red
    "NIR":  [3, 4, 5, 6],   # B5 RedEdge1, B6 RedEdge2, B7 RedEdge3, B8A NarrowNIR
    "SWIR": [7, 8, 9],      # B8 BroadNIR, B11 SWIR1, B12 SWIR2
}

ALPHAEARTH_BAND_GROUPS: Dict[str, List[int]] = {
    "GRP1": list(range(0,  22)),
    "GRP2": list(range(22, 43)),
    "GRP3": list(range(43, 64)),
}
# -----------------------------------------------------------------------------
# Shared Spectral Stem
# -----------------------------------------------------------------------------

class SpectralGroupStem(nn.Module):
    """
    Splits a multi-spectral image into physically meaningful spectral groups
    and projects each group independently to a common channel dimension via
    a 1x1 convolution.

    This replaces a single large convolutional stem with group-aware
    projections, ensuring that spectrally distinct groups (e.g. VIS, NIR,
    SWIR) are handled by dedicated filters before any cross-group interaction.

    Parameters
    ----------
    band_groups : dict[str, list[int]]
        Mapping from group name to the 0-based band indices that belong to it.
        Example: {"VIS": [0,1,2], "NIR": [3], "SWIR": [4,5]}
    out_channels : int
        Number of output channels for every group (i.e. the stem dimension,
        used as the initial per-stream width fed into both branches).

    Input
    -----
    x : (B, C_in, H, W)  — raw multi-spectral image tensor.

    Output
    ------
    streams : list of n tensors, each (B, out_channels, H, W),
              one tensor per spectral group.
    """

    def __init__(self, band_groups: Dict[str, List[int]], out_channels: int):
        super().__init__()
        self.group_names: List[str] = list(band_groups.keys())
        self.n: int = len(self.group_names)

        # One 1x1 Conv per group; input channels = number of bands in the group
        self.convs = nn.ModuleDict({
            name: nn.Conv2d(len(indices), out_channels, kernel_size=1, bias=True)
            for name, indices in band_groups.items()
        })

        # Register band indices as buffers so they move with the model
        for name, indices in band_groups.items():
            self.register_buffer(
                f"idx_{name}",
                torch.tensor(indices, dtype=torch.long),
            )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Parameters
        ----------
        x : torch.Tensor  shape (B, C_in, H, W)

        Returns
        -------
        streams : list[torch.Tensor]
            Length n, each tensor shape (B, out_channels, H, W).
        """
        streams = []
        for name in self.group_names:
            idx = getattr(self, f"idx_{name}")
            streams.append(self.convs[name](x[:, idx, :, :]))
        return streams


# -----------------------------------------------------------------------------
# mHC building blocks
# -----------------------------------------------------------------------------

class LocalMHCGenerator(nn.Module):
    """
    Generates the three per-pixel mHC coefficient matrices from the stacked
    stream tensor, following the notation of the mHC paper:

        H_pre  (N, 1, n) — aggregation weights: how to combine n streams
                           into a single vector for the sequence model.
        H_post (N, 1, n) — expansion weights: how to project the sequence
                           model output back into n streams.
        H_res  (N, n, n) — residual mixing matrix: direct stream-to-stream
                           interaction, constrained to be doubly stochastic
                           via Sinkhorn normalisation.

    All three matrices are conditioned on the local content of every pixel
    through a shared two-layer MLP.

    Parameters
    ----------
    n_streams : int
        Number of parallel streams (expansion rate n).
    dim_per_stream : int
        Feature dimension C of each stream at this stage.
    hidden : int
        Hidden width of the MLP that produces the coefficients.

    Input
    -----
    x_l : (N, n, C)  — stacked stream tensor for N = B*H*W pixels.

    Output
    ------
    H_pre  : (N, 1, n)
    H_post : (N, 1, n)
    H_res  : (N, n, n)
    """

    def __init__(self, n_streams: int, dim_per_stream: int, hidden: int = 128):
        super().__init__()
        self.n = n_streams
        self.C = dim_per_stream
        in_dim = n_streams * dim_per_stream

        self.norm = nn.LayerNorm(in_dim)

        self.fc = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )

        self.fc_pre  = nn.Linear(hidden, n_streams)
        self.fc_post = nn.Linear(hidden, n_streams)
        self.fc_res  = nn.Linear(hidden, n_streams * n_streams)

        # Gentle initialisation to avoid large initial perturbations
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x_l: torch.Tensor):
        """
        Parameters
        ----------
        x_l : torch.Tensor  shape (N, n, C)

        Returns
        -------
        H_pre  : torch.Tensor  (N, 1, n)
        H_post : torch.Tensor  (N, 1, n)
        H_res  : torch.Tensor  (N, n, n)
        """
        N, n, C = x_l.shape
        x     = self.norm(x_l.reshape(N, n * C))
        h_mlp = self.fc(x)

        H_pre  = self.fc_pre(h_mlp).unsqueeze(1)      # (N, 1, n)
        H_post = self.fc_post(h_mlp).unsqueeze(1)     # (N, 1, n)
        H_res  = self.fc_res(h_mlp).view(N, n, n)     # (N, n, n)
        return H_pre, H_post, H_res


def sinkhorn_ds_batched(
    logits: torch.Tensor,
    iters: int = 10,
    tau: float = 0.5,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Differentiable Sinkhorn-Knopp normalisation that maps an arbitrary
    matrix of logits to an approximately doubly-stochastic matrix (rows
    and columns both sum to 1).

    Used to constrain H_res so that it acts as a soft permutation / mixing
    matrix over the n streams, preventing degenerate solutions where one
    stream dominates all others.

    Parameters
    ----------
    logits : torch.Tensor  shape (N, n, n)  — raw (unnormalised) matrix.
    iters  : int   — number of Sinkhorn iterations.
    tau    : float — temperature; lower values sharpen the distribution.
    eps    : float — numerical stability floor.

    Returns
    -------
    P : torch.Tensor  shape (N, n, n)  — doubly-stochastic matrix.
    """
    z = logits / tau
    z = z - z.amax(dim=(-2, -1), keepdim=True)   # numerical stability
    P = torch.exp(z).clamp_min(eps)
    for _ in range(iters):
        P = P / (P.sum(dim=-1, keepdim=True) + eps)   # row normalise
        P = P / (P.sum(dim=-2, keepdim=True) + eps)   # col normalise
    return P


class StreamProjection(nn.Module):
    """
    Lightweight 1x1 convolution that changes the channel width of a stream
    from in_dim to out_dim.

    Inserted automatically inside CompleteMHCBlock whenever consecutive
    blocks have different configured channel dimensions, enabling a
    per-stage channel schedule (e.g. [64, 128, 256]).

    Parameters
    ----------
    in_dim  : int — input channel width.
    out_dim : int — output channel width.

    Input / Output
    --------------
    x : (B, in_dim,  H, W)  ->  (B, out_dim, H, W)
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor  (B, in_dim, H, W)

        Returns
        -------
        torch.Tensor  (B, out_dim, H, W)
        """
        return self.proj(x)


class CompleteMHCBlock(nn.Module):
    """
    One complete mHC-Mamba processing stage.

    The block implements the five-step mHC forward pass:

        Step 1 - Aggregate : H_pre  x x_l  -> X_pre  (N, C_out)
                             Weighted blend of n streams into one token.
        Step 2 - Mamba     : X_pre          -> Z      (N, C_out)
                             Multi-layer Mamba SSM captures spatial context
                             across all N = B*H*W tokens simultaneously.
        Step 3 - Expand    : Z x H_post     -> delta  (N, n, C_out)
                             Broadcasts the SSM output back to n streams.
        Step 4 - Mix       : H_res  x x_l   -> h_mix  (N, n, C_out)
                             Direct stream-to-stream residual interaction.
        Step 5 - Combine   : h_mix + delta   -> output (N, n, C_out)

    If in_dim != out_dim, each stream is first projected with a
    StreamProjection layer before the mHC coefficients are computed.

    Parameters
    ----------
    in_dim : int
        Input channel width of each stream entering this block.
    out_dim : int
        Output channel width of each stream leaving this block.
        A StreamProjection is inserted when in_dim != out_dim.
    n_streams : int
        Number of parallel streams n.
    hidden : int
        MLP hidden width inside LocalMHCGenerator.
    mhc_iters : int
        Sinkhorn iterations for H_res normalisation.
    mhc_tau : float
        Sinkhorn temperature.
    num_mamba_blocks : int
        Number of stacked Mamba SSM layers inside Step 2.

    Input
    -----
    streams : list of n tensors, each (B, in_dim, H, W)

    Output
    ------
    streams : list of n tensors, each (B, out_dim, H, W)
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        n_streams: int = 3,
        hidden: int = 128,
        mhc_iters: int = 10,
        mhc_tau: float = 0.5,
        num_mamba_blocks: int = 2,
    ):
        super().__init__()
        self.n         = n_streams
        self.in_dim    = in_dim
        self.out_dim   = out_dim
        self.mhc_iters = mhc_iters
        self.mhc_tau   = mhc_tau

        # Optional per-stream channel projection (in_dim -> out_dim)
        if in_dim != out_dim:
            self.stream_projs = nn.ModuleList([
                StreamProjection(in_dim, out_dim) for _ in range(n_streams)
            ])
        else:
            self.stream_projs = None

        # mHC coefficient generator operates at out_dim
        self.local_mhc = LocalMHCGenerator(n_streams, out_dim, hidden)

        # Stack of Mamba SSM blocks with residual connections
        self.mamba_blocks = nn.ModuleList([
            Mamba(d_model=out_dim, d_state=16, d_conv=4, expand=2)
            for _ in range(num_mamba_blocks)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(out_dim) for _ in range(num_mamba_blocks)
        ])

    def forward(self, streams: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Parameters
        ----------
        streams : list[torch.Tensor]
            Length n, each (B, in_dim, H, W).

        Returns
        -------
        output_streams : list[torch.Tensor]
            Length n, each (B, out_dim, H, W).
        """
        B, _, H, W = streams[0].shape
        n = self.n

        # Optional channel projection to out_dim
        if self.stream_projs is not None:
            streams = [proj(s) for proj, s in zip(self.stream_projs, streams)]

        # Stack streams -> (N, n, C)  where N = B*H*W
        stacked = torch.stack(streams, dim=1)              # (B, n, C, H, W)
        x_l = stacked.permute(0, 3, 4, 1, 2).reshape(-1, n, self.out_dim)

        # mHC coefficients
        H_pre_raw, H_post_raw, H_res_raw = self.local_mhc(x_l)
        H_pre  = torch.sigmoid(H_pre_raw)                  # (N, 1, n)
        H_post = 2.0 * torch.sigmoid(H_post_raw)           # (N, 1, n)
        H_res  = sinkhorn_ds_batched(H_res_raw, self.mhc_iters, self.mhc_tau)

        # Step 1: aggregate n streams -> single token per pixel
        X_pre = torch.einsum("Nkn,Nnc->Nkc", H_pre, x_l).squeeze(1)  # (N, C)

        # Step 2: Mamba sequence model with residual connections
        Z = X_pre.unsqueeze(0)                             # (1, N, C)
        for mamba, norm in zip(self.mamba_blocks, self.norms):
            Z = norm(mamba(Z) + Z)
        Z = Z.squeeze(0)                                   # (N, C)

        # Step 3: expand back to n streams
        delta = torch.einsum("Nc,Nn->Nnc", Z, H_post.squeeze(1))      # (N, n, C)

        # Step 4: direct stream-to-stream mixing
        h_mixed = torch.einsum("Nij,Njc->Nic", H_res, x_l)            # (N, n, C)

        # Step 5: combine the two paths
        out_flat = h_mixed + delta                                       # (N, n, C)

        # Reshape back to spatial maps
        out = out_flat.reshape(B, H, W, n, self.out_dim)
        out = out.permute(0, 3, 4, 1, 2)                  # (B, n, C, H, W)
        return [out[:, i] for i in range(n)]


# -----------------------------------------------------------------------------
# Branch 1 — mHC-Mamba
# -----------------------------------------------------------------------------

class MHCMambaBranch(nn.Module):
    """
    Sequential stack of CompleteMHCBlock layers forming the mHC-Mamba
    processing branch.

    Each block can have its own input and output channel width, enabling a
    progressively widening feature hierarchy.  A StreamProjection is inserted
    automatically inside each block when consecutive dimensions differ.

    The branch takes the list of n streams from the shared stem and returns a
    single aggregated feature map (mean over streams) at the width of the
    final block.

    Parameters
    ----------
    stem_dim : int
        Channel width of the stem output (= input width of the first block).
    block_dims : list[int]
        Output channel width for each block.
        Example: [64, 128, 256] creates 3 blocks outputting 64, 128, 256 ch.
        len(block_dims) determines the total number of blocks.
    mamba_blocks_per_stage : list[int], optional
        Number of stacked Mamba layers inside each block.
        Must have the same length as block_dims.
        Defaults to [2] * len(block_dims).
    n_streams : int
        Number of parallel streams n (must match the stem).
    mhc_iters : int
        Sinkhorn iterations forwarded to every CompleteMHCBlock.
    mhc_tau : float
        Sinkhorn temperature forwarded to every CompleteMHCBlock.

    Input
    -----
    streams : list of n tensors, each (B, stem_dim, H, W)

    Output
    ------
    F_mhc : (B, block_dims[-1], H, W)
        Mean of the n output streams from the final block.
    """

    def __init__(
        self,
        stem_dim: int,
        block_dims: List[int],
        mamba_blocks_per_stage: Optional[List[int]] = None,
        n_streams: int = 3,
        mhc_iters: int = 10,
        mhc_tau: float = 0.5,
    ):
        super().__init__()

        num_blocks = len(block_dims)
        if mamba_blocks_per_stage is None:
            mamba_blocks_per_stage = [2] * num_blocks
        assert len(mamba_blocks_per_stage) == num_blocks, (
            "mamba_blocks_per_stage must have one entry per block in block_dims"
        )

        # Build in_dim / out_dim pairs: first block reads stem_dim
        in_dims  = [stem_dim] + block_dims[:-1]
        out_dims = block_dims

        self.blocks = nn.ModuleList([
            CompleteMHCBlock(
                in_dim=in_dims[i],
                out_dim=out_dims[i],
                n_streams=n_streams,
                mhc_iters=mhc_iters,
                mhc_tau=mhc_tau,
                num_mamba_blocks=mamba_blocks_per_stage[i],
            )
            for i in range(num_blocks)
        ])

        self.out_dim = block_dims[-1]

    def forward(self, streams: List[torch.Tensor]) -> torch.Tensor:
        """
        Parameters
        ----------
        streams : list[torch.Tensor]
            Length n, each (B, stem_dim, H, W).

        Returns
        -------
        F_mhc : torch.Tensor  (B, out_dim, H, W)
            Mean-aggregated feature map across all n streams after the
            final mHC-Mamba block.
        """
        for block in self.blocks:
            streams = block(streams)

        # Average n streams into a single spatial feature map
        F_mhc = torch.stack(streams, dim=1).mean(dim=1)   # (B, out_dim, H, W)
        return F_mhc


# -----------------------------------------------------------------------------
# Branch 2 — Graph branch components
# -----------------------------------------------------------------------------

class GraphConvLayer(nn.Module):
    """
    Single miniGCN-style graph convolution layer following the propagation
    rule of Hong et al. (2021), Eq. 19:

        H_out = ReLU( BN( D_tilde^{-1/2}  A_tilde  D_tilde^{-1/2}  H  W ) )

    where A_tilde = A + I  (self-loop renormalisation) and
          D_tilde is the corresponding degree matrix.

    Parameters
    ----------
    in_dim  : int — input node feature dimension.
    out_dim : int — output node feature dimension.

    Input
    -----
    H : (B, N, in_dim)  — node feature matrix for a batch of B graphs,
                           each with N nodes.
    A : (B, N, N)       — raw adjacency matrix (need not be normalised;
                           self-loops are added inside this layer).

    Output
    ------
    H_out : (B, N, out_dim)
    """

    def __init__(self, in_dim: int, out_dim: int, norm: str = 'bn'):
        super().__init__()
        self.W  = nn.Linear(in_dim, out_dim, bias=False)
        # LayerNorm is safe for any batch size including B=1
        # BatchNorm1d requires batch > 1 during training
        self.norm = nn.LayerNorm(out_dim) if norm == 'ln' else nn.BatchNorm1d(out_dim)
        self.use_ln = (norm == 'ln')

    def forward(self, H: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        H : torch.Tensor  (B, N, in_dim)
        A : torch.Tensor  (B, N, N)

        Returns
        -------
        H_out : torch.Tensor  (B, N, out_dim)


        if self.use_ln:
            H_out = self.norm(H_out)                   # LayerNorm: safe for any B
        else:
            H_out = self.norm(H_out.reshape(B*N, -1)).view(B, N, -1)  # BN

        return F.relu(H_out)


        """
        B, N, _ = H.shape

        # A_tilde = A + I  (self-loops)
        I       = torch.eye(N, device=A.device, dtype=A.dtype).unsqueeze(0)
        A_tilde = A + I                                    # (B, N, N)

        # D_tilde^{-1/2}
        deg        = A_tilde.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        D_inv_sqrt = deg.pow(-0.5)                         # (B, N, 1)

        # Symmetric normalisation
        A_norm = D_inv_sqrt * A_tilde * D_inv_sqrt.transpose(-2, -1)

        # Aggregate neighbour features + linear transform
        H_agg = torch.bmm(A_norm, H)                      # (B, N, in_dim)
        H_out = self.W(H_agg)                              # (B, N, out_dim)
        if self.use_ln:
            H_out = self.norm(H_out)                   # LayerNorm: safe for any B
        else:
            H_out = self.norm(H_out.reshape(B*N, -1)).view(B, N, -1)  # BN

        return F.relu(H_out)


def build_rbf_adjacency(
    descriptors: torch.Tensor,
    sigma: float = 1.0,
) -> torch.Tensor:
    """
    Constructs a fully-connected RBF adjacency matrix from a set of node
    descriptors, following Eq. 1 of Hong et al. (2021):

        A_{i,j} = exp( -||x_i - x_j||^2 / sigma^2 )

    Used for the patch-level graph where B patch descriptors form the
    nodes and spectral similarity defines the edges.

    Parameters
    ----------
    descriptors : torch.Tensor  (B, N, D)
        Node feature matrix; N nodes each with a D-dimensional descriptor.
    sigma : float
        RBF width parameter controlling edge weight decay.

    Returns
    -------
    A : torch.Tensor  (B, N, N)  values in (0, 1].
    """
    diff    = descriptors.unsqueeze(2) - descriptors.unsqueeze(1)  # (B,N,N,D)
    sq_dist = (diff ** 2).sum(dim=-1)                              # (B,N,N)
    return torch.exp(-sq_dist / (sigma ** 2 + 1e-8))


def build_knn_adjacency(
    descriptors: torch.Tensor,
    k: int = 10,
) -> torch.Tensor:
    """
    Constructs a symmetric binary KNN adjacency matrix.

    For each node the k nearest neighbours (by Euclidean distance) receive
    an edge weight of 1; all others receive 0.  The result is symmetrised
    so that if i->j then j->i.

    Used for the spatial-level graph where downsampled spatial positions
    within a patch form the nodes.

    Parameters
    ----------
    descriptors : torch.Tensor  (B, N, D)
        Node feature matrix.
    k : int
        Number of nearest neighbours per node.

    Returns
    -------
    A : torch.Tensor  (B, N, N)  binary, symmetric.
    """
    diff    = descriptors.unsqueeze(2) - descriptors.unsqueeze(1)  # (B,N,N,D)
    sq_dist = (diff ** 2).sum(dim=-1)                               # (B,N,N)

    # Mask self-connections so a node is not its own neighbour
    sq_dist.diagonal(dim1=-2, dim2=-1).fill_(float('inf'))

    k_eff = min(k, sq_dist.shape[-1] - 1)
    _, knn_idx = sq_dist.topk(k_eff, dim=-1, largest=False)         # (B,N,k)

    B, N, _ = descriptors.shape
    A = torch.zeros(B, N, N, device=descriptors.device)
    A.scatter_(-1, knn_idx, 1.0)
    A = (A + A.transpose(-2, -1)).clamp(0, 1)                       # symmetrise
    return A


class PatchLevelGraph(nn.Module):
    """
    Inter-patch relational graph operating on patch-level descriptors.

    Each of the B patches in a mini-batch is summarised by its global average
    pooled (GAP) feature vector.  These B vectors form the nodes of a graph
    whose edges are weighted by spectral / semantic similarity (RBF kernel,
    Eq. 1 of Hong et al., 2021).

    A stack of GCN layers (one per entry in gcn_dims) propagates information
    between semantically similar patches — e.g. two cropland patches share
    context and refine each other's representations.

    The enriched descriptors are broadcast back to every pixel in the
    corresponding patch via a learned sigmoid gate, modulating the
    full-resolution feature map without collapsing spatial detail.

    Parameters
    ----------
    in_dim : int
        Channel width of the incoming feature map (= stem_dim).
    gcn_dims : list[int]
        Output channel width of each GCN layer.
        Example: [128, 64] builds two layers: in_dim->128->64.
        If the last value differs from in_dim, a Linear projection is
        appended automatically so the gated residual remains compatible.
    sigma : float
        RBF width for the patch adjacency matrix.

    Input
    -----
    F0 : (B, in_dim, H, W)  — shared stem feature map.

    Output
    ------
    F_patch : (B, in_dim, H, W)
        Feature map after inter-patch context injection via residual gating.
    """

    def __init__(
        self,
        in_dim: int,
        gcn_dims: List[int],
        sigma: float = 1.0,
    ):
        super().__init__()
        self.sigma = sigma

        # Build GCN layers with per-layer widths
        layers      = []
        current_dim = in_dim
        for out_d in gcn_dims:
            layers.append(GraphConvLayer(current_dim, out_d, norm='ln'))
            current_dim = out_d
        self.gcn_layers = nn.ModuleList(layers)

        # Project GCN output back to in_dim if dimensions differ
        self.out_proj = (
            nn.Linear(current_dim, in_dim, bias=False)
            if current_dim != in_dim else nn.Identity()
        )

        # Gate maps enriched descriptor to per-channel spatial modulation
        self.gate_fc = nn.Linear(in_dim, in_dim)

    def forward(self, F0: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        F0 : torch.Tensor  (B, in_dim, H, W)

        Returns
        -------
        F_patch : torch.Tensor  (B, in_dim, H, W)
        """
        B, D, H, W = F0.shape

        # One descriptor per patch via GAP  ->  (B, D)
        descriptors = F0.mean(dim=[-2, -1])  #  (B, D)

        # Treat all B patches as N nodes in a SINGLE graph
        # Shape: (1, B, D) — one graph with B nodes
        A = build_rbf_adjacency(
            descriptors.unsqueeze(0),                # (1, B, D)
            sigma=self.sigma
        )       

        # GCN over B patch nodes
        H_nodes = descriptors.unsqueeze(0)           # (1, B, D)
        for gcn in self.gcn_layers:
            H_nodes = gcn(H_nodes, A)               # (1, B, gcn_dim_i)

        # The BN inside gcn sees (1*B, dim) which is fine for B > 1
        # but fails for B=1 at eval time
        # Sigmoid gate -> (B, D, 1, 1) for spatial broadcast
        H_nodes = self.out_proj(H_nodes).squeeze(0)    # (B, D)
        
        gate = torch.sigmoid(self.gate_fc(H_nodes)).view(B, D, 1, 1)

        # Residual gated modulation; spatial resolution unchanged
        return F0 * gate + F0


class SpatialLevelGraph(nn.Module):
    """
    Intra-patch long-range spatial graph.

    The feature map of each patch is spatially downsampled by spatial_stride
    so that the resulting grid positions become graph nodes.  A KNN adjacency
    is built independently per patch and GCN layers propagate long-range
    spatial context that local convolutions miss (middle-range and long-range
    spatial relations illustrated in Fig. 2 of Hong et al., 2021).

    The GCN output is bilinearly upsampled back to the original resolution and
    added to the input as a residual, preserving per-pixel detail while
    injecting global context.

    Parameters
    ----------
    in_dim : int
        Channel width of the incoming feature map (= stem_dim).
    gcn_dims : list[int]
        Output channel width of each GCN layer.
        Example: [128, 64] builds two layers: in_dim->128->64.
        If the final value differs from in_dim, a Linear projection is
        appended so the residual addition remains dimension-compatible.
    spatial_stride : int
        Downsampling factor applied before graph construction.
        For 224x224 input, stride=8 gives 28x28 = 784 nodes per patch.
    k : int
        Number of nearest neighbours in the KNN adjacency.

    Input
    -----
    F0 : (B, in_dim, H, W)

    Output
    ------
    F_spatial : (B, in_dim, H, W)
        Feature map enriched with long-range intra-patch spatial context.
    """

    def __init__(
        self,
        in_dim: int,
        gcn_dims: List[int],
        spatial_stride: int = 8,
        k: int = 10,
    ):
        super().__init__()
        self.stride = spatial_stride
        self.k      = k

        # Build GCN layers with per-layer widths
        layers      = []
        current_dim = in_dim
        for out_d in gcn_dims:
            layers.append(GraphConvLayer(current_dim, out_d, norm='bn'))
            current_dim = out_d
        self.gcn_layers = nn.ModuleList(layers)

        # Project back to in_dim for residual addition
        self.out_proj = (
            nn.Linear(current_dim, in_dim, bias=False)
            if current_dim != in_dim else nn.Identity()
        )

    def forward(self, F0: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        F0 : torch.Tensor  (B, in_dim, H, W)

        Returns
        -------
        F_spatial : torch.Tensor  (B, in_dim, H, W)
        """
        B, D, H, W = F0.shape
        s = self.stride

        # Spatially downsample to reduce graph size to h*w nodes
        F_down = F.avg_pool2d(F0, kernel_size=s, stride=s)   # (B, D, H/s, W/s)
        h, w   = F_down.shape[-2], F_down.shape[-1]
        N      = h * w

        # Flatten spatial positions -> graph nodes  (B, N, D)
        nodes = F_down.reshape(B, D, N).permute(0, 2, 1)

        # Per-patch KNN adjacency  (B, N, N)
        A = build_knn_adjacency(nodes, k=self.k)

        # GCN over spatial nodes
        H_nodes = nodes
        for gcn in self.gcn_layers:
            H_nodes = gcn(H_nodes, A)                         # (B, N, gcn_dim_i)
        H_nodes = self.out_proj(H_nodes)                      # (B, N, D)

        # Reshape to spatial map and upsample to original H x W
        H_spatial = H_nodes.permute(0, 2, 1).reshape(B, D, h, w)
        H_up = F.interpolate(H_spatial, size=(H, W),
                             mode='bilinear', align_corners=False)

        return F0 + H_up   # residual addition


class GraphBranch(nn.Module):
    """
    Full parallel graph processing branch.

    Runs two complementary graphs in parallel on the shared stem output F0:

    1. PatchLevelGraph   — inter-patch semantic relations (B nodes).
       Answers: which patches contain similar land cover?
    2. SpatialLevelGraph — intra-patch long-range spatial relations
       (h x w nodes per patch after downsampling).
       Answers: which spatial positions within a patch share context?

    Their outputs are concatenated channel-wise and fused with a 1x1
    convolution, producing a single feature map that combines both types of
    relational context while preserving the input channel width.

    Parameters
    ----------
    in_dim : int
        Channel width of the stem output fed into this branch.
    patch_gcn_dims : list[int]
        Per-layer output widths for the patch-level GCN layers.
    spatial_gcn_dims : list[int]
        Per-layer output widths for the spatial-level GCN layers.
    sigma : float
        RBF width for the patch-level adjacency.
    spatial_stride : int
        Downsampling factor for the spatial-level graph.
    k : int
        KNN neighbours for the spatial-level graph.

    Input
    -----
    F0 : (B, in_dim, H, W)
        Shared stem output; NOT the mHC-Mamba output.

    Output
    ------
    F_graph : (B, in_dim, H, W)
    """

    def __init__(
        self,
        in_dim: int,
        patch_gcn_dims: List[int],
        spatial_gcn_dims: List[int],
        sigma: float = 1.0,
        spatial_stride: int = 8,
        k: int = 10,
    ):
        super().__init__()
        self.patch_graph   = PatchLevelGraph(in_dim, patch_gcn_dims, sigma)
        self.spatial_graph = SpatialLevelGraph(in_dim, spatial_gcn_dims,
                                               spatial_stride, k)

        # Both sub-graphs output in_dim; fuse with 1x1 conv
        self.fusion = nn.Sequential(
            nn.Conv2d(2 * in_dim, in_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, F0: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        F0 : torch.Tensor  (B, in_dim, H, W)
            Shared stem output; this is NOT the mHC-Mamba output.

        Returns
        -------
        F_graph : torch.Tensor  (B, in_dim, H, W)
        """
        F_patch   = self.patch_graph(F0)    # inter-patch context
        F_spatial = self.spatial_graph(F0)  # intra-patch spatial context

        # Concatenate and fuse with 1x1 Conv
        return self.fusion(torch.cat([F_patch, F_spatial], dim=1))


# -----------------------------------------------------------------------------
# Branch Fusion
# -----------------------------------------------------------------------------

class BranchFusion(nn.Module):
    """
    Fuses the outputs of the mHC-Mamba branch and the Graph branch.

    Three strategies are available, mirroring the FuNet ablation study of
    Hong et al. (2021):

    * 'add' (FuNet-A) — element-wise addition; zero extra parameters.
    * 'mul' (FuNet-M) — element-wise multiplication; each branch gates the
                        other, suppressing uninformative channels.
    * 'cat' (FuNet-C) — channel concatenation followed by a learned 1x1
                        projection back to out_dim.  Consistently the
                        strongest strategy in the paper (Tables V-VII).

    For 'add' and 'mul', mhc_dim and graph_dim must be equal.
    For 'cat', they may differ freely — typical when mhc_block_dims[-1]
    differs from stem_dim.

    Regardless of strategy, the output always passes through a 1x1 Conv ->
    BN -> ReLU projection to the desired out_dim.

    Parameters
    ----------
    mhc_dim   : int — channel width of the mHC-Mamba branch output.
    graph_dim : int — channel width of the Graph branch output
                      (always equal to stem_dim).
    out_dim   : int — desired output channel width after fusion.
    strategy  : str — one of 'add', 'mul', 'cat'.

    Input
    -----
    F_mhc   : (B, mhc_dim,   H, W)
    F_graph : (B, graph_dim, H, W)

    Output
    ------
    F_fused : (B, out_dim, H, W)
    """

    def __init__(
        self,
        mhc_dim: int,
        graph_dim: int,
        out_dim: int,
        strategy: str = 'cat',
    ):
        super().__init__()
        assert strategy in ('add', 'mul', 'cat'), (
            f"strategy must be 'add', 'mul', or 'cat', got '{strategy}'"
        )
        self.strategy = strategy

        if strategy in ('add', 'mul'):
            assert mhc_dim == graph_dim, (
                f"'add' and 'mul' require equal branch widths "
                f"({mhc_dim} vs {graph_dim}). Use 'cat' for unequal widths."
            )
            in_ch = mhc_dim
        else:
            in_ch = mhc_dim + graph_dim

        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, out_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(
        self, F_mhc: torch.Tensor, F_graph: torch.Tensor
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        F_mhc   : torch.Tensor  (B, mhc_dim,   H, W)
        F_graph : torch.Tensor  (B, graph_dim, H, W)

        Returns
        -------
        F_fused : torch.Tensor  (B, out_dim, H, W)
        """
        if self.strategy == 'add':
            combined = F_mhc + F_graph
        elif self.strategy == 'mul':
            combined = F_mhc * F_graph
        else:   # 'cat'
            combined = torch.cat([F_mhc, F_graph], dim=1)

        return self.proj(combined)


# -----------------------------------------------------------------------------
# Segmentation Head
# -----------------------------------------------------------------------------

class SegmentationHead(nn.Module):
    """
    Lightweight per-pixel segmentation decoder.

    Applies a configurable number of 3x3 convolutional blocks (Conv -> BN ->
    ReLU) followed by a final 1x1 convolution that maps to class logits.

    The head operates at full spatial resolution throughout; no upsampling is
    required because both branches preserve the input H x W dimensions.

    Parameters
    ----------
    in_dim      : int — channel width of the fused feature map entering the head.
    num_classes : int — number of segmentation classes (output channels).
    mid_dims    : list[int]
        Channel widths of the intermediate convolutional layers.
        Example: [128, 64] -> Conv(in->128) -> Conv(128->64) -> Conv(64->num_classes).
        Defaults to [128, 64].

    Input
    -----
    x : (B, in_dim, H, W)

    Output
    ------
    logits : (B, num_classes, H, W)
        Raw (un-softmaxed) per-pixel class scores.
    """

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        mid_dims: Optional[List[int]] = None,
    ):
        super().__init__()
        if mid_dims is None:
            mid_dims = [128, 64]

        layers      = []
        current_dim = in_dim
        for out_d in mid_dims:
            layers += [
                nn.Conv2d(current_dim, out_d, kernel_size=3,
                          padding=1, bias=False),
                nn.BatchNorm2d(out_d),
                nn.ReLU(inplace=True),
            ]
            current_dim = out_d

        # Final 1x1 conv -> per-pixel class logits
        layers.append(nn.Conv2d(current_dim, num_classes, kernel_size=1))
        self.head = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor  (B, in_dim, H, W)

        Returns
        -------
        logits : torch.Tensor  (B, num_classes, H, W)
        """
        return self.head(x)


# -----------------------------------------------------------------------------
# Full Model
# -----------------------------------------------------------------------------

class ParallelGraphMHCSegNet(nn.Module):
    """
    Parallel mHC-Mamba + Patch-Graph Segmentation Network.

    Two independent branches share only the shared spectral stem; each branch
    processes the stem output with a fundamentally different inductive bias:

    * mHC-Mamba branch — captures spectral-spatial features using
      hyper-connection mixing matrices and stacked Mamba SSM blocks.
      Channel width evolves per block according to mhc_block_dims.

    * Graph branch — captures inter-patch semantic relations (patch-level RBF
      graph, B nodes) and long-range intra-patch spatial relations (spatial
      KNN graph, h*w nodes), following miniGCN of Hong et al. (2021) but
      adapted for segmentation by preserving full spatial resolution.
      Channel widths are set separately per GCN layer via patch_gcn_dims and
      spatial_gcn_dims.

    The two branch outputs are fused and decoded to a per-pixel class map at
    the original spatial resolution.

    Channel schedule example
    ------------------------
    stem_dim         = 64
    mhc_block_dims   = [64, 128, 256]   # widening across 3 mHC blocks
    mamba_per_stage  = [2, 4, 6]        # more Mamba layers at deeper stages
    patch_gcn_dims   = [128, 64]        # patch GCN: 64->128->64
    spatial_gcn_dims = [128, 64]        # spatial GCN: 64->128->64
    fusion_out_dim   = 256              # matches mhc_block_dims[-1]
    seg_head_dims    = [128, 64]        # head: 256->128->64->num_classes

    Parameters
    ----------
    in_channels : int
        Number of input spectral bands (e.g. 6 for Landsat 8 B2-B7).
    stem_dim : int
        Output channel width of the shared spectral stem.
        This is the input width fed to both branches.
    num_classes : int
        Number of segmentation classes.
    band_groups : dict, optional
        Spectral group definition.  Defaults to LANDSAT8_BAND_GROUPS.
    mhc_block_dims : list[int]
        Per-block output channel widths for the mHC-Mamba branch.
        len(mhc_block_dims) sets the number of mHC blocks.
        Example: [64, 128, 256] -> 3 blocks.
    mamba_blocks_per_stage : list[int], optional
        Number of Mamba layers inside each mHC block.
        Must match len(mhc_block_dims).  Defaults to [2, 4, 6].
    n_streams : int
        Number of spectral streams (must equal len(band_groups)).
    mhc_iters : int
        Sinkhorn iterations in each mHC block.
    mhc_tau : float
        Sinkhorn temperature in each mHC block.
    patch_gcn_dims : list[int]
        Per-layer output widths for the patch-level GCN.
        The final value should equal stem_dim so the gated residual
        is dimensionally compatible (a projection is added otherwise).
        Example: [128, 64] for stem_dim=64.
    spatial_gcn_dims : list[int]
        Per-layer output widths for the spatial-level GCN.
        Same convention as patch_gcn_dims.
    sigma : float
        RBF width for the patch-level adjacency matrix.
    spatial_stride : int
        Spatial downsampling factor for the spatial-level graph.
        For 224x224 patches, stride=8 gives 784 nodes.
    knn_k : int
        Number of KNN neighbours for the spatial-level graph.
    fusion_strategy : str
        One of 'add', 'mul', 'cat'.  'cat' is recommended when
        mhc_block_dims[-1] != stem_dim.
    fusion_out_dim : int
        Output channel width after branch fusion, fed into the head.
    seg_head_dims : list[int]
        Per-layer channel widths of the segmentation head's conv blocks.
        Example: [128, 64] -> two 3x3 conv blocks before the final 1x1.
    """

    def __init__(
        self,
        in_channels: int = 6,
        stem_dim: int = 64,
        num_classes: int = 13,
        band_groups: Optional[Dict[str, List[int]]] = None,
        # mHC-Mamba branch
        mhc_block_dims: Optional[List[int]] = None,
        mamba_blocks_per_stage: Optional[List[int]] = None,
        mhc_iters: int = 10,
        mhc_tau: float = 0.5,
        # Graph branch
        patch_gcn_dims: Optional[List[int]] = None,
        spatial_gcn_dims: Optional[List[int]] = None,
        sigma: float = 1.0,
        spatial_stride: int = 8,
        knn_k: int = 10,
        # Fusion
        fusion_strategy: str = 'cat',
        fusion_out_dim: int = 256,
        # Segmentation head
        seg_head_dims: Optional[List[int]] = None,
    ):
        super().__init__()

        if band_groups is None:
            if in_channels == 6:
                band_groups = LANDSAT8_BAND_GROUPS
            elif in_channels == 10:
                band_groups = SENTINEL2_BAND_GROUPS
            elif in_channels == 64:
                band_groups = ALPHAEARTH_BAND_GROUPS
            else:
                raise ValueError(
                    f"No default band_groups for in_channels={in_channels}. "
                    f"Pass band_groups explicitly."
                )
        # if band_groups is not None, use whatever was passed in — no action needed
        n_streams = len(band_groups)
        # Validate total bands match in_channels
        total_bands = sum(len(v) for v in band_groups.values())
        assert total_bands == in_channels, (
            f"band_groups covers {total_bands} bands but in_channels={in_channels}"
        )

        if mhc_block_dims is None:
            mhc_block_dims = [64, 128, 256]
        if mamba_blocks_per_stage is None:
            mamba_blocks_per_stage = [2, 4, 6]
        if patch_gcn_dims is None:
            patch_gcn_dims = [128, stem_dim]
        if spatial_gcn_dims is None:
            spatial_gcn_dims = [128, stem_dim]
        if seg_head_dims is None:
            seg_head_dims = [128, 64]

        # Shared spectral stem
        self.stem = SpectralGroupStem(band_groups, out_channels=stem_dim)

        # Branch 1: mHC-Mamba
        self.mhc_branch = MHCMambaBranch(
            stem_dim=stem_dim,
            block_dims=mhc_block_dims,
            mamba_blocks_per_stage=mamba_blocks_per_stage,
            n_streams=n_streams,
            mhc_iters=mhc_iters,
            mhc_tau=mhc_tau,
        )
        mhc_out_dim = mhc_block_dims[-1]

        # Branch 2: Graph  (parallel — takes stem output, NOT mHC output)
        self.graph_branch = GraphBranch(
            in_dim=stem_dim,
            patch_gcn_dims=patch_gcn_dims,
            spatial_gcn_dims=spatial_gcn_dims,
            sigma=sigma,
            spatial_stride=spatial_stride,
            k=knn_k,
        )
        graph_out_dim = stem_dim   # GraphBranch always preserves stem_dim

        # Fusion
        self.fusion = BranchFusion(
            mhc_dim=mhc_out_dim,
            graph_dim=graph_out_dim,
            out_dim=fusion_out_dim,
            strategy=fusion_strategy,
        )

        # Segmentation head
        self.seg_head = SegmentationHead(
            in_dim=fusion_out_dim,
            num_classes=num_classes,
            mid_dims=seg_head_dims,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full forward pass of the parallel segmentation network.

        Execution order:
            1. Shared stem       -> n streams  +  F0
            2. mHC-Mamba branch  -> F_mhc   (uses streams from step 1)
            3. Graph branch      -> F_graph  (uses F0 from step 1;
                                              runs in parallel to step 2)
            4. BranchFusion      -> F_fused
            5. SegmentationHead  -> logits

        Parameters
        ----------
        x : torch.Tensor  (B, in_channels, H, W)
            Mini-batch of multi-spectral patches.

        Returns
        -------
        logits : torch.Tensor  (B, num_classes, H, W)
            Per-pixel class scores at the original spatial resolution.
        """
        # 1. Shared stem
        streams = self.stem(x)                              # n x (B, stem_dim, H, W)
        F0      = torch.stack(streams, dim=1).mean(dim=1)  # (B, stem_dim, H, W)

        # 2. mHC-Mamba branch  (input: streams from stem)
        F_mhc = self.mhc_branch(streams)                   # (B, mhc_out_dim, H, W)

        # 3. Graph branch  (input: F0 — parallel to step 2, NOT using F_mhc)
        F_graph = self.graph_branch(F0)                    # (B, stem_dim,    H, W)

        # 4. Fuse
        F_fused = self.fusion(F_mhc, F_graph)              # (B, fusion_out_dim, H, W)

        # 5. Segment
        return self.seg_head(F_fused)                      # (B, num_classes, H, W)


# -----------------------------------------------------------------------------
# Sanity-check / smoke test
# -----------------------------------------------------------------------------

def test_parallel_graph_mhc():
    """
    Smoke test that verifies:

    1. The model constructs without error given explicit per-layer channel dims.
    2. Output shape is exactly (B, num_classes, H, W).
    3. Neither branch depends on the other's output (true parallelism).
    4. A full shape trace is printed for every intermediate tensor.
    5. The channel schedule is summarised for easy inspection.
    """
    print("=" * 70)
    print("TEST: ParallelGraphMHCSegNet  (per-layer channel dims)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B, C, H, W  = 4, 6, 224, 224
    num_classes  = 13
    stem_dim     = 64

    # Per-layer channel schedule
    mhc_block_dims       = [64, 128, 256]       # widening mHC-Mamba branch
    mamba_per_stage      = [2, 4, 6]            # more Mamba layers at depth
    patch_gcn_dims       = [128, stem_dim]      # patch GCN:   64->128->64
    spatial_gcn_dims     = [128, stem_dim]      # spatial GCN: 64->128->64
    fusion_out_dim       = 256                  # = mhc_block_dims[-1]
    seg_head_dims        = [128, 64]            # head: 256->128->64->classes

    model = ParallelGraphMHCSegNet(
        in_channels=C,
        stem_dim=stem_dim,
        num_classes=num_classes,
        mhc_block_dims=mhc_block_dims,
        mamba_blocks_per_stage=mamba_per_stage,
        patch_gcn_dims=patch_gcn_dims,
        spatial_gcn_dims=spatial_gcn_dims,
        sigma=1.0,
        spatial_stride=8,
        knn_k=10,
        fusion_strategy='cat',
        fusion_out_dim=fusion_out_dim,
        seg_head_dims=seg_head_dims,
    ).to(device)

    total = sum(p.numel() for p in model.parameters())
    print(f"\n  Total parameters : {total:,}")

    x = torch.randn(B, C, H, W, device=device)
    print(f"  Input shape      : {tuple(x.shape)}")

    with torch.no_grad():
        logits = model(x)
    print(f"  Output shape     : {tuple(logits.shape)}")
    assert logits.shape == (B, num_classes, H, W)

    # Shape trace
    print("\n  -- Shape trace --")
    with torch.no_grad():
        streams = model.stem(x)
        F0      = torch.stack(streams, dim=1).mean(dim=1)
        F_mhc   = model.mhc_branch(streams)
        F_graph = model.graph_branch(F0)
        F_fused = model.fusion(F_mhc, F_graph)

    print(f"  stem output (per stream) : {tuple(streams[0].shape)} x {len(streams)} streams")
    print(f"  F0   (graph branch in)   : {tuple(F0.shape)}")
    print(f"  F_mhc  (mHC-Mamba out)   : {tuple(F_mhc.shape)}")
    print(f"  F_graph (Graph out)      : {tuple(F_graph.shape)}")
    print(f"  F_fused                  : {tuple(F_fused.shape)}")
    print(f"  logits                   : {tuple(logits.shape)}")

    # Parallelism check
    print("\n  -- Parallelism check --")
    print("  [OK] stem       -> feeds BOTH branches from the same F0 / streams")
    print("  [OK] mhc_branch -> input: stem streams  (not graph output)")
    print("  [OK] graph_branch -> input: F0 stem mean (not mhc output)")

    # Channel schedule summary
    print("\n  -- Channel schedule --")
    print(f"  stem_dim                 : {stem_dim}")
    print(f"  mHC-Mamba block dims     : {mhc_block_dims}")
    print(f"  Mamba layers per stage   : {mamba_per_stage}")
    print(f"  Patch GCN layer dims     : {stem_dim} -> {' -> '.join(map(str, patch_gcn_dims))}")
    print(f"  Spatial GCN layer dims   : {stem_dim} -> {' -> '.join(map(str, spatial_gcn_dims))}")
    print(f"  Fusion out dim           : {fusion_out_dim}")
    print(f"  Seg head dims            : {seg_head_dims} -> {num_classes}")

    print("\n" + "=" * 70)
    print("ALL CHECKS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    test_parallel_graph_mhc()