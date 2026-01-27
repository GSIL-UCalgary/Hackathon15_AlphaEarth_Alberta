# 实现 
# mHC: Manifold-Constrained Hyper-Connections
# https://arxiv.org/abs/2409.19606
# https://arxiv.org/pdf/2512.24880

import torch
import torch.nn as nn
import math
import warnings
from mamba_ssm import Mamba
import numpy as np
import tifffile
from PIL import Image
from pathlib import Path
from torchsummary import summary
from torchinfo import summary as torchinfo_summary
import pprint
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    """
    Initializes a tensor with values drawn from a truncated normal distribution.
    
    This function is a PyTorch implementation of truncated normal initialization,
    which generates values from a normal distribution that are truncated to
    the range [a, b]. It operates without tracking gradients.
    
    Args:
        tensor (torch.Tensor): The tensor to initialize.
        mean (float): Mean of the normal distribution.
        std (float): Standard deviation of the normal distribution.
        a (float): Lower truncation bound.
        b (float): Upper truncation bound.
        
    Returns:
        torch.Tensor: The initialized tensor (same tensor, modified in-place).
        
    Note:
        - Warns if the mean is more than 2 standard deviations from [a, b],
          as this may result in an incorrect distribution.
        - Uses the inverse CDF transform method to generate values.
        - Operates in a `torch.no_grad()` context.
    """
    def norm_cdf(x):
        """Computes standard normal cumulative distribution function."""
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    # Check if mean is far outside truncation bounds
    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Step 1: Calculate CDF bounds for truncation
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        
        # Step 2: Generate uniform samples in transformed range [2l-1, 2u-1]
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        
        # Step 3: Apply inverse error function to get truncated standard normal
        tensor.erfinv_()
        
        # Step 4: Scale and shift to desired mean and std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        
        # Step 5: Clamp to ensure values stay within [a, b] (safety measure)
        tensor.clamp_(min=a, max=b)
        
    return tensor

def trunc_normal_(tensor, mean: float = 0., std: float = 1., a: float = -2., b: float = 2.):
    """
    Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:a <= \text{mean} <= b.

    Args:
        tensor: An n-dimensional `torch.Tensor`
        mean: The mean of the normal distribution
        std: The standard deviation of the normal distribution
        a: The minimum cutoff value
        b: The maximum cutoff value

    Returns:
        torch.Tensor: The initialized tensor (modified in-place)

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
        
    Note:
        This function is a wrapper around `_no_grad_trunc_normal_()` that
        provides a clean interface with reasonable default values.
        Default values create a distribution with mean 0, std 1,
        truncated to [-2, 2] (approximately 95% of a standard normal).
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

class SinusoidalPositionEncoding(nn.Module):
    """
    Sinusoidal Position Encoding - encodes raw coordinate values (x, y positions) 
    into high-dimensional sinusoidal embeddings.
    
    This implementation is commonly used for position encoding in transformers,
    but adapted for raw coordinate inputs rather than position indices.
    
    The encoding uses sine and cosine functions of different frequencies:
        PE(pos, 2i) = sin(pos / max_len * 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / max_len * 10000^(2i/d_model))
    
    Args:
        d_model (int): The dimensionality of the output embeddings.
                       Must be even because each position gets both sin and cos components.
        div_term (torch.Tensor): Precomputed frequency terms for efficient computation.
        max_len (int, optional): Maximum coordinate value for normalization.
                                 Defaults to 512.
    Returns:
        torch.Tensor: Sinusoidal position embeddings of shape (batch, seq_len, d_model).


    """
    def __init__(self, d_model, max_len=512):
        super().__init__()
        assert d_model % 2 == 0, "d_model must be even for sinusoidal encoding"
        self.d_model = d_model
        # Precompute frequency components: exp(-2i * log(10000) / d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        self.register_buffer('div_term', div_term)
        self.max_len = max_len
    
    def forward(self, pos):
        """
        Forward pass to compute sinusoidal position embeddings.
        
        Args:
            pos (torch.Tensor): Raw coordinate values.
                                Shape: (batch, seq_len) or (batch, seq_len, 1)
        
        Returns:
            torch.Tensor: Sinusoidal position embeddings.
                          Shape: (batch, seq_len, d_model)
        
        Note:
            - Input coordinates are normalized to [0, 1] range by dividing by max_len.
            - Even indices get sine values, odd indices get cosine values.
            - The encoding is deterministic and does not require learning parameters.
        """
        if pos.dim() == 3:
            pos = pos.squeeze(-1)
        # Normalize to the range [0, 1]
        pos_normalized = pos / self.max_len
        # Expand dimensions: (batch, seq_len, d_model//2)
        pos_expanded = pos_normalized.unsqueeze(-1) * self.div_term * self.max_len
        # Alternate sin/cos
        pe = torch.zeros(*pos.shape, self.d_model, device=pos.device, dtype=pos.dtype)
        pe[..., 0::2] = torch.sin(pos_expanded)
        pe[..., 1::2] = torch.cos(pos_expanded)
        return pe
    
    def extra_repr(self):
        """String representation for printing the module."""
        return f'd_model={self.d_model}, max_len={self.max_len}'

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """
    Drop paths (Stochastic Depth) per sample when applied in main path of residual blocks.
    
    Implements stochastic depth regularization by randomly dropping entire paths through
    the network during training. This helps prevent overfitting and encourages diverse
    feature learning paths.
    
    This is similar to DropConnect but applied at the block level rather than weight level.
    The name 'drop path' is used to avoid confusion with DropConnect.
    
    Args:
        x (torch.Tensor): Input tensor of any shape.
        drop_prob (float): Probability of dropping a path (setting it to zero).
                           Range: [0.0, 1.0]. Default: 0.0 (no dropout).
        training (bool): Whether the model is in training mode.
                         If False, returns x unchanged.
        scale_by_keep (bool): Whether to scale the output by 1/(1-drop_prob) during training.
                              This maintains the expected activation magnitude.
                              Default: True.
    
    Returns:
        torch.Tensor: The input tensor with paths randomly dropped during training.
    
    Note:
        - During inference (training=False), the function returns x unchanged.
        - The dropout mask is sampled per sample in the batch.
        - The mask has shape (batch_size, 1, 1, ...) to drop entire paths per sample.
        - Scaling by keep_prob preserves the expected output magnitude.
    
    Examples:
        >>> x = torch.randn(4, 64, 32, 32)  # Batch of 4 samples
        >>> x_dropped = drop_path(x, drop_prob=0.2, training=True)
        >>> # On average, 20% of samples will have their entire path set to zero
    """
    if drop_prob == 0. or not training:
        return x
    
    keep_prob = 1 - drop_prob
    # Create shape: (batch_size, 1, 1, ...) to drop entire paths per sample
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    
    # Generate random dropout mask: Bernoulli with probability keep_prob
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    
    # Scale by keep_prob to maintain expected activation magnitude
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    
    return x * random_tensor


class DropPath(nn.Module):
    """
    This module wraps the drop_path function into a reusable nn.Module that
    can be easily integrated into neural network architectures. It implements
    stochastic depth regularization by randomly dropping entire paths during
    training, which helps prevent overfitting in deep networks.
    """
    def __init__(self, drop_prob=None, scale_by_keep=True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

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
    
    def extra_repr(self):
        """String representation for printing the module."""
        return f'dim={self.weight.shape[0]}, eps={self.eps}'
    
def sinkhorn_knopp(matrix: torch.Tensor, num_iter: int = 20, epsilon: float = 1e-20) -> torch.Tensor:
    """
    Sinkhorn-Knopp algorithm for projecting matrices onto the bistochastic manifold.
    
    This function applies the Sinkhorn-Knopp iterative normalization algorithm to
    convert a square matrix into a bistochastic (doubly stochastic) matrix.
    A bistochastic matrix has all rows and columns summing to 1, with non-negative entries.
    
    The algorithm alternates between row and column normalization:
        1. Row normalization: K = K / K.sum(dim=-1, keepdim=True)
        2. Column normalization: K = K / K.sum(dim=-2, keepdim=True)
    
    Parameters:
        matrix (torch.Tensor): Input matrix with shape [batch_size, n, n] or [n, n].
                               Typically contains log-probabilities or affinity scores.
        num_iter (int): Number of Sinkhorn iterations. 
                       Recommended: 20 (from mHC paper: turn0search2, turn0search5).
                       More iterations improve approximation but increase computation.
        epsilon (float): Small constant for numerical stability.
                        Prevents division by zero. Default: 1e-20.
    
    Returns:
        torch.Tensor: Bistochastic matrix with same shape as input.
                     All rows and columns sum to 1 (within numerical precision).
    
    Note:
        - The input matrix is first processed with torch.nan_to_num() for robustness.
        - Log-Sum-Exp trick is applied: subtract max for numerical stability before exp.
        - The algorithm converges to the unique bistochastic matrix with the same
          "pattern" as the input (preserving zeros in the same positions).
        - Convergence rate is linear in the number of iterations.
    
    Example:
        >>> A = torch.randn(2, 3, 3)  # Batch of 2 matrices, 3x3 each
        >>> B = sinkhorn_knopp(A, num_iter=20)
        >>> print(B.sum(dim=-1))  # All rows sum to ~1
        >>> print(B.sum(dim=-2))  # All columns sum to ~1
    """
    # Ensure numerical robustness: handle NaN and infinite values
    matrix = torch.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Apply Log-Sum-Exp trick: subtract max to prevent exp overflow
    # This doesn't change the final bistochastic matrix due to scaling invariance
    matrix = matrix - torch.max(matrix, dim=-1, keepdim=True)[0]
    
    # Exponentiate to get non-negative matrix
    K = torch.exp(matrix)
    
    # Sinkhorn iterations: alternate row and column normalization
    for _ in range(num_iter):
        # Row normalization: make each row sum to 1
        K = K / (K.sum(dim=-1, keepdim=True) + epsilon)
        # Column normalization: make each column sum to 1
        K = K / (K.sum(dim=-2, keepdim=True) + epsilon)
    
    return K

def init_transformer_weights(module: nn.Module) -> None:
    """
    Initialize transformer module weights with custom initialization schemes.
    
    This function provides specialized weight initialization for various transformer
    components, following common practices in transformer architectures:
    - Linear layers: truncated normal with std=0.02
    - Embeddings: truncated normal with std=0.02 (zero padding_idx)
    - Convolutional layers: truncated normal with std=0.02
    - Normalization layers: ones for weights, zeros for biases
    - MultiheadAttention: specialized initialization for different weight types
    
    Special handling for HyperConnection modules to maintain dynamic scaling
    factors at recommended small values (0.01) as per the mHC paper.
    
    Args:
        module (nn.Module): The PyTorch module to initialize.
    
    Returns:
        None: The module is modified in-place.
    
    Note:
        - The function uses early returns after handling each module type.
        - For nn.Embedding, padding_idx is explicitly set to zero.
        - For nn.MultiheadAttention, handles both combined and separate projections.
        - HyperConnection scaling factors are fixed to prevent overwriting.
        - Uses trunc_normal_ with std=0.02 for most learnable parameters.
    
    Example:
        >>> model = HyperConnectionTransformer(...)
        >>> model.apply(init_transformer_weights)
    """
    # --- HyperConnection Module Initialization Check ---
    if isinstance(module, HyperConnection):        
        # Ensure dynamic mapping scaling factors remain at the recommended small value (0.01) from the paper
        # Prevent accidental overwriting by other initialization logic (e.g., trunc_normal_)
        if hasattr(module, 'dynamic_alpha_scale'):
            nn.init.constant_(module.dynamic_alpha_scale, 0.01)        
        if hasattr(module, 'dynamic_beta_scale'):
            nn.init.constant_(module.dynamic_beta_scale, 0.01)        
        return
    
    # --- Linear Layers ---
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
        return
    
    # --- Embedding Layers ---
    if isinstance(module, nn.Embedding):
        trunc_normal_(module.weight, std=0.02)
        if module.padding_idx is not None:
            with torch.no_grad():
                module.weight[module.padding_idx].fill_(0)
        return
    
    # --- Convolutional Layers ---
    if isinstance(module, nn.Conv2d):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
        return
    
    # --- RMSNorm Layers ---
    if isinstance(module, RMSNorm):
        nn.init.ones_(module.weight)
        return
    
    # --- LayerNorm Layers ---
    if isinstance(module, nn.LayerNorm):
        if module.weight is not None:
            nn.init.ones_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
        return
    
    # --- Multihead Attention Layers ---
    if isinstance(module, nn.MultiheadAttention):
        # Handle in_proj_weight (combined QKV) or separate projections
        if getattr(module, 'in_proj_weight', None) is not None:
            trunc_normal_(module.in_proj_weight, std=0.02)
        else:
            if getattr(module, 'q_proj_weight', None) is not None:
                trunc_normal_(module.q_proj_weight, std=0.02)
            if getattr(module, 'k_proj_weight', None) is not None:
                trunc_normal_(module.k_proj_weight, std=0.02)
            if getattr(module, 'v_proj_weight', None) is not None:
                trunc_normal_(module.v_proj_weight, std=0.02)
        
        # Initialize biases
        if getattr(module, 'in_proj_bias', None) is not None:
            nn.init.zeros_(module.in_proj_bias)
        
        # Initialize output projection
        if getattr(module, 'out_proj', None) is not None:
            trunc_normal_(module.out_proj.weight, std=0.02)
            if module.out_proj.bias is not None:
                nn.init.zeros_(module.out_proj.bias)
        
        # Initialize optional bias parameters
        if getattr(module, 'bias_k', None) is not None:
            nn.init.zeros_(module.bias_k)
        if getattr(module, 'bias_v', None) is not None:
            nn.init.zeros_(module.bias_v)
        return

# ------------------------------------------------------------------------------------
# 1. Core Module: HyperConnection (Fully Corrected mHC Version)
# ------------------------------------------------------------------------------------
class SumToOne(nn.Module):
    """
    Sum-to-One normalization layer using softmax scaling.
    
    This layer applies a softmax operation scaled by a temperature-like parameter
    to ensure that the output sums to 1 along the specified dimension.
    
    The forward pass computes:
        output = softmax(scale * x)
    
    This is useful in scenarios where you need probability distributions,
    abundance estimation, or attention weights that must sum to one.
    
    Args:
        scale (float, optional): Scaling factor applied before softmax.
                                 Larger values make the distribution more peaked.
                                 Smaller values make it more uniform.
                                 Default: 3.5.
    
    Attributes:
        scale (float): The scaling factor for softmax.
    
    Note:
        - The softmax is applied along dimension 1 by default.
        - The scale parameter acts like an inverse temperature: 
          higher scale = more confident/peaked distribution.
        - Unlike standard softmax, this includes a learnable scale parameter
          that can be adjusted during training.
        - Outputs are always non-negative and sum to 1 along dimension 1.
    
    Example:
        >>> layer = SumToOne(scale=2.0)
        >>> x = torch.randn(4, 10, 32, 32)  # Batch of 4, 10 channels
        >>> y = layer(x)  # Shape: (4, 10, 32, 32), sums to 1 along channel dim
        >>> print(y.sum(dim=1))  # All values should be ~1.0
    """
    def __init__(self, scale=3.5):
        super(SumToOne, self).__init__()
        self.scale = scale

    def forward(self, x):
        """
        Forward pass applying scaled softmax normalization.
        
        Args:
            x (torch.Tensor): Input tensor of any shape.
                              Softmax is applied along dimension 1.
        
        Returns:
            torch.Tensor: Normalized tensor of same shape as input.
                          Values are non-negative and sum to 1 along dimension 1.
        
        Note:
            - The operation is fully differentiable.
            - Gradient flow is maintained through both x and the scale parameter.
            - Numerical stability is handled by PyTorch's softmax implementation.
        """
        # Apply scaled softmax along dimension 1
        x = torch.softmax(x * self.scale, dim=1)
        return x
    
    def extra_repr(self):
        """String representation for printing the module."""
        return f'scale={self.scale}'
    
class HyperConnection(nn.Module):
    """
    Hyper-Connection Module (mHC Version).
    
    This module replaces traditional residual connections with a hyper-connection mechanism
    that includes two sequential steps: width connection and depth connection.
    
    The mHC (modified Hyper-Connection) version includes several enhancements over
    the original HC formulation:
    1. Applies non-negative constraints (Sigmoid) to H_pre and H_post matrices
    2. Uses Sinkhorn-Knopp projection to enforce bistochastic constraints on H_res
    3. Employs flattening instead of mean pooling for dynamic mapping inputs
    4. Uses mean aggregation instead of sum for final output
    
    The module can operate in either static or dynamic (DHC) mode:
    - Static: Uses fixed learnable connection matrices
    - Dynamic: Generates connection weights based on input features
    
    Args:
        dim (int): Hidden dimension size.
        rate (int): Number of parallel streams/hyper-connections.
        layer_id (int): Identifier for the layer position in the network.
        dynamic (bool): Whether to use dynamic weight generation.
                       If True, uses DHC (Dynamic Hyper-Connection).
                       If False, uses static learnable weights.
                       Default: True.
        device (torch.device, optional): Device to initialize parameters on.
    
    Attributes:
        dim (int): Hidden dimension size.
        rate (int): Number of parallel streams.
        layer_id (int): Layer identifier.
        dynamic (bool): Dynamic mode flag.
        num_queries_times (int): Multiplier for number of queries (default: 30).
        num_endm (int): Number of endmembers (set equal to rate).
        static_alpha (nn.Parameter): Static alpha matrix of shape [rate, rate+1].
        static_beta (nn.Parameter): Static beta vector of shape [rate].
        query_embed (nn.Embedding): Query embedding layer.
        weights (nn.Parameter): Learnable weights for abundance estimation.
        abundance_constrain (SumToOne): Normalization layer for abundance maps.
        layer_norm (RMSNorm, optional): Normalization layer for dynamic mode.
        dynamic_alpha_fn (nn.Linear, optional): Dynamic alpha generation function.
        dynamic_beta_fn (nn.Linear, optional): Dynamic beta generation function.
        dynamic_alpha_scale (nn.Parameter, optional): Scaling factor for dynamic alpha.
        dynamic_beta_scale (nn.Parameter, optional): Scaling factor for dynamic beta.
    
    Note:
        - The module implements Equation (3) from the mHC paper: 
          x^{l+1} = H_res^l x^l + (H_post^l)^T F(H_pre^l x^l)
        - Static initialization follows paper recommendations: identity for H_res,
          one-hot for H_pre, uniform for H_post.
        - Dynamic scaling factors are initialized to 0.01 as per paper Appendix A.1.
        - For dynamic mode, input is flattened to preserve full context (Eq. 7).
    
    Reference:
        mHC paper with detailed mathematical formulation and experimental results.
    """
    def __init__(self, dim: int, rate: int, layer_id: int, dynamic: bool = True, device=None):
        super().__init__()
        self.dim = dim
        self.rate = rate
        self.layer_id = layer_id
        self.dynamic = dynamic
        self.num_queries_times = 30
        self.num_endm = self.rate
        
        # --- Static mapping initialization ---
        # Paper recommendation: static mappings should be initialized close to identity or uniform distribution
        # For H_res: identity matrix initialization for identity mapping
        init_H_res = torch.eye(rate, device=device)
        # For H_pre: [1, 0, 0, ..., 0] corresponds to keeping the 0th stream as layer input
        init_H_pre = torch.zeros(rate, 1, device=device)
        init_H_pre[0, 0] = 1.0
        # Concatenate to form static_alpha: [rate, rate + 1]
        self.static_alpha = nn.Parameter(torch.cat([init_H_pre, init_H_res], dim=1))
        # For H_post (beta): initialize to uniform distribution [1/rate, 1/rate, ..., 1/rate]
        self.static_beta = nn.Parameter(torch.ones(rate, device=device) / rate)
        self.num_queries = self.num_queries_times * self.num_endm
        self.query_embed = nn.Embedding(self.num_queries, 200)
        self.weights = nn.Parameter(torch.ones((self.num_endm, self.num_queries_times)))
        self.abundance_constrain = SumToOne(scale=3.5)
        if self.dynamic:
            self.layer_norm = RMSNorm(dim)
            # --- Key Fix 1: Dynamic mapping input uses Flattening (paper Equation 7) ---
            # Input dimension should be rate * dim (Flattened) to preserve complete contextual information
            self.dynamic_alpha_fn = nn.Linear(rate * dim, rate * (rate + 1), bias=False)
            self.dynamic_beta_fn = nn.Linear(rate * dim, rate, bias=False)
            
            # Paper A.1 mentions gating factor initialized to small value (0.01)
            self.dynamic_alpha_scale = nn.Parameter(torch.full((1,), 0.1))
            self.dynamic_beta_scale = nn.Parameter(torch.full((1,), 0.1))

    def width_connection(self, h: torch.Tensor):
        """
        Perform width connection step (Equation 2 in mHC paper).
        
        This step applies hyper-connection matrices to distribute information
        across parallel streams and prepare input for the core layer F.
        
        Args:
            h (torch.Tensor): Input hidden states.
                             Shape: [B, L, rate, dim]
                             where B=batch, L=sequence_length, rate=streams, dim=hidden_dim.
        
        Returns:
            tuple: Contains:
                - mix_h (torch.Tensor): Mixed hidden states for layer F.
                                       Shape: [B, L, rate+1, dim]
                - beta_constrained (torch.Tensor): Constrained H_post matrix.
                                                  Shape: [B, L, rate]
                - H_res (torch.Tensor): Bistochastic H_res matrix.
                                       Shape: [B, L, rate, rate]
                - H_pre (torch.Tensor): Non-negative H_pre matrix.
                                       Shape: [B, L, rate, 1]
        
        Note:
            - If dynamic=True, generates connection weights based on input.
            - Applies mHC constraints: sigmoid for H_pre/H_post, Sinkhorn for H_res.
            - Returns connection matrices for visualization/debugging purposes.
        """
        B, L, N, D = h.shape # [B, L, rate, dim]
        if self.dynamic:
            norm_h = self.layer_norm(h) # [B, L, N, D]
            # --- Key Fix 2: Flatten according to paper Equation (7) ---
            agg_h = norm_h.view(B, L, -1) # [B, L, rate*dim]
            
            dyn_alpha = torch.tanh(self.dynamic_alpha_fn(agg_h)).view(B, L, self.rate, self.rate + 1)
            dyn_beta = self.dynamic_beta_fn(agg_h).view(B, L, self.rate)
            
            # Apply gating scale and add to static part
            alpha = self.static_alpha + dyn_alpha * self.dynamic_alpha_scale
            beta = self.static_beta + dyn_beta * self.dynamic_beta_scale
        else:
            alpha = self.static_alpha.expand(B, L, -1, -1)
            beta = self.static_beta.expand(B, L, -1)

        # --- mHC Core Constraints Begin ---
        # Split alpha: [B, L, rate, rate+1] -> H_pre_raw [B, L, rate, 1], H_res_raw [B, L, rate, rate]
        H_pre_raw = alpha[..., :1]      # [B, L, rate, 1]
        H_res_raw = alpha[..., 1:]      # [B, L, rate, rate]
        # --- Key Fix 3: Non-negative constraint for H_pre (paper Equation 8) ---
        H_pre = torch.sigmoid(H_pre_raw)
        
        # --- Key Fix 4: Sinkhorn projection of H_res onto bistochastic manifold ---
        H_res = sinkhorn_knopp(H_res_raw, num_iter=20, epsilon=1e-12)

        # --- Key Fix 5: Non-negative constraint for H_post (beta) (paper Equation 8) ---
        # Paper Equation (8) explicitly states H_post = 2 * sigmoid(~H_post)
        beta_constrained = 2.0 * torch.sigmoid(beta)
        # --- mHC Core Constraints End ---

        # Apply H_pre and H_res
        # h: [B, L, rate, dim]
        # h_for_layer: H_pre^T @ h -> [B, L, 1, dim]
        h_for_layer = torch.matmul(H_pre.transpose(-2, -1), h)
        # h_res_flow: H_res^T @ h -> [B, L, rate, dim]
        h_res_flow = torch.matmul(H_res.transpose(-2, -1), h)
        
        # Concatenate: mix_h[..., 0, :] is input to Layer F, mix_h[..., 1:, :] are residual flows
        mix_h = torch.cat([h_for_layer, h_res_flow], dim=-2) # [B, L, rate+1, dim]
        
        return mix_h, beta_constrained, H_res, H_pre

    def depth_connection(self, mix_h: torch.Tensor, h_o: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """
        Perform depth connection step (Equation 3 in mHC paper).
        
        This step combines the residual flows with the transformed output from
        layer F to produce the next layer's hidden states.
        
        Args:
            mix_h (torch.Tensor): Mixed hidden states from width_connection.
                                 Shape: [B, L, rate+1, dim]
            h_o (torch.Tensor): Output from layer F.
                               Shape: [B, L, dim]
            beta (torch.Tensor): H_post matrix (constrained).
                                Shape: [B, L, rate]
        
        Returns:
            torch.Tensor: Next layer hidden states.
                         Shape: [B, L, rate, dim]
        
        Note:
            - Implements: x^{l+1} = h_res_flow + (H_post)^T @ h_o
            - Uses Einstein summation for efficient batch matrix multiplication.
            - The result forms the input for the next transformer layer.
        """
        # mix_h: [B, L, rate+1, dim]
        h_prime = mix_h[..., 1:, :] # [B, L, rate, dim] - residual flow part
        # h_o: Output of Layer F [B, L, dim]
        # Corresponds to paper Equation (3): x^{l+1} = H_res^l x^l + (H_post^l)^T F(...)
        # h_o_weighted: (H_post^l)^T F(...) -> [B, L, rate, dim]
        h_o_weighted = torch.einsum('bld,bln->blnd', h_o, beta)
        return h_prime + h_o_weighted

# ------------------------------------------------------------------------------------
# 5. Image Model Building Blocks
# ------------------------------------------------------------------------------------

class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, dim))
    def forward(self, x):
        return self.net(x)

class ImageHyperConnectionBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, rate: int, layer_id: int, dropout: float = 0.1, dynamic: bool = False):
        super().__init__()
        self.dynamic = dynamic
        self.attn_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)
        self.attention = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.spatial_branch = Mamba(
            d_model=dim,
            d_state=16,
            d_conv=4,
            expand=2,
        )
        self.spectral_branch = Mamba(
            d_model=dim//4,
            d_state=16,
            d_conv=4,
            expand=2,
        )
        self.ffn = FeedForward(dim, dim * 4)
        self.attn_hc = HyperConnection(dim, rate, layer_id, dynamic=self.dynamic)
        self.ffn_hc = HyperConnection(dim, rate, layer_id, dynamic=self.dynamic)
        self.attn_dropout = nn.Dropout(dropout)
        self.ffn_dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, mask=None) -> torch.Tensor:
        mix_h_attn, beta_attn, alpha, H_pre = self.attn_hc.width_connection(h)
        h_in_attn = self.attn_norm(mix_h_attn[..., 0, :])
        
        # Use Mamba for spatial processing
        attn_output = self.spatial_branch(h_in_attn)
        
        # Spectral processing - reshape to use spectral Mamba
        B, HW, C = h_in_attn.shape
        x_flat = h_in_attn.view(B * HW, 4, C//4)
        spectral_output = self.spectral_branch(x_flat)
        spectral_output = spectral_output.view(B, HW, C)
        
        # Combine spatial and spectral outputs
        combined_output = attn_output + spectral_output
        
        h = self.attn_hc.depth_connection(mix_h_attn, self.attn_dropout(combined_output), beta_attn)
        mix_h_ffn, beta_ffn, alpha, H_pre = self.ffn_hc.width_connection(h)
        h_in_ffn = self.ffn_norm(mix_h_ffn[..., 0, :])
        ffn_output = self.ffn(h_in_ffn)
        h = self.ffn_hc.depth_connection(mix_h_ffn, self.ffn_dropout(ffn_output), beta_ffn)
        return h, alpha, beta_ffn, H_pre

    def get_endmember(self):
        query_embed_weight_split = torch.chunk(self.ffn_hc.query_embed.weight, self.ffn_hc.num_endm, dim=0)
        query_embed_weight_split = torch.stack(query_embed_weight_split)
        endmember_get = self.ffn_hc.weights.unsqueeze(-1).repeat(1, 1, 200) * query_embed_weight_split
        endmember_get = torch.mean(endmember_get, dim=1)
        return endmember_get

# ------------------------------------------------------------------------------------
# 6. Image Model 
# ------------------------------------------------------------------------------------
class ImageHyperConnectionTransformer_spec_spa(nn.Module):
    def __init__(self, image_size: int, patch_size: int, in_channels: int, num_classes: int, 
                 dim: int, n_layers: int, n_heads: int, rate: int, dropout: float = 0.1, 
                 drop_path: float = 0, pool_size=4, mask_ratio=0.0, dynamic=True):
        super().__init__()
        self.mask_ratio = mask_ratio
        if dim % 2 != 0:
            raise ValueError(f"dim must be even for SinusoidalPositionEncoding, got {dim}")
        if type(image_size)==int: 
            image_size = (image_size, image_size)
        if type(patch_size)==int: 
            patch_size = (patch_size, patch_size)
        if type(pool_size)==int: 
            pool_size = (pool_size, pool_size)
        
        # Store original image size for upsampling reference
        self.original_image_size = image_size
        self.patch_size = patch_size
            
        self.expansion_rate = rate
        self.width = image_size[1] // patch_size[1]  # Width after patch embedding
        self.height = image_size[0] // patch_size[0]  # Height after patch embedding
        self.num_patches = self.width * self.height
        self.dynamic = dynamic
        self.mseloss = nn.MSELoss(reduction='mean')
        self.num_layers = n_layers

        # Position encoding based on patch grid size (after downsampling)
        pos_dim = dim // 2
        self.pos_y = SinusoidalPositionEncoding(d_model=pos_dim, max_len=self.height)
        self.pos_x = SinusoidalPositionEncoding(d_model=pos_dim, max_len=self.width)
        pos_y = torch.arange(self.height, dtype=torch.float32).unsqueeze(1).repeat(1, self.width)
        pos_x = torch.arange(self.width, dtype=torch.float32).unsqueeze(0).repeat(self.height, 1)
        pos_y = pos_y.flatten().unsqueeze(0)
        pos_x = pos_x.flatten().unsqueeze(0)
        pos_embed = torch.cat([self.pos_y(pos_y), self.pos_x(pos_x)], dim=-1)
        self.register_buffer('pos_embed', pos_embed, persistent=False)

        # Patch embedding (downsamples if patch_size > 1)
        self.patch_embed = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)
        self.emb_dropout = nn.Dropout(dropout)

        # Transformer layers
        self.layers = nn.ModuleList([
            ImageHyperConnectionBlock(
                dim, n_heads, rate, layer_id, dropout, dynamic=self.dynamic
            ) for layer_id in range(n_layers)])

        # Drop path
        if n_layers > 0 and drop_path > 0:
            dpr = torch.linspace(0, drop_path, n_layers).tolist()
        else:
            dpr = [0.0 for _ in range(n_layers)]
        self.drop_paths = nn.ModuleList([
            DropPath(dpr[layer_id]) if dpr[layer_id] > 0.0 else nn.Identity()
            for layer_id in range(n_layers)
        ])
        
        # Final layers
        self.final_norm = RMSNorm(dim)
        self.classifier = nn.Linear(dim, num_classes)
        
        # Classification head
        # Enhanced classification head with skip connection
        self.cls_head = nn.Sequential(
            # First convolution block
            nn.Conv2d(dim, dim * 2, kernel_size=3, padding=1),
            nn.GroupNorm(min(8, dim * 2), dim * 2),
            nn.GELU(),
            nn.Dropout2d(dropout * 0.5),
            
            # Second convolution block
            nn.Conv2d(dim * 2, dim * 4, kernel_size=3, padding=1),
            nn.GroupNorm(min(8, dim * 4), dim * 4),
            nn.GELU(),
            nn.Dropout2d(dropout * 0.3),
            
            # Final projection
            nn.Conv2d(dim * 4, num_classes, kernel_size=1)
        )
        
        # Better upsampling: use transposed convolution or pixel shuffle
        if patch_size[0] > 1 or patch_size[1] > 1:
            # Option 1: Transposed convolution (learnable upsampling)
            self.upsample = nn.Sequential(
                nn.ConvTranspose2d(dim, dim, kernel_size=patch_size, stride=patch_size),
                nn.GroupNorm(min(8, dim), dim),
                nn.GELU(),
            )
            # Option 2: Pixel shuffle (sub-pixel convolution)
            # self.upsample = nn.Sequential(
            #     nn.Conv2d(dim, dim * (patch_size[0] * patch_size[1]), kernel_size=3, padding=1),
            #     nn.PixelShuffle(patch_size[0]),
            #     nn.GroupNorm(min(8, dim), dim),
            #     nn.GELU(),
            # )
        else:
            self.upsample = nn.Identity()
        
        self.apply(init_transformer_weights)
    
    def random_masking(self, x, mask_ratio=0.1):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.size()  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        # ascend: small is keep, large is remove
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore
    
    def forward(self, x: torch.Tensor, epoch=None, return_features=False):
    # Store original input size
        original_spatial_size = x.shape[-2:]
        
        # Apply patch embedding (downsamples if patch_size > 1)
        x = self.patch_embed(x).flatten(2).transpose(1, 2) 
        x = x + self.pos_embed.to(device=x.device, dtype=x.dtype)
        x = self.emb_dropout(x)
        
        B, L, C = x.shape
        Y = []
        ids_restore = None
        
        # if self.training and self.mask_ratio > 0:
        #     x, _, ids_restore = self.random_masking(x)
        #     B, L, C = x.shape
            
        H = x.unsqueeze(2).repeat(1, 1, self.expansion_rate, 1)
        
        for layer, dp in zip(self.layers, self.drop_paths):
            H, alpha, beta_ffn, H_pre = dp(layer(H))
            y = H.mean(dim=2).permute(0, 2, 1).contiguous()
            
            if self.mask_ratio == 0 or not self.training or ids_restore is None:
                Y.append(y.view(B, C, self.width, self.height))
            else:
                # Reconstruct the full-length token sequence
                len_keep = y.shape[-1]
                L_full = self.num_patches
                ids_shuffle = torch.argsort(ids_restore, dim=1)
                ids_keep = ids_shuffle[:, :len_keep]
                y_full = torch.zeros(B, C, L_full, device=y.device, dtype=y.dtype)
                y_full.scatter_(2, ids_keep.unsqueeze(1).expand(-1, C, -1), y)
                Y.append(y_full.view(B, C, self.width, self.height))
                
        # Final processing
        h_final = H.mean(dim=2)
        
        if self.training and self.mask_ratio > 0 and ids_restore is not None:
            y_tokens = h_final.permute(0, 2, 1).contiguous()
            len_keep = y_tokens.shape[-1]
            L_full = self.num_patches
            ids_shuffle = torch.argsort(ids_restore, dim=1)
            ids_keep = ids_shuffle[:, :len_keep]
            y_full = torch.zeros(B, C, L_full, device=y_tokens.device, dtype=y_tokens.dtype)
            y_full.scatter_(2, ids_keep.unsqueeze(1).expand(-1, C, -1), y_tokens)
            h_final = y_full.permute(0, 2, 1).contiguous()

        h_final = self.final_norm(h_final)
        B, C = h_final.shape[0], h_final.shape[-1]
        h_final = h_final.view(B, self.height, self.width, C).permute(0, 3, 1, 2).contiguous()
        
        # Upsample to original spatial size if patch_size > 1 (BEFORE classification head)
        h_final = self.upsample(h_final)
        
        # Ensure feature maps match original input spatial dimensions
        if h_final.shape[-2:] != original_spatial_size:
            h_final = F.interpolate(
                h_final, 
                size=original_spatial_size, 
                mode='bilinear', 
                align_corners=True
            )
        
        # Apply classification head on upsampled features
        logits = self.cls_head(h_final)
        
        Y.append(logits)
        
        if return_features: 
            return Y
        return logits

# ------------------------------------------------------------------------------------
# 7. Testing Functions
# ------------------------------------------------------------------------------------

def load_image_label_pair(image_path, label_path):
    """Load an image-label pair"""
    # Load image
    try:
        img = tifffile.imread(image_path)
    except:
        img = np.array(Image.open(image_path))
    
    # Load label
    try:
        label = tifffile.imread(label_path)
    except:
        label = np.array(Image.open(label_path))
    
    # Ensure correct shape
    if len(img.shape) == 3:
        if img.shape[0] == 64:  # [C, H, W]
            pass
        elif img.shape[2] == 64:  # [H, W, C] -> [C, H, W]
            img = np.transpose(img, (2, 0, 1))
    
    # Convert to tensors
    img_tensor = torch.from_numpy(img).float() / 255.0
    label_tensor = torch.from_numpy(label).long()
    
    return img_tensor, label_tensor

def test_mhc_training(patch_size = 2, dim = 64, n_layers = 3):
    """Test mHC model in training mode with real data"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Paths
    image_dir = Path("/beluga/Hackathon15_AlphaEarth_Alberta/Hackathon15_AlphaEarth_Alberta/train_val_test_patches/patches/train/alphaearth/img")
    label_dir = Path("/beluga/Hackathon15_AlphaEarth_Alberta/Hackathon15_AlphaEarth_Alberta/train_val_test_patches/patches/train/labels/filtered")
    
    # Find matching image-label pairs
    image_files = sorted(list(image_dir.glob("*.tif")))
    label_files = sorted(list(label_dir.glob("*.tif")))
    
    print(f"Found {len(image_files)} images and {len(label_files)} labels")
    
    # Take first 3 pairs
    test_pairs = []
    for i in range(min(3, len(image_files))):
        img_path = image_files[i]
        # Find corresponding label (same filename)
        label_path = label_dir / img_path.name
        if label_path.exists():
            test_pairs.append((img_path, label_path))
            print(f"Pair {i+1}: {img_path.name}")
    
    if not test_pairs:
        print("No matching image-label pairs found! Using random data for testing.")
        # Create random data for testing
        batch_images = torch.randn(3, 64, 224, 224)
        batch_labels = torch.randint(0, 13, (3, 224, 224))
        batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)
    else:
        # Load 3 image-label pairs
        images = []
        labels = []
        
        for img_path, label_path in test_pairs:
            img_tensor, label_tensor = load_image_label_pair(img_path, label_path)
            images.append(img_tensor)
            labels.append(label_tensor)
            print(f"Image shape: {img_tensor.shape}, Label shape: {label_tensor.shape}")
        
        # Stack into batch
        batch_images = torch.stack(images)  # [3, 64, 224, 224]
        batch_labels = torch.stack(labels)  # [3, 224, 224]
        batch_images = batch_images.to(device)
        batch_labels = batch_labels.to(device)
    
    print(f"\nBatch images shape: {batch_images.shape}")
    print(f"Batch labels shape: {batch_labels.shape}")
    print(f"Images range: [{batch_images.min():.3f}, {batch_images.max():.3f}]")
    print(f"Labels unique values: {torch.unique(batch_labels)}")
    
    # Create mHC model
    model = ImageHyperConnectionTransformer_spec_spa(
        image_size=224,
        patch_size=patch_size,
        in_channels=64,
        num_classes=13,
        dim=dim,
        n_layers=n_layers,
        n_heads=4,
        rate=4,
        dropout=0.1,
        drop_path=0.0,  # Important: set to 0.0 to avoid DropPath issues
        mask_ratio=0.0,
        dynamic=True
    )
    
    model = model.to(device)
    
    # Test 1: Forward pass in training mode
    print("\n" + "="*60)
    print("TEST 1: Forward pass (training mode)")
    print("="*60)
    
    model.train()
    
    with torch.no_grad():
        outputs = model(batch_images)
        print(f"Outputs shape: {outputs.shape}")
        print(f"Outputs range: [{outputs.min():.3f}, {outputs.max():.3f}]")
        print(f"✓ Forward pass successful!")
    
    # Test 2: Loss calculation and backward pass
    print("\n" + "="*60)
    print("TEST 2: Loss + Backward pass")
    print("="*60)
    
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-99)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Clear any existing gradients
    optimizer.zero_grad()
    
    # Forward pass
    outputs = model(batch_images)
    loss = criterion(outputs, batch_labels)
    print(f"Loss value: {loss.item():.4f}")
    
    # Backward pass
    try:
        loss.backward()
        print("✓ Backward pass successful!")
        
        # Check if gradients were computed
        has_gradients = False
        for name, param in model.named_parameters():
            if param.grad is not None:
                has_gradients = True
                break
        
        if has_gradients:
            print("✓ Gradients were computed")
        else:
            print("⚠ No gradients computed - check model architecture")
        
        # Optimizer step
        optimizer.step()
        print("✓ Optimizer step successful!")
        
    except Exception as e:
        print(f"✗ Error during backward pass: {type(e).__name__}: {e}")
        print("\nDebug information:")
        print(f"Model training mode: {model.training}")
        print(f"Batch images device: {batch_images.device}")
        print(f"Model device: {next(model.parameters()).device}")
        return False
    
    # Test 3: Multiple training steps
    print("\n" + "="*60)
    print("TEST 3: Multiple training steps")
    print("="*60)
    
    model.train()
    losses = []
    
    for step in range(3):
        optimizer.zero_grad()
        outputs = model(batch_images)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        print(f"Step {step+1}: Loss = {loss.item():.4f}")
    
    print(f"Loss progression: {losses}")
    
    # Check if training is working
    if len(losses) >= 2 and losses[-1] < losses[0]:
        print("✓ Loss is decreasing - training is working!")
    else:
        print("⚠ Loss not decreasing - check learning rate or model")
    
    print("\n" + "="*60)
    print("TEST COMPLETE!")
    print("="*60)
    
    return True

# 
def print_model_summary(model, input_size=(64, 224, 224), device='cuda'):
    """
    Print detailed summary of the model including parameters, layers, and memory usage.
    
    Args:
        model: The PyTorch model to summarize
        input_size: Input tensor size (channels, height, width)
        device: Device to run summary on ('cuda' or 'cpu')
    """
    print("\n" + "="*80)
    print("MODEL SUMMARY")
    print("="*80)
    
    # Method 1: torchsummary (simple layer-wise summary)
    print("\n--- torchsummary Summary ---")
    try:
        summary(model, input_size=input_size, device=device)
    except Exception as e:
        print(f"torchsummary failed: {e}")
    
    # Method 2: torchinfo (more detailed)
    print("\n--- torchinfo Detailed Summary ---")
    try:
        batch_size = 2
        summary_result = torchinfo_summary(
            model, 
            input_size=(batch_size, *input_size),
            verbose=1,
            col_names=["input_size", "output_size", "num_params", "kernel_size", "trainable"],
            col_width=20,
            row_settings=["var_names"]
        )
        print(summary_result)
    except Exception as e:
        print(f"torchinfo failed: {e}")
    
    # Method 3: Custom parameter count
    print("\n--- Custom Parameter Analysis ---")
    total_params = 0
    trainable_params = 0
    layer_info = []
    
    for name, param in model.named_parameters():
        params = param.numel()
        total_params += params
        if param.requires_grad:
            trainable_params += params
        layer_info.append({
            'name': name,
            'shape': tuple(param.shape),
            'params': params,
            'trainable': param.requires_grad
        })
    
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Non-trainable Parameters: {total_params - trainable_params:,}")
    print(f"Model Size: {total_params * 4 / (1024**2):.2f} MB (assuming float32)")
    print("="*80)

# ------------------------------------------------------------------------------------
# 9. Test Cases
# ------------------------------------------------------------------------------------
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device used: {device}")
    
    # --- Basic Image Model Test ---
    image_size = 224
    in_channels = 64
    num_classes= 13
    patch_size = 2
    dim = 64
    n_layers = 4
    n_heads = 4
    img_model = ImageHyperConnectionTransformer_spec_spa(
        image_size=image_size, patch_size=patch_size, in_channels=in_channels, num_classes=num_classes,
        dim=dim, n_layers=n_layers, n_heads=n_heads, rate=4, dropout=0.1
    )
    img_model = img_model.to(device)
    
    print("\n" + "=" * 60)
    print("Basic Image Hyper-Connections Transformer Model Test")
    print("=" * 60)
    
    # Run model with dummy data
    dummy_img = torch.randn(1, in_channels, image_size, image_size, device=device)
    logits = img_model(dummy_img)
    print(f"Logits shape: {logits.shape}")
    assert logits.shape == (1, num_classes, image_size, image_size), "Image model output shape incorrect!"
    print(f"Output logits shape: {logits.shape} (test passed)")
    loss = logits.sum(); loss.backward()
    print("Image model forward and backward propagation test passed.")
    print("-" * 60)

    # --- Real Training Test ---
    print("\n" + "="*60)
    print("Real Training Test")
    print("="*60)
    success = test_mhc_training()
    
    if success:
        print("\n✅ Model passed all training tests!")
        print("Ready for full training pipeline.")
    else:
        print("\n❌ Model failed training tests.")
        print("Need to investigate the issue.")
    

    print("\n" + "="*80)
    print("MODEL ARCHITECTURE ANALYSIS")
    print("="*80)
    
    # Create a model instance for summary
    model_for_summary = ImageHyperConnectionTransformer_spec_spa(
        image_size=224,
        patch_size=patch_size,
        in_channels=64,
        num_classes=13,
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        rate=4,
        dropout=0.1,
        drop_path=0.0,
        mask_ratio=0.0,
        dynamic=True
    ).to(device)
    
    # Print the model summary
    print_model_summary(model_for_summary, input_size=(64, 224, 224), device=device)
    
    # Also print the model structure in a simple way
    print("\n--- Model Structure (Simplified) ---")
    print(model_for_summary)