import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

class FeatureFusion(nn.Module):
    """
    Transformer-based feature fusion module
    Input: G [B, spatial_dim, p] (spatial features), L [B, spectral_dim, p] (spectral features)
    Output: Gl [B, spatial_dim, p], Lg [B, spectral_dim, p] (fused features)
    
    Formulas:
    G_hat = T.transpose x L
    L_hat = T x G
    For Gl: w1@G as Q, w2@G as K, G_hat@w6 as V
    For Lg: L_hat@w5 as Q, w3@L.transpose as K, G@w4 as V
    """
    def __init__(self, p, spatial_dim=4, spectral_dim=128*128, d_model=256, nhead=8, num_layers=2, dropout=0.1):
        super(FeatureFusion, self).__init__()
        self.p = p
        self.spatial_dim = spatial_dim
        self.spectral_dim = spectral_dim
        self.d_model = d_model
        
        # Feature transformation matrix T: [spectral_dim, spatial_dim]
        self.T = nn.Parameter(torch.randn(spectral_dim, spatial_dim))
        
        # Attention weight matrices
        self.w1 = nn.Parameter(torch.ones(p, p))  # For Gl Q: w1@G
        self.w2 = nn.Parameter(torch.ones(p, p))  # For Gl K: w2@G
        self.w3 = nn.Parameter(torch.ones(p, p))  # For Lg K: w3@L.transpose
        self.w4 = nn.Parameter(torch.ones(p, p))  # For Lg V: G@w4
        self.w5 = nn.Parameter(torch.ones(p, p))  # For Lg Q: L_hat@w5
        self.w6 = nn.Parameter(torch.ones(p, p))  # For Gl V: G_hat@w6
        
        # Feature projection layers
        self.spatial_proj = nn.Linear(spatial_dim, d_model)
        self.spectral_proj = nn.Linear(spectral_dim, d_model)
        self.spectral_transpose_proj = nn.Linear(spectral_dim, d_model)  # For Q_lg which is [B, p, spectral_dim]
        
        # Positional encoding
        self.pos_encoder = nn.Parameter(torch.zeros(1, p, d_model), requires_grad=False)
        
        # Multi-head attention for Gl and Lg
        self.attention_gl = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        
        self.attention_lg = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        
        # Transformer encoders for additional processing
        encoder_layer_gl = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_gl = nn.TransformerEncoder(encoder_layer_gl, num_layers=num_layers)
        
        encoder_layer_lg = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_lg = nn.TransformerEncoder(encoder_layer_lg, num_layers=num_layers)
        
        # Output projection layers
        self.spatial_out = nn.Linear(d_model, spatial_dim)
        self.spectral_out = nn.Linear(d_model, spectral_dim)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights"""
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.zeros_(param)

        pos_embed = get_2d_sincos_pos_embed(self.pos_encoder.shape[-1], int(self.p ** 0.5))
        self.pos_encoder.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
    
    def forward(self, G, L, T):
        """
        Args:
            G: Spatial features [B, spatial_dim, p]
            L: Spectral features [B, spectral_dim, p]
        Returns:
            Gl: Fused spatial features [B, spatial_dim, p]
            Lg: Fused spectral features [B, spectral_dim, p]
        """
        B = G.size(0)
        
        # Feature transformation: G_hat = T.transpose x L, L_hat = T x G
        # G_hat: [B, spatial_dim, p] = [B, spatial_dim, spectral_dim] @ [B, spectral_dim, p]
        G_hat = torch.bmm(T.permute(0, 2, 1).float(), L)
        
        # L_hat: [B, spectral_dim, p] = [B, spectral_dim, spatial_dim] @ [B, spatial_dim, p]  
        L_hat = torch.bmm(T.float(), G)
        
        # For Gl: w1@G as Q, w2@G as K, G_hat@w6 as V
        # Q: w1@G -> [B, spatial_dim, p] @ [B, p, p] = [B, spatial_dim, p]
        Q_gl = torch.bmm(G, self.w1.unsqueeze(0).expand(B, -1, -1))
        
        # K: w2@G -> [B, spatial_dim, p] @ [B, p, p] = [B, spatial_dim, p]
        K_gl = torch.bmm(G, self.w2.unsqueeze(0).expand(B, -1, -1))
        
        # V: G_hat@w6 -> [B, spatial_dim, p] @ [B, p, p] = [B, spatial_dim, p]
        V_gl = torch.bmm(G_hat, self.w6.unsqueeze(0).expand(B, -1, -1))
        
        # For Lg: L_hat@w5 as Q, w3@L.transpose as K, G@w4 as V
        # Q: L_hat@w5 -> [B, spectral_dim, p] @ [B, p, p] = [B, spectral_dim, p]
        Q_lg = torch.bmm(L_hat, self.w5.unsqueeze(0).expand(B, -1, -1))
        
        # K: w3@L.transpose -> [B, p, p] @ [B, p, spectral_dim] = [B, p, spectral_dim]
        K_lg = torch.bmm(self.w3.unsqueeze(0).expand(B, -1, -1), L.transpose(1, 2))
        
        # V: G@w4 -> [B, spatial_dim, p] @ [B, p, p] = [B, spatial_dim, p]
        V_lg = torch.bmm(G, self.w4.unsqueeze(0).expand(B, -1, -1))
        
        # Project to d_model dimension for attention
        Q_gl_proj = self.spatial_proj(Q_gl.transpose(1, 2))  # [B, p, d_model]
        K_gl_proj = self.spatial_proj(K_gl.transpose(1, 2))  # [B, p, d_model]
        V_gl_proj = self.spatial_proj(V_gl.transpose(1, 2))  # [B, p, d_model]
        
        # Q_lg: [B, spectral_dim, p], K_lg: [B, p, spectral_dim], V_lg: [B, spatial_dim, p]
        Q_lg_proj = self.spectral_proj(Q_lg.transpose(1, 2))  # [B, spectral_dim, p] -> [B, p, spectral_dim] -> [B, p, d_model]
        K_lg_proj = self.spectral_transpose_proj(K_lg)  # [B, p, spectral_dim] -> [B, p, d_model]
        V_lg_proj = self.spatial_proj(V_lg.transpose(1, 2))  # [B, spatial_dim, p] -> [B, p, spatial_dim] -> [B, p, d_model]
        
        # Add positional encoding
        Q_gl_proj = Q_gl_proj + self.pos_encoder
        K_gl_proj = K_gl_proj + self.pos_encoder
        V_gl_proj = V_gl_proj + self.pos_encoder
        
        Q_lg_proj = Q_lg_proj + self.pos_encoder
        K_lg_proj = K_lg_proj + self.pos_encoder
        V_lg_proj = V_lg_proj + self.pos_encoder
        
        # Attention for Gl
        Gl_attended, _ = self.attention_gl(
            query=Q_gl_proj,
            key=K_gl_proj,
            value=V_gl_proj
        )
        Gl_attended = self.norm1(Gl_attended)
        
        # Attention for Lg
        Lg_attended, _ = self.attention_lg(
            query=Q_lg_proj,
            key=K_lg_proj,
            value=V_lg_proj
        )
        Lg_attended = self.norm2(Lg_attended)
        
        # Project back to original dimensions
        Gl = self.spatial_out(Gl_attended).transpose(1, 2)  # [B, spatial_dim, p]
        Lg = self.spectral_out(Lg_attended).transpose(1, 2)  # [B, spectral_dim, p]
        
        return Gl, Lg

if __name__ == "__main__":
    # Test different configurations
    p = 64
    batch_sizes = [1, 2, 4, 8]
    
    # Test different spatial and spectral dimensions
    configs = [
        {"num_superpixel": 500, "total_pixel": 128*128, "name": "Original (4, 16384)"},
        {"num_superpixel": 500, "total_pixel": 128*128, "name": "Larger spatial (8, 16384)"},
        {"num_superpixel": 500, "total_pixel": 128*128, "name": "Smaller spectral (4, 4096)"},
        {"num_superpixel": 500, "total_pixel": 128*128, "name": "Larger both (16, 65536)"}
    ]
    
    for config in configs:
        print(f"\n{'='*50}")
        print(f"Testing configuration: {config['name']}")
        print(f"spatial_dim={config['total_pixel']}, spectral_dim={config['num_superpixel']}")
        print(f"{'='*50}")
        
        model = FeatureFusion(
            p=p, 
            spatial_dim=config['num_superpixel'],
            spectral_dim=config['total_pixel'],
            d_model=256, 
            nhead=8, 
            num_layers=2, 
            dropout=0.1
        )
        
        for b in batch_sizes:
            print(f"\nTesting batch_size = {b}")
            
            # Input features
            G = torch.randn(b, config['num_superpixel'], p)  # num super pixels
            L = torch.randn(b, config['total_pixel'], p) # total pixels
            
            # Forward pass
            Gl, Lg = model(G, L)
            
            print(f"Input G shape: {G.shape}")
            print(f"Input L shape: {L.shape}")
            print(f"Output Gl shape: {Gl.shape}")
            print(f"Output Lg shape: {Lg.shape}")
            
            # Check gradients
            loss = Gl.sum() + Lg.sum()
            loss.backward()
            print(f"Gradient computation successful")
            
            # Print model parameter statistics
            if b == 1:  # Only print once per config
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                print(f"\nModel Statistics:")
                print(f"Total parameters: {total_params:,}")
                print(f"Trainable parameters: {trainable_params:,}")
                
                # Print T matrix shape
                print(f"Feature transformation matrix T shape: {model.T.shape}")
                print(f"Attention weight matrices shape: {model.w1.shape}")

