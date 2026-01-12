import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from einops import rearrange
import random
import numpy as np
import os
import math
# ==================== REPRODUCIBILITY SETUP ====================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(42)

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Root Mean Square Layer Normalization (RMSNorm).

        RMSNorm normalizes the input tensor along the last dimension using 
        the root mean square (RMS) of the elements instead of the variance 
        (as used in standard LayerNorm). This normalization technique is 
        computationally more efficient and has been used in various transformer-based models.

        Formula:
            RMSNorm(x) = gamma * x / (RMS(x) + eps)
            where RMS(x) = sqrt(mean(x ** 2))

        Args:
            dim (int): The number of features in the input (i.e., size of the last dimension).
            eps (float): A small constant added to the denominator for numerical stability. Default: 1e-6.

        Attributes:
            weight (nn.Parameter): Learnable scaling parameter of shape (dim,).

        Shape:
            - Input: (N, ..., dim)
            - Output: (N, ..., dim) — same shape as input
        """
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = x.norm(2, dim=-1, keepdim=True) # Compute L2 norm of each feature vector
        rms_x = norm_x * (x.shape[-1] ** -0.5)  # Convert L2 norm to RMS value
        return self.gamma * (x / (rms_x + self.eps))
    
class SimplifiedMambaBlock(nn.Module):
    """
    Simplified Mamba Block for Sequence Modeling.

    This block is inspired by the Mamba architecture and combines 
    convolutional processing with a simplified state-space model (SSM)
    to capture both local and long-range dependencies in sequences.

    The architecture includes:
        - RMSNorm normalization on the input
        - Linear projection to an expanded feature space
        - A depthwise 1D convolutional branch to capture local context
        - A state-space recurrence branch to model long-term dependencies
        - A final linear projection and residual connection

    State-Space Model:
        The recurrence is governed by learnable parameters A, B, and C:
            h[t] = sigmoid(A) * h[t-1] + sigmoid(B) * x[t]
            y[t] = sum(sigmoid(C) * h[t])
        where h is the hidden state and x[t] is the input at timestep t.

    Args:
        dim (int): The input and output feature dimension.
        d_state (int): The number of internal SSM states per feature. Default: 16.
        d_conv (int): The kernel size of the depthwise convolution. Default: 4.
        expand (int): Factor to expand the feature dimension internally. Default: 2.

    Attributes:
        norm (RMSNorm): RMS-based normalization layer.
        proj_in (nn.Linear): Linear layer projecting input to expanded dimension.
        proj_out (nn.Linear): Linear layer projecting back to input dimension.
        A (nn.Parameter): Learnable state-transition weights (used in SSM).
        B (nn.Parameter): Learnable input weights (used in SSM).
        C (nn.Parameter): Learnable output weights (used in SSM).
        conv (nn.Conv1d): Depthwise convolutional layer.

    Input Shape:
        (batch_size, sequence_length, dim)

    Output Shape:
        (batch_size, sequence_length, dim)
    """

    def __init__(self, dim: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.expand = expand
        self.expanded_dim = dim * expand
        self.norm = RMSNorm(dim)
        self.proj_in = nn.Linear(dim, self.expanded_dim)
        self.proj_out = nn.Linear(self.expanded_dim, dim)
        # SSM parameters
        self.A = nn.Parameter(torch.zeros(self.expanded_dim, d_state)) # Shape: (expanded_dim, d_state)
        self.B = nn.Parameter(torch.zeros(self.expanded_dim, d_state)) # Shape: (expanded_dim, d_state)
        self.C = nn.Parameter(torch.zeros(self.expanded_dim, d_state)) # Shape: (expanded_dim, d_state)
        # Convolution layer
        self.conv = nn.Conv1d(
            in_channels=self.expanded_dim,
            out_channels=self.expanded_dim,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.expanded_dim, # # Shape: (expanded_dim, d_state)
            bias=False
        )

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.proj_in(x) # Exapnd the input feature
        # Conv branch --> depthwise 1D convolution over the sequence
        x_conv = x.transpose(1, 2) # (B, D, T)
        x_conv = self.conv(x_conv)[..., :x.shape[1]] # Trims output to match input sequence length.
        x_conv = x_conv.transpose(1, 2) # (B, T, D)
        # SSM branch
        batch_size, seq_len, _ = x.shape
        # Initializes hidden state
        h = torch.zeros(batch_size, self.expanded_dim, self.d_state, device=x.device)
        outputs = []
        for t in range(seq_len):
            x_t = x_conv[:, t].unsqueeze(-1) # Get input at time step t
            Bx = torch.sigmoid(self.B) * x_t
            h = torch.sigmoid(self.A.unsqueeze(0)) * h + Bx # Updates hidden state
            out_t = (h * torch.sigmoid(self.C.unsqueeze(0))).sum(-1) # Computes output using parameter C OR Weighted sum of hidden state
            outputs.append(out_t)
        x = torch.stack(outputs, dim=1) # Reassemble the sequence
        x = self.proj_out(x) #  back to original dim
        return x + residual # Add the residual connection
    
class SparseDeformableMambaBlock2(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, sparsity_ratio=0.8):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.expand = expand
        self.expanded_dim = dim * expand
        self.sparsity_ratio = sparsity_ratio

        self.norm = RMSNorm(dim)
        self.proj_in = nn.Linear(dim, self.expanded_dim)
        self.proj_out = nn.Linear(self.expanded_dim, dim)

        self.A = nn.Parameter(torch.zeros(self.expanded_dim, d_state))
        self.D = nn.Parameter(torch.ones(self.expanded_dim))

        self.delta_proj = nn.Linear(self.expanded_dim, self.expanded_dim)
        self.B_proj = nn.Linear(self.expanded_dim, d_state)
        self.C_proj = nn.Linear(self.expanded_dim, d_state)

        self.conv = nn.Conv1d(
            in_channels=self.expanded_dim,
            out_channels=self.expanded_dim,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.expanded_dim,
            bias=False
        )

        # Attention layer for center token generation
        self.attention_proj = nn.Linear(self.expanded_dim, 1)

    def forward(self, x):
        B, L, C = x.shape
        residual = x

        # Reshape to (B, H*W, C) for sequence processing
        x_flat = x

        x_norm = self.norm(x_flat)
        x_proj = self.proj_in(x_norm)

        # Generate attention weights
        attention_logits = self.attention_proj(x_proj).squeeze(-1)  # (B, H*W)
        attention_weights = F.softmax(attention_logits, dim=-1).unsqueeze(-1)  # (B, H*W, 1)

        # Compute weighted average to get the center token (attention-based) --> A learned summary vector
        #    1- Multiply each token in x_proj by its attention weight
        #    2- Then sum all H*W weighted vectors to get one vector
        center_token = torch.sum(x_proj * attention_weights, dim=1, keepdim=True)  # (B, 1, expanded_dim)

        # Broadcast the Center Token so we can compute similarity between each token and the center token.
        center_token = center_token.expand(-1, L, -1)  # (B, H*W, expanded_dim)

        # Compute similarity to the generated center token
        sim = torch.sum(F.normalize(x_proj, p=2, dim=-1) * F.normalize(center_token, p=2, dim=-1), dim=-1)

        # Select top-k most relevant tokens
        k = max(1, int(L * self.sparsity_ratio)) # can we use a learnable ratio?
        # indices of the top-k tokens most similar to the center.
        _, topk_idx = torch.topk(sim, k=k, dim=-1) # (B, k) 

        # Gathers only the top-k tokens (most relevant) from x_proj
        sparse_x = torch.gather(
            x_proj,
            dim=1,
            index=topk_idx.unsqueeze(-1).expand(-1, -1, self.expanded_dim)
        ) # (B, k, expanded_dim)

        # Conv processing on sparse tokens
        sparse_x_conv = sparse_x.transpose(1, 2) #  (B, expanded_dim, k)
        sparse_x_conv = self.conv(sparse_x_conv)[..., :k] # A simple convolution layer is applied to the sparse tokens.
        sparse_x_conv = sparse_x_conv.transpose(1, 2)  # (B, expanded_dim, k)

        # Sparse SSM processing
        delta = F.softplus(self.delta_proj(sparse_x)) # (B, k, d_state)
        A = -torch.exp(self.A) # (expanded_dim, d_state)
        B_sparse = self.B_proj(sparse_x) # (B, k, d_state)
        C_sparse = self.C_proj(sparse_x) # (B, k, d_state)

        deltaA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)) # # (B, k, expanded_dim, d_state)
        deltaB = delta.unsqueeze(-1) * B_sparse.unsqueeze(2) # (B, k, 1, d_state)

        h = torch.zeros(B, self.expanded_dim, self.d_state, device=x.device)
        outputs = []
        for t in range(k):
            h = deltaA[:, t] * h + deltaB[:, t] # update state
            output_t = torch.sum(h * C_sparse[:, t].unsqueeze(1), dim=-1) + self.D * sparse_x[:, t]
            outputs.append(output_t)
        # Stack and Project Output    
        sparse_output = torch.stack(outputs, dim=1) # (B, k, expanded_dim)
        sparse_output_proj = self.proj_out(sparse_output)

        # Scatter the processed sparse tokens back to the original spatial positions
        output_flat = torch.zeros(B, L, C, device=x.device)
        # scatter_ inserts each processed token back into its original location (topk_idx). All other locations remain zero
        output_flat.scatter_(1, topk_idx.unsqueeze(-1).expand(-1, -1, C), sparse_output_proj)
        output = output_flat

        return output + residual
    
class MIMBlock(nn.Module):
    """
    Multi-level Interaction Mamba Block (MIMBlock)

    This module captures and enhances hierarchical feature representations using a dual-branch architecture: 
    one for fine-grained local features ("visual words") and one for coarse-grained global context ("visual sentences").
    It supports bidirectional attention-based feedback and fusion between these two levels.

    Architecture Overview:
    - **Inner Mamba Branch**:
        - Processes fine-grained word-level features extracted from local windows in the input image.
        - Uses a Mamba-based temporal modeling block (`SimplifiedMambaBlock`) followed by LayerNorm and an MLP.
    - **Outer Mamba Branch**:
        - Processes coarse sentence-level features derived by average pooling over local windows.
        - Uses a sparse, deformable Mamba block for efficient and flexible modeling of spatial dependencies.
    - **Cross-Level Interaction**:
        - Bidirectional communication between words and sentences via `MultiheadAttention` layers.
        - Aggregation from words → sentences using a learned projection (`word_to_sentence`).
        - Feedback from sentences → words using `sentence_to_word` projection or attention-based mechanisms.
    
    Args:
        dim (int): Dimension of the sentence-level feature representations.
        inner_dim (int): Dimension of the word-level feature representations.
        d_state (int): State dimension for the Mamba blocks.
        d_conv (int): Convolutional expansion factor inside Mamba blocks.
        expand (int): MLP expansion ratio for the FFNs.

    Inputs:
        visual_words (Tensor): Word-level features of shape (B, N, W, D),
                               where B is the batch size, N is the number of local windows,
                               W is the number of pixels per window, and D is the word dimension.
        visual_sentences (Tensor): Sentence-level features of shape (B, H, W, D_s),
                                   where H and W denote spatial dimensions and D_s is the sentence dimension.

    Returns:
        Tuple[Tensor, Tensor]:
            - Updated visual_words of shape (B, N, W, D) with feedback-enhanced local features.
            - Updated visual_sentences of shape (B, H, W, D_s) with fused contextual information from words.
    """

    def __init__(self, dim, inner_dim, heads, d_state=16, d_conv=4, expand=2):
        super().__init__()
        # Inner Mamba (words) --> Operates on fine-grained features ("words").
        self.inner_mamba = SimplifiedMambaBlock(inner_dim, d_state, d_conv, expand)
        self.inner_norm = nn.LayerNorm(inner_dim)
        # Post-processing for the word features using MLP.
        self.inner_ffn = nn.Sequential(
            nn.Conv2d(inner_dim, inner_dim * 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(inner_dim * 4, inner_dim, kernel_size=1)
        )

        # Outer Mamba (sentences) --> Processes coarser sentence-level representations using sparse attention
        # sentence-level representations typically cover a large spatial area, and sparse attention lets the model focus only on important regions, improving efficiency and performance.
        self.outer_mamba = SparseDeformableMambaBlock2(dim, d_state, d_conv, expand) 
        self.outer_norm = nn.LayerNorm(dim)
        # Post-processing for the sentence features using MLP.
        self.outer_ffn = nn.Sequential(
            nn.Conv2d(dim, dim * 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim * 4, dim, kernel_size=1)
        )

        # Cross-level Connections
        # Added by saied
        self.word_to_sentence_attn = nn.MultiheadAttention(embed_dim=dim, kdim=inner_dim, vdim=inner_dim, num_heads=heads, batch_first=True)
        self.sentence_to_word_attn = nn.MultiheadAttention(embed_dim=inner_dim, kdim=dim, vdim=dim, num_heads=heads, batch_first=True)

        # Bidirectional projections
        self.word_to_sentence = nn.Linear(inner_dim, dim) #  Aggregates word features into sentence-level inputs.
        self.sentence_to_word = nn.Linear(dim, inner_dim)  # Allows sentence-level features to modulate or enhance word features.

    def forward(self, visual_words, visual_sentences):  
        """ 
        Args:
            visual_words (Tensor): (B, N, W, D) —  word-level tokens
            visual_sentences (Tensor): (B, H, W, D_s) — sentence-level tokens

        Returns:
            Tuple[Tensor, Tensor]: Updated (visual_words, visual_sentences)
        """
        #  both visual_words and visual_sentences share the same N = number of local windows
        # num_sentences = number of local windows
        # num_words= number of pixels in a local window
        # word_dim =  numbero f feature that each pixel has
        B, num_sentences, num_words, word_dim = visual_words.shape
        res1 = visual_words
        #print(f"visual_words.shape: B, num_sentences, num_words, word_dim --> {visual_words.shape}")
        # H, W = number of local windows in hieght and width directions
        _, H, W, sentence_dim = visual_sentences.shape
        #print(f"sentence_dim in MIMBlock: {sentence_dim}")
        res2 = visual_sentences

        # -------- Step 1: Process Words with Inner Mamba --------
        words_reshaped = visual_words.reshape(B * num_sentences, num_words, word_dim)
        words_out = self.inner_mamba(self.inner_norm(words_reshaped))
        visual_words = words_out.reshape(B, num_sentences, num_words, word_dim)

        # -------- Step 2: Feedback Word → Sentence via MultiheadAttention --> Sentence features are informed by attending to word features. --------
        # Treat each sentence as a query, and words in that sentence as key/value
        word_tokens = visual_words.reshape(B * num_sentences, num_words, word_dim)  
        sentence_queries = visual_sentences.reshape(B * num_sentences, 1, sentence_dim)
 
        word_to_sentence_out, _ = self.word_to_sentence_attn(
        query=sentence_queries,           # [B*num_sentences, 1, sentence_dim]
        key=word_tokens,                  # [B*num_sentences, num_words, word_dim]
        value=word_tokens                 # [B*num_sentences, num_words, word_dim]
    )
        visual_sentences = visual_sentences.reshape(B * num_sentences, 1, sentence_dim) + word_to_sentence_out
        visual_sentences = visual_sentences.reshape(B, -1, sentence_dim)

        # -------- Step 3: Process Sentences with Outer Mamba --------
        visual_sentences = self.outer_mamba(self.outer_norm(visual_sentences))

        # -------- Step 4: Feedback Sentence → Word via MultiheadAttention --------
        # Each word in a window queries the sentence embedding
        word_queries = visual_words.reshape(B * num_sentences, num_words, word_dim)  # [B*N, W, D]
        sentence_keys = visual_sentences.reshape(B * num_sentences, 1, sentence_dim)  # [B*N, 1, D_s]

        sentence_to_word_out, _ = self.sentence_to_word_attn(
        query=word_queries,              # [B*num_sentences, num_words, word_dim]
        key=sentence_keys,              # [B*num_sentences, 1, sentence_dim]
        value=sentence_keys             # [B*num_sentences, 1, sentence_dim]
    )
        visual_words = visual_words + sentence_to_word_out.reshape(B, num_sentences, num_words, word_dim)

        # -------- Step 5: Residual Connections --------
        visual_words = visual_words + res1

        # Convert visual_sentences back to [1, H, D, sentence_dim] before adding
        visual_sentences = visual_sentences.reshape(B, H, W, sentence_dim)
        visual_sentences = visual_sentences + res2

        return visual_words, visual_sentences
        
class LearnableUpsample(nn.Module):  
    """A learnable upsampling module using transposed convolution.
    
        Performs upsampling with a transposed convolution followed by batch normalization
        and GELU activation. The spatial dimensions are increased by the scale factor
        while channels are transformed to the specified output dimension.

        Args:
            in_channels (int): Number of channels in the input tensor
            out_channels (int): Number of channels in the output tensor
            scale_factor (int or tuple): Scaling factor for spatial dimensions. 
                                If int, same factor used for height and width.
                                If tuple, should be (scale_h, scale_w).

        Input:
            x (torch.Tensor): Input tensor of shape [batch_size, in_channels, height, width]

        Returns:
            torch.Tensor: Output tensor of shape [
            batch_size, 
            out_channels, 
            height * scale_factor, 
            width * scale_factor
            ]

        Example:
            >>> upsample = LearnableUpsample(256, 128, scale_factor=2)
            >>> x = torch.randn(1, 256, 16, 16)
            >>> out = upsample(x)
            >>> print(out.shape)
            torch.Size([1, 128, 32, 32])
    """
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor
        self.upsample = nn.ConvTranspose2d(
            in_channels, 
            out_channels, 
            kernel_size=scale_factor, 
            stride=scale_factor
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        """Forward pass with shape requirements.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, C_in, H, W]
            
        Returns:
            torch.Tensor: Output tensor of shape [B, C_out, H*scale, W*scale]
        """
        x = self.upsample(x)
        x = self.norm(x)
        x = self.act(x)
        return x

class SkipConnectionModule(nn.Module):
    def __init__(self, dim=256, spatial_scale=4):
        super().__init__()
        self.dim = dim
        self.spatial_scale = spatial_scale  # (p1, p2) scaling factors

        self.word_fusion_proj =  nn.Conv2d(2*dim, dim, kernel_size=1)
        self.sentence_fusion_proj =  nn.Conv2d(2*dim, dim, kernel_size=1) 
        
        # # If you want use one fused image instead of fused words nad fused sentences
        # # you must first upsample fused sentence to fused it with the fused_words
        # self.sentences_upsample = nn.Sequential(
        #     nn.ConvTranspose2d(self.dim, self.dim, 
        #                      kernel_size=spatial_scale, 
        #                      stride=spatial_scale),
        #     nn.BatchNorm2d(self.dim),
        #     nn.GELU()
        # )
        # self.fusion_conv = nn.Sequential(
        #     nn.Conv2d(self.dim * 2, self.dim, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(self.dim),
        #     nn.GELU()
        # )

    def forward(self, dec_words, dec_sentences, enc_words, enc_sentences):
        """
        Args:
            dec_words: Decoder words [B, h*w, p1*p2, c]
            dec_sentences: Decoder sentences [B, c, h, w]
            enc_words: Encoder words [B, h*w, p1*p2, c]
            enc_sentences: Encoder sentences [B, c, h, w]
        Returns:
            Fused features [B, c, H, W] where H=h*p1, W=w*p2
        """
        # --- Process words  ---
        fused_words = torch.cat([dec_words, enc_words], dim=-1)  # [B, h*w, p1*p2, 2c]
        # reshape the fused_words to aply conv
        h= w= int(math.sqrt(fused_words.shape[1]))
        fused_words = rearrange(fused_words, 'b (h w) (p1 p2) c -> b c (h p1) (w p2)', h=h, w=w, p1=self.spatial_scale, p2= self.spatial_scale)
        fused_words = self.word_fusion_proj(fused_words)  # [B, c, h*p1, w*p2]
        # return it into b, (hw), (p1p2) c shape
        fused_words = rearrange(fused_words, 'b c (h p1) (w p2)-> b (h w) (p1 p2) c', h=h, w=w, p1=self.spatial_scale, p2= self.spatial_scale)

        # --- Process sentences  ---
        fused_sentences = torch.cat([dec_sentences, enc_sentences], dim=1)  # [B, 2c, h, w]
        fused_sentences = self.sentence_fusion_proj(fused_sentences) # [B, c, h, w]

        return fused_words, fused_sentences

class MIMUNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        in_channels = config['model']['in_channels']
        depths = config['model']['depths']
        self.local_window_size = config['model']['mim']['local_window_size']
        num_classes = config['model']['num_classes']
        self.num_layers = len(depths)

        # Stem
        stem_layers = [
                nn.Conv2d(in_channels,
                        config['model']['stem_dim'],
                        kernel_size=config['model']['stem_kernel'],
                        padding=config['model']['stem_padding']),
                nn.BatchNorm2d(config['model']['stem_dim']),
                nn.GELU(),
            ]

        # Only add pooling if stem_downsampling is True
        if config['model'].get('stem_downsampling', False):
            stem_layers.append(nn.AvgPool2d(kernel_size=2, stride=2))

        self.stem = nn.Sequential(*stem_layers)

        # Channel dimensions per stage
        dims = [config['model']['stem_dim'] * (2 ** i) for i in range(self.num_layers)]
        #print(f"dims: {dims}")
        
        # -------- Encoder  --------

        # Downsampling for feature maps (for initial projection)
        self.feature_downsamples = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dims[i], dims[i+1], 3, stride=2, padding=1),
                nn.BatchNorm2d(dims[i+1]),
                nn.GELU()
            ) for i in range(self.num_layers - 1)
        ])

        # Sentence projection
        self.sentence_projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dims[i], dims[i], 1),
                nn.BatchNorm2d(dims[i]),
                nn.GELU()
            ) for i in range(self.num_layers)
        ])

        # MIM Blocks
        self.stages = nn.ModuleList()
        for i in range(self.num_layers):
            blocks = nn.Sequential(*[
                MIMBlock(
                    dim=dims[i],
                    inner_dim=dims[i],
                    heads=config['model']['mim']['heads'],
                    d_state=config['mamba']['d_state'],
                    d_conv=config['mamba']['d_conv'],
                    expand=config['mamba']['expand']
                ) for _ in range(depths[i])
            ])
            self.stages.append(blocks)

        # -------- Decoder  --------
        # generate the input of the decoer from the last encoder visual words
        # Saied: it can be replaces with the CBAM class in Yimin's code
        self.decoder_words_input = nn.Conv2d(dims[-1], dims[-1], kernel_size=3, stride=1, padding=1)
        self.decoder_sentences_input = nn.Conv2d(dims[-1], dims[-1], kernel_size=1)

        # dims = [32,64,128,256]
        self.decoder_stages = nn.ModuleList()
        decoder_dims = dims[:-1][::-1]  # reverse dims except the last one -> [128, 64, 32]. We skip the last layer because we are using bridge
        decoder_depths = depths[:-1][::-1]  # reverse depths except the last one. We skip the last layer because we are using bridge

        # scale factor is 2 because in self.feature_downsamples we downsample feture map by factor 2
        self.upsample_decoder_words =  nn.ModuleList([ LearnableUpsample(dims[i+1], dims[i], scale_factor= 2) for i in range(self.num_layers - 1)]) 
        self.upsample_decoder_sentences = nn.ModuleList([ LearnableUpsample(dims[i+1], dims[i], scale_factor= 2) for i in range(self.num_layers - 1)]) 
        
        # MIM Blocks
        self.decoder_stages = nn.ModuleList()
        for i in range(self.num_layers-1): # since there is a bridge, we skip one of the layers
            blocks = nn.Sequential(*[
                MIMBlock(
                    dim=dims[i],
                    inner_dim=dims[i],
                    heads=config['model']['mim']['heads'],
                    d_state=config['mamba']['d_state'],
                    d_conv=config['mamba']['d_conv'],
                    expand=config['mamba']['expand']
                ) for _ in range(decoder_depths[i])
            ])
            self.decoder_stages.append(blocks)

        # Skip connections
        self.skip_layers = nn.ModuleList()
        # we will have num_layers -1 skip connections
        for i in range(self.num_layers-1):  # i will be 2,1, and 0
            skip_layer = SkipConnectionModule (dim = dims[i], spatial_scale= self.local_window_size)
            self.skip_layers.append(skip_layer)
         
                # final upsampling for senteces and words
        self.final_up_w = nn.Sequential(nn.ConvTranspose2d(
                    in_channels= dims[0],  
                    out_channels= dims[0],  
                    kernel_size= 2,  
                    stride= 2,  
                    padding= 0  
                ),
                    nn.BatchNorm2d(dims[0]),  
                    nn.GELU(),  
        )

        self.final_up_s = nn.Sequential(
                        nn.ConvTranspose2d(dims[0], dims[0], kernel_size= self.local_window_size, stride= self.local_window_size, padding=0),  # 4× up (8→32)
                        nn.ConvTranspose2d(dims[0], dims[0], kernel_size= self.local_window_size//2, stride= self.local_window_size//2, padding=0),  # Another 4× up (32→128)
                        nn.BatchNorm2d(dims[0]),
                        nn.GELU()
                    )
        self.final_conv = nn.Conv2d(dims[0], num_classes, 1)

    def forward(self, x):
        B = x.size(0)
        x = self.stem(x)        
        word_skips = []
        sentence_skips = []
        visual_words = None
        local_window_size = self.config['model']['mim']['local_window_size']

        # -------------------  Encoder Part  -------------------
        for i in range(self.num_layers):
            _, C, H, W = x.shape
            h = H // self.local_window_size
            w = W // self.local_window_size
            visual_words = rearrange(
            x, 'b c (h p1) (w p2) -> b (h w) (p1 p2) c',
            p1=local_window_size, p2=local_window_size
            )
            avg_pooled = F.avg_pool2d(x, self.local_window_size)
            visual_sentences = self.sentence_projs[i](avg_pooled)
            visual_sentences = visual_sentences.permute(0, 2, 3, 1)
            for block in self.stages[i]:
                visual_words, visual_sentences = block(visual_words, visual_sentences)
            word_skips.append((visual_words, h, w))
            sentence_skips.append(visual_sentences.permute(0, 3, 1, 2))
            if i < self.num_layers - 1:
                # Reshape visual_words back to (B, C, H, W) before downsampling
                x = rearrange(
                    visual_words,
                    'b (h w) (p1 p2) c -> b c (h p1) (w p2)',
                    h=h, w=w, p1=local_window_size, p2=local_window_size
                )
                x = self.feature_downsamples[i](x)
        # -------------------  Decoder Part  -------------------
        # --- Aplly Bridge to start decoder
        last_encoder_visual_words, h, w = word_skips[-1]  # [B, h*w, p1*p1, c]
        last_encoder_visual_words = rearrange(
            visual_words, 'b (h w) (p1 p2) c -> b c (h p1) (w p2)',
            h=h, w=w, p1=local_window_size, p2=local_window_size
        )
        decoder_visual_words = self.decoder_words_input(last_encoder_visual_words) # [B, c, (h p1), (w p2)]
        # reshape decoder_visual_words into [B, h*w, p1*p1, c]
        #decoder_visual_words = rearrange (decoder_visual_words, 'b c (h p1) (w p2) -> b (h w) (p1 p2) c', h=h, w=w, p1=local_window_size, p2=local_window_size)

        last_encoder_visual_sentences = sentence_skips [-1] # [B, c, h, w]
        decoder_visual_sentences = self.decoder_sentences_input(last_encoder_visual_sentences) # [B, c, h, w]
        
        # There are num_layers -1 upsampling process

        for i in range (self.num_layers - 2, -1, -1): # i will be 2, 1 , 0
            # upsample decoder_wrods and decoder_sentences
            # Input tensor upsampling class must be in shape of [b, c, h, w] 
            decoder_visual_sentences = self.upsample_decoder_sentences[i](decoder_visual_sentences) # [B, c_new, h_new, w_new] -> for example for i=2 [B, 256, 2, 2] -> [B, 128, 4, 4]
            
            # since decoder_visual_words is already in the shape of [b, c, h*p1, w*p2], there is no need to change it
            decoder_visual_words = self.upsample_decoder_words[i](decoder_visual_words)
            # reshape the upsampled_decoder_visual_words into [B, h*w, p1*p2, c]
            w= decoder_visual_words.shape[-1]// local_window_size
            h= decoder_visual_words.shape[2]// local_window_size
            decoder_visual_words = rearrange(decoder_visual_words, 'b c (h p1) (w p2) -> b (h w) (p1 p2) c', h=h, w=w, p1=local_window_size, p2= local_window_size)

            # skip the encoder words and sentences
            encoder_visual_words,_,_ = word_skips[i]
            encoder_visual_sentences = sentence_skips[i]

            decoder_visual_words, decoder_visual_sentences = self.skip_layers[i](decoder_visual_words, decoder_visual_sentences, encoder_visual_words, encoder_visual_sentences)

            # Very important : in using MIMBlock, shape of visual_sentence must be [B, h, w, c]
            decoder_visual_sentences = decoder_visual_sentences.permute(0, 2, 3, 1)

            # apply MIMBlock in decoder
            for decode_block in self.decoder_stages[i]:
                decoder_visual_words , decoder_visual_sentences = decode_block(decoder_visual_words , decoder_visual_sentences)

            # we must reshape decoder_visual_sentences from [b, h, w, c] into [b, c, h, w] because of the upsampling class
            decoder_visual_sentences = rearrange(decoder_visual_sentences, 'b h w c -> b c h w')
            # we must reshape decoder_visual_words from [b, h*w, p1*p2, c] into [b, c, h*p1, w*p2] form because of the upsampling class
            decoder_visual_words = rearrange(decoder_visual_words, 'b (h w) (p1 p2) c -> b c (h p1) (w p2)', h=h, w=w, p1=local_window_size, p2= local_window_size)

        #print(f"before upsampling decoder_visual_sentences.shape: {decoder_visual_sentences.shape} -> b c h w")
        #print(f"before upsampling decoder_visual_words.shape: {decoder_visual_words.shape} -> b c (h p1) (w p2)")
        # Upsample 
        if self.config['model']['stem_downsampling']:
            decoder_visual_words = self.final_up_w(decoder_visual_words)
            decoder_visual_sentences = self.final_up_s(decoder_visual_sentences)
           
        # we only use visual_words to classify objects
        decoder_visual_words = self.final_conv(decoder_visual_words)
        return decoder_visual_words