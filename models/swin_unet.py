import torch
import torch.nn as nn
from einops import rearrange
from timm.models.layers import to_2tuple
import math

############################################
# Patch Embedding
############################################
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=32, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = (
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
        )
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.norm = norm_layer(embed_dim) if norm_layer else None

    def forward(self, x):
        B, C, H, W = x.shape
        assert (H, W) == self.img_size, "Input size mismatch"
        x = self.proj(x).flatten(2).transpose(1, 2)
        if self.norm:
            x = self.norm(x)
        return x

############################################
# MLP
############################################
class Mlp(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0.):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

############################################
# Window helpers
############################################
def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

def create_mask(H, W, window_size, shift_size):
    Hp = int(math.ceil(H / window_size)) * window_size
    Wp = int(math.ceil(W / window_size)) * window_size
    img_mask = torch.zeros((1, Hp, Wp, 1))
    h_slices = (slice(0, -window_size),
                slice(-window_size, -shift_size),
                slice(-shift_size, None))
    w_slices = (slice(0, -window_size),
                slice(-window_size, -shift_size),
                slice(-shift_size, None))
    cnt = 0
    for h in h_slices:
        for w in w_slices:
            img_mask[:, h, w, :] = cnt
            cnt += 1
    mask_windows = window_partition(img_mask, window_size)
    mask_windows = mask_windows.view(-1, window_size * window_size)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask

############################################
# Window Attention
############################################
class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2*window_size[0]-1)*(2*window_size[1]-1), num_heads)
        )
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1,2,0).contiguous()
        relative_coords[:,:,0] += window_size[0]-1
        relative_coords[:,:,1] += window_size[1]-1
        relative_coords[:,:,0] *= 2*window_size[1]-1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C//self.num_heads).permute(2,0,3,1,4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = (q @ k.transpose(-2,-1))
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(N,N,-1).permute(2,0,1)
        attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_//nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1,2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

############################################
# Swin Transformer Block
############################################
class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size % window_size if shift_size>0 else 0
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size=(window_size,window_size), num_heads=num_heads,
                                    qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, mlp_ratio=mlp_ratio, drop=drop)
        if self.shift_size > 0:
            H,W = input_resolution
            attn_mask = create_mask(H, W, window_size, shift_size)
            self.register_buffer("attn_mask", attn_mask)  # ✓ Auto moves to device
        else:
            self.attn_mask = None

    def forward(self,x):
        H,W = self.input_resolution
        B,L,C = x.shape
        assert L == H*W, "input feature has wrong size"
        shortcut = x
        x = self.norm1(x).view(B,H,W,C)
        if self.shift_size>0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size,-self.shift_size), dims=(1,2))
        else:
            shifted_x = x
        x_windows = window_partition(shifted_x,self.window_size).view(-1,self.window_size*self.window_size,C)
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        shifted_x = window_reverse(attn_windows.view(-1,self.window_size,self.window_size,C), self.window_size,H,W)
        if self.shift_size>0:
            x = torch.roll(shifted_x, shifts=(self.shift_size,self.shift_size), dims=(1,2))
        else:
            x = shifted_x
        x = x.view(B,H*W,C)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x

############################################
# Basic Layer (stack of Swin blocks)
############################################
class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if i%2==0 else window_size//2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop
            ) for i in range(depth)
        ])
    def forward(self,x):
        for blk in self.blocks:
            x = blk(x)
        return x

############################################
# Patch Merging / Expand
############################################
class PatchMerging(nn.Module):
    def __init__(self,input_resolution,dim,norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4*dim,2*dim,bias=False)
        self.norm = norm_layer(4*dim)
    def forward(self,x):
        H,W = self.input_resolution
        B,L,C = x.shape
        assert L == H*W, "input feature has wrong size"
        x = x.view(B,H,W,C)
        x0 = x[:,0::2,0::2,:]
        x1 = x[:,1::2,0::2,:]
        x2 = x[:,0::2,1::2,:]
        x3 = x[:,1::2,1::2,:]
        x = torch.cat([x0,x1,x2,x3],-1)
        x = x.view(B,-1,4*C)
        x = self.norm(x)
        x = self.reduction(x)
        return x
class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, (dim_scale**2) * dim // dim_scale, bias=False) if dim_scale > 1 else nn.Identity()
        self.output_dim = dim // dim_scale
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        x = self.expand(x)
        if self.dim_scale > 1:
            x = rearrange(x, 'b (h w) (p1 p2 c) -> b (h p1) (w p2) c', 
                          h=H, w=W, p1=self.dim_scale, p2=self.dim_scale, c=self.output_dim)
        x = x.view(B, -1, self.output_dim)
        x = self.norm(x)
        return x

class FinalPatchExpand_X4(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 16*dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        H,W = self.input_resolution
        B,L,C = x.shape
        assert L == H*W, "input feature has wrong size"
        x = self.expand(x)
        x = rearrange(x, 'b (h w) (p1 p2 c) -> b (h p1) (w p2) c', 
                      h=H, w=W, p1=4, p2=4, c=self.output_dim)
        x = x.view(B, -1, self.output_dim)
        x = self.norm(x)
        return x

############################################
# SwinUNet
############################################
class SwinUNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        m = config['model']
        swin = m['swin']
        self.embed_dim = swin['embed_dim']
        self.depths = m['depths']
        self.num_layers = len(self.depths)
        self.window_size = swin.get('window_size',7)
        self.patch_embed = PatchEmbed(img_size=m['img_size'], patch_size=swin['patch_size'],
                                      in_chans=m['in_channels'], embed_dim=self.embed_dim,
                                      norm_layer=nn.LayerNorm)
        res = self.patch_embed.patches_resolution

        # Encoder
        self.encoder = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        for i in range(self.num_layers):
            dim = self.embed_dim * (2**i)
            input_res = (res[0] // (2**i), res[1] // (2**i))
            layer = BasicLayer(dim=dim, input_resolution=input_res, depth=self.depths[i],
                               num_heads=m['heads'][i], window_size=self.window_size)
            self.encoder.append(layer)
            
            if i < self.num_layers - 1:
                downsample = PatchMerging(input_res, dim)
                self.downsample_layers.append(downsample)

        # Bottleneck
        bottleneck_dim = self.embed_dim * (2**(self.num_layers-1))
        bottleneck_res = (res[0] // (2**(self.num_layers-1)), res[1] // (2**(self.num_layers-1)))
        self.bottleneck = BasicLayer(dim=bottleneck_dim, input_resolution=bottleneck_res,
                                     depth=self.depths[-1], num_heads=m['heads'][-1],
                                     window_size=self.window_size)

        # Decoder
        self.decoder = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()
        self.skip_proj = nn.ModuleList()
        
        for i in range(self.num_layers-1, 0, -1):
            # Current resolution (before upsampling)
            curr_res = (res[0] // (2**i), res[1] // (2**i))
            # Target resolution (after upsampling)
            target_res = (res[0] // (2**(i-1)), res[1] // (2**(i-1)))
            
            # Upsample layer
            dim = self.embed_dim * (2**i)
            upsample = PatchExpand(input_resolution=curr_res, dim=dim, dim_scale=2)
            self.upsample_layers.append(upsample)
            
            # Decoder blocks - use concatenation approach
            decoder_dim = dim // 2
            skip_dim = decoder_dim  # Skip connection has same dimension as decoder_dim
            concat_dim = decoder_dim * 2  # After concatenation
            
            # Projection to reduce concatenated features back to decoder_dim
            self.skip_proj.append(nn.Linear(concat_dim, decoder_dim))
            
            layer = BasicLayer(dim=decoder_dim, input_resolution=target_res,
                               depth=self.depths[i-1], num_heads=m['heads'][i-1],
                               window_size=self.window_size)
            self.decoder.append(layer)

        # Final upsampling
        self.final_up = FinalPatchExpand_X4(input_resolution=res, dim=self.embed_dim)
        self.output = nn.Conv2d(self.embed_dim, m['num_classes'], 1)

    def forward(self, x):
        # Encoder
        x = self.patch_embed(x)
        skips = []
        
        for i in range(self.num_layers):
            x = self.encoder[i](x)
            if i < self.num_layers - 1:  # Don't save the last encoder output as skip
                skips.append(x)
                x = self.downsample_layers[i](x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder
        for i in range(self.num_layers-1):
            # Upsample
            x = self.upsample_layers[i](x)
            
            # Skip connection - retrieve from the end of skips list
            skip_idx = len(skips) - 1 - i
            skip = skips[skip_idx]
            
            # Concatenate skip connection
            x = torch.cat([x, skip], dim=-1)
            
            # Project concatenated features
            x = self.skip_proj[i](x)
            
            # Decoder blocks
            x = self.decoder[i](x)
        
        # Final upsampling
        x = self.final_up(x)
        B, L, C = x.shape
        H = W = int(L**0.5)
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)
        return self.output(x)

############################################
# Wrapper
############################################
class SwinUNetWrapper(nn.Module):
    def __init__(self, in_channels=3,
                num_classes=10,
                img_size=224,
                embed_dim=32,
                depths=[1,1,2,1],
                heads=[1, 2, 4, 8],
                patch_size=4,
                window_size=7):
        super().__init__()
        config = {
            'model': {
                'img_size': img_size,
                'in_channels': in_channels,
                'num_classes': num_classes,
                'depths': depths,
                'heads': heads,
                'swin': {
                    'embed_dim': embed_dim,
                    'patch_size': patch_size,
                    'window_size': window_size,
                }
            }
        }
        self.model = SwinUNet(config)
    
    def forward(self, x):
        return self.model(x)

############################################
# Test
############################################
if __name__ == "__main__":
    model = SwinUNetWrapper(in_channels=3, num_classes=10, img_size=224)
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        y = model(x)
    print("Input :", x.shape)
    print("Output:", y.shape)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")