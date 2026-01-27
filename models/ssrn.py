import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatAttn(nn.Module):
    """ Position attention module"""
    def __init__(self, in_dim, ratio=8):
        super(SpatAttn, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//ratio, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//ratio, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out


class SARes(nn.Module):
    def __init__(self, in_dim, ratio=8, resin=False):
        super(SARes, self).__init__()
        
        self.sa1 = SpatAttn(in_dim, ratio)
        self.sa2 = SpatAttn(in_dim, ratio)
        
    def forward(self, x):
        identity = x 
        x = self.sa1(x)
        x = self.sa2(x)
        return F.relu(x + identity)


class SPC32(nn.Module):
    def __init__(self, msize=24, outplane=49, kernel_size=[7,1,1], stride=[1,1,1], padding=[3,0,0], spa_size=9, bias=True):
        super(SPC32, self).__init__()
                                                  
        self.convm0 = nn.Conv3d(1, msize, kernel_size=kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(outplane)
        
        self.convm2 = nn.Conv3d(1, msize, kernel_size=kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(outplane)

    def forward(self, x, identity=None):
        if identity is None:
            identity = x
        n,c,h,w = identity.size()
        
        mask0 = self.convm0(x.unsqueeze(1)).squeeze(2)
        mask0 = torch.softmax(mask0.view(n,-1,h*w), -1)
        mask0 = mask0.view(n,-1,h,w)
        _,d,_,_ = mask0.size()
        
        fk = torch.einsum('ndhw,nchw->ncd', mask0, x)
        out = torch.einsum('ncd,ndhw->ncdhw', fk, mask0)
        out = F.leaky_relu(out)
        out = out.sum(2)
        out = out
        out0 = self.bn1(out.view(n,-1,h,w))
        
        mask2 = self.convm2(out0.unsqueeze(1)).squeeze(2)
        mask2 = torch.softmax(mask2.view(n,-1,h*w), -1)
        mask2 = mask2.view(n,-1,h,w)
        
        fk = torch.einsum('ndhw,nchw->ncd', mask2, x)
        out = torch.einsum('ncd,ndhw->ncdhw', fk, mask2)
        out = F.leaky_relu(out)
        out = out.sum(2)
        out = out + identity
        out = self.bn2(out.view(n,-1,h,w))

        return out


class SSRNForSegmentation(nn.Module):
    def __init__(self, in_channels=10, num_classes=13, msize=18, inter_size=49, 
                 downsample=4, apply_downsampling=True):
        super(SSRNForSegmentation, self).__init__()
        self.num_classes = num_classes
        self.downsample = downsample
        self.apply_downsampling = apply_downsampling
        
        if apply_downsampling:
            # With convolutional downsampling
            self.stem = nn.Sequential(
                # Conv with stride to reduce spatial size
                nn.Conv2d(in_channels, inter_size, kernel_size=downsample, stride=downsample, padding=0),
                nn.BatchNorm2d(inter_size),
                nn.LeakyReLU(),
            )
            self.upsample = nn.Upsample(scale_factor=downsample, mode='bilinear', align_corners=True)
        else:
            # Without downsampling (original SSRN)
            self.layer1 = nn.Sequential(
                nn.Conv2d(in_channels, inter_size, 1, bias=False),
                nn.LeakyReLU(),
                nn.BatchNorm2d(inter_size),
            )
            self.upsample = nn.Identity()
        
        # Common layers
        self.layer2 = SPC32(msize, outplane=inter_size, kernel_size=[inter_size,1,1], padding=[0,0,0])
        self.layer3 = SARes(inter_size, ratio=8)
        self.layer4 = nn.Conv2d(inter_size, msize, kernel_size=1)
        self.bn4 = nn.BatchNorm2d(msize)
        self.layer5 = SARes(msize, ratio=8)
        self.layer6 = SPC32(msize, outplane=msize, kernel_size=[msize,1,1], padding=[0,0,0])

        self.segmentation_head = nn.Sequential(
            nn.Conv2d(msize, msize, 3, padding=1),
            nn.BatchNorm2d(msize),
            nn.LeakyReLU(),
            nn.Conv2d(msize, msize // 2, 3, padding=1),
            nn.BatchNorm2d(msize // 2),
            nn.LeakyReLU(),
            nn.Conv2d(msize // 2, num_classes, 1)
        )

    def forward(self, x):
        if self.apply_downsampling:
            x = self.stem(x)
        else:
            x = self.layer1(x)
        
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.bn4(F.leaky_relu(self.layer4(x)))
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.upsample(x)
        return self.segmentation_head(x)

def main():
    """Test function to print output sizes of each layer"""
    print("=" * 60)
    print("SSRNForSegmentation Model Test with Stem")
    print("=" * 60)
    
    # Create device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Test different input sizes
    test_cases = [
        (2, 64, 224, 224),    # Batch=2, Channels=64, Height=224, Width=224
    ]
    
    for i, (batch, channels, height, width) in enumerate(test_cases):
        print(f"\n{'='*60}")
        print(f"Test Case {i+1}: Input shape ({batch}, {channels}, {height}, {width})")
        print(f"{'='*60}")
        
        # Create model with stem (4x downsampling)
        model = SSRNForSegmentation(
            in_channels=channels,
            num_classes=13,
            msize=18,
            inter_size=49,
            downsample=4
        ).to(device)
        
        # Create dummy input
        dummy_input = torch.randn(batch, channels, height, width).to(device)
        print(f"Dummy input shape: {dummy_input.shape}")
        print(f"Dummy input size: {dummy_input.element_size() * dummy_input.nelement() / 1024**2:.2f} MB")
        
        # Forward pass with gradient disabled
        with torch.no_grad():
            try:
                output = model(dummy_input)
                
                # Print summary
                print(f"\n✅ Test PASSED!")
                print(f"   Input:  {dummy_input.shape}")
                print(f"   Output: {output.shape}")
                print(f"   Expected output: (batch, num_classes, height, width)")
                print(f"   Actual output: ({batch}, {13}, {height}, {width})")
                
                # Check if output shape is correct
                expected_shape = (batch, 13, height, width)
                if output.shape == expected_shape:
                    print(f"   ✅ Output shape matches expected: {expected_shape}")
                else:
                    print(f"   ❌ Output shape mismatch!")
                    print(f"      Expected: {expected_shape}")
                    print(f"      Got:      {output.shape}")
                
                # Memory analysis
                print(f"\nMemory Analysis:")
                print(f"   Input memory: {dummy_input.element_size() * dummy_input.nelement() / 1024**2:.2f} MB")
                print(f"   Output memory: {output.element_size() * output.nelement() / 1024**2:.2f} MB")
                
                # Attention matrix size after downsampling
                down_h = height // 4
                down_w = width // 4
                attention_matrix_size = (down_h * down_w) * (down_h * down_w) * 4 / 1024**2  # in MB
                print(f"   Attention matrix size (after {down_h}×{down_w}): {attention_matrix_size:.1f} MB")
                
                # Compare with original (without stem)
                original_attention_size = (height * width) * (height * width) * 4 / 1024**3  # in GB
                print(f"   Original attention would be: {original_attention_size:.2f} GB")
                print(f"   Memory reduction: {original_attention_size * 1024 / attention_matrix_size:.0f}x")
                
            except torch.cuda.OutOfMemoryError:
                print(f"\n❌ CUDA Out of Memory!")
                print(f"   Input shape: {dummy_input.shape}")
                down_h = height // 4
                down_w = width // 4
                attention_matrix_size = (down_h * down_w) * (down_h * down_w) * 4 / 1024**2
                print(f"   Attention matrix would be: {attention_matrix_size:.1f} MB")
                print(f"   Try reducing batch size")
            except Exception as e:
                print(f"\n❌ Error: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"\n{'='*60}")
        
        # Clear GPU memory
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

if __name__ == '__main__':
    # Set random seed for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # Run main test
    main()