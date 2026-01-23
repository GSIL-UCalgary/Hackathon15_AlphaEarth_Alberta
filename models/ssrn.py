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
    def __init__(self, in_channels=10, num_classes=13, msize=18, inter_size=49):
        super(SSRNForSegmentation, self).__init__()

        # Initial projection
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_size, 1, bias=False),
            nn.LeakyReLU(),
            nn.BatchNorm2d(inter_size),
        )
        
        # Spectral-spatial processing blocks
        self.layer2 = SPC32(msize, outplane=inter_size, kernel_size=[inter_size,1,1], padding=[0,0,0])        
        self.layer3 = SARes(inter_size, ratio=8)
        
        # Dimension reduction
        self.layer4 = nn.Conv2d(inter_size, msize, kernel_size=1)
        self.bn4 = nn.BatchNorm2d(msize)
        
        # Additional processing
        self.layer5 = SARes(msize, ratio=8)
        self.layer6 = SPC32(msize, outplane=msize, kernel_size=[msize,1,1], padding=[0,0,0])

        # Segmentation head
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
        n, c, h, w = x.size()

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.bn4(F.leaky_relu(self.layer4(x)))
        x = self.layer5(x)
        x = self.layer6(x)

        # Segmentation output
        seg_map = self.segmentation_head(x)
        return seg_map