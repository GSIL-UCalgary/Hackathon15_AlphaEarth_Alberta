import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        hidden_planes = max(1, in_planes // ratio)
        self.fc1 = nn.Conv2d(in_planes, hidden_planes, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(hidden_planes, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(x_cat)
        return self.sigmoid(out)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out

class TransformerBottleneck(nn.Module):
    def __init__(self, in_channels, num_heads=8, hidden_dim=512, dropout=0.1):
        super(TransformerBottleneck, self).__init__()
        self.in_channels = in_channels
        self.embedding = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim*4, 
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.project_back = nn.Conv2d(hidden_dim, in_channels, kernel_size=1)
        
    def forward(self, x):
        b, c, h, w = x.shape
        x_embed = self.embedding(x).flatten(2).transpose(1, 2)
        x_trans = self.transformer_layer(x_embed)
        x_trans = x_trans.transpose(1, 2).reshape(b, -1, h, w)
        out = self.project_back(x_trans)
        return x + out

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))
        dilations = [6, 12, 18]
        for dilation in dilations:
            modules.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        global_pool = self.global_avg_pool(x)
        global_pool = F.interpolate(global_pool, size=x.size()[2:], mode='bilinear', align_corners=False)
        res.append(global_pool)
        res = torch.cat(res, dim=1)
        return self.project(res)

class AttentionDeepLabV3Plus(nn.Module):
    def __init__(self, num_classes=13, in_channels=10, backbone='resnet50', pretrained=False):
        super(AttentionDeepLabV3Plus, self).__init__()
        
        # --- Encoder (Backbone) ---
        if backbone == 'resnet50':
            resnet = models.resnet50(pretrained=pretrained)
            if in_channels != 3:
                original_conv1 = resnet.conv1
                resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
                with torch.no_grad():
                    resnet.conv1.weight[:, :3] = original_conv1.weight
                    if in_channels > 3:
                        nn.init.kaiming_normal_(resnet.conv1.weight[:, 3:], mode='fan_out', nonlinearity='relu')

            self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
            self.layer1 = resnet.layer1 
            self.layer2 = resnet.layer2
            self.layer3 = resnet.layer3
            self.layer4 = resnet.layer4 
            low_level_channels = 256
            high_level_channels = 2048
        
        self.cbam_low = CBAM(low_level_channels)
        self.aspp = ASPP(high_level_channels, 256)
        self.transformer_bottleneck = TransformerBottleneck(in_channels=256)
        
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        
        self.final_conv = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False), 
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            CBAM(256),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )

    def forward(self, x, segments=None, assignment=None):
        input_shape = x.shape[-2:]
        
        x = self.layer0(x)
        low_level = self.layer1(x) 
        x = self.layer2(low_level)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.aspp(x)
        x = self.transformer_bottleneck(x)
        
        x = F.interpolate(x, size=low_level.size()[2:], mode='bilinear', align_corners=False)
        
        low_level = self.cbam_low(low_level)
        low_level = self.low_level_conv(low_level)
        
        x = torch.cat([x, low_level], dim=1)
        
        x = self.final_conv(x)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        
        return x