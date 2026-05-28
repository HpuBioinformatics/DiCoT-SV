import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, num_groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class AttentionPooling(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.attn = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x):
       
        B, C, H, W = x.shape

        attn_map = self.attn(x)                 
        attn_map = attn_map.view(B, -1)
        attn_weight = F.softmax(attn_map, dim=1)
        attn_weight = attn_weight.view(B, 1, H, W)

        x = x * attn_weight
        x = x.sum(dim=(2, 3))                    
        return x


class CNN_Attention_Classifier(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()

        # ========= Stage 1 =========
        self.stage1 = nn.Sequential(
            ConvBlock(1, 32, num_groups=8),
            ConvBlock(32, 32, num_groups=8)
        )

        # ========= Stage 2 =========
        self.stage2 = nn.Sequential(
            ConvBlock(32, 64, stride=2, num_groups=8),
            ConvBlock(64, 64, num_groups=8)
        )

        # ========= Stage 3 =========
        self.stage3 = nn.Sequential(
            ConvBlock(64, 128, stride=2, num_groups=8),
            ConvBlock(128, 128, num_groups=8)
        )

        # ========= Attention Pooling =========
        self.attn_pool = AttentionPooling(128)

        # ========= Classifier =========
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x: (B, 1, H, W)
        x = self.stage1(x)     # (B, 32, H, W)
        x = self.stage2(x)     # (B, 64, H/2, W/2)
        x = self.stage3(x)     # (B, 128, H/4, W/4)

        x = self.attn_pool(x) # (B, 128)
        out = self.classifier(x)
        return out








