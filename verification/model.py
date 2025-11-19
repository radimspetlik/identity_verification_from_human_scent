import torch
import torch.nn as nn
import torch.functional as F
import numpy as np
import torchvision
from tqdm import tqdm
from torchvision import transforms
from torchvision.utils import make_grid
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import random
from torchvision import models
from torchvision.models import (
    resnet18, resnet34, resnet50, resnet101, resnet152,
    ResNet18_Weights, ResNet34_Weights, ResNet50_Weights,
    ResNet101_Weights, ResNet152_Weights,

    vgg16, VGG16_Weights,
    inception_v3, Inception_V3_Weights,
    googlenet, GoogLeNet_Weights
)

pretrained_nets = [
    'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
    'vgg16', 'inception_v3', 'googlenet'
]

# Constants
DROP_RATE = 0


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, ker_size, stride=2, padding=1, droprate=DROP_RATE, pool=False):
        super().__init__()
        if not pool:
            self.layers = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=ker_size,
                          stride=stride, padding=padding),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                nn.Dropout(droprate))
        else:
            self.layers = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=ker_size,
                          stride=stride, padding=padding),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Dropout(droprate))

    def forward(self, x):
        return self.layers(x)


class ResidualConvBlock(nn.Module):
    def __init__(self, in_ch,out_ch):
        super(ResidualConvBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
        self.skip = nn.Identity()
        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1)


    def forward(self, x):
        identity = self.skip(x)
        out = self.layers(x)
        return out + identity


import torch
import torch.nn as nn




class MLP(nn.Module):
    def __init__(self, in_features, out_features, droprate=DROP_RATE):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(),
            nn.Dropout(droprate))

    def forward(self, x):
        return self.layers(x)


class ResidualMLP(nn.Module):
    def __init__(self, in_features, out_features, droprate=DROP_RATE):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(),
            nn.Dropout(droprate)
        )
        self.skip = nn.Linear(
            in_features, out_features) if in_features != out_features else nn.Identity()

    def forward(self, x):
        identity = self.skip(x)  # Transform input if necessary
        out = self.layers(x)
        return out + identity


class Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            ConvBlock(1, 4, ker_size=5, stride=1, padding=1, pool=False),
            ConvBlock(4, 8, ker_size=5, stride=2, padding=1, pool=False),
            ConvBlock(8, 16, ker_size=5, stride=1, padding=1, pool=False),
            ConvBlock(16, 32, ker_size=5, stride=2, padding=1, pool=False),
            ConvBlock(32, 64, ker_size=3, stride=1, padding=1, pool=False),
            ConvBlock(64, 128, ker_size=3, stride=2, padding=1, pool=False),
            ConvBlock(128, 256, ker_size=3, stride=1, padding=1, pool=True),
            ConvBlock(256, 512, ker_size=3, stride=2, padding=1, pool=True),
            ConvBlock(512, 1024, ker_size=3, stride=1, padding=1, pool=True),
            ConvBlock(1024, 2048, ker_size=3, stride=2, padding=1, pool=True),

        )
        self.MLP_layers = nn.Sequential(
            ResidualMLP(8192, 128)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.MLP_layers(x)
        return x

    def encode(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        return x


class ClassificationNet(nn.Module):

    def __init__(self, num_classes):
        super().__init__()
        self.linear = nn.Sequential(
            MLP(8192, 4096),
            MLP(4096, 2048),
            MLP(2048, 1024),
            MLP(1024, 512),
            MLP(512, 256),
            MLP(256, 128),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.linear(x)


class VerificationNetWorking(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            ConvBlock(1, 4, ker_size=5, stride=2, padding=1, pool=False),
            ConvBlock(4, 8, ker_size=5, stride=2, padding=1, pool=False),
            ConvBlock(8, 16, ker_size=5, stride=2, padding=1, pool=False),
            ConvBlock(16, 32, ker_size=5, stride=2, padding=1, pool=False),
            ConvBlock(32, 64, ker_size=5, stride=2, padding=1, pool=False),
            ConvBlock(64, 128, ker_size=5, stride=2, padding=1, pool=False),
            ConvBlock(128, 256, ker_size=5, stride=2, padding=1, pool=False),
        )
        self.MLP_layers = nn.Sequential(
            ResidualMLP(7168, 128)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.MLP_layers(x)
        return x


class VerificationNetShallow(nn.Module):
    def __init__(self,input_channels=1, output_dim=128,dropouts=False):
        """
        ConvBlock(1, 4, ker_size=5, stride=2, padding=1, pool=False),
        ConvBlock(4, 8, ker_size=5, stride=2, padding=1, pool=False),
        ConvBlock(8, 16, ker_size=5, stride=2, padding=1, pool=False),
        ConvBlock(16, 32, ker_size=5, stride=2, padding=1, pool=False),
        ConvBlock(32, 64, ker_size=5, stride=2, padding=1, pool=False),
        ConvBlock(64, 128, ker_size=5, stride=2, padding=1, pool=False),
        ConvBlock(128, 256, ker_size=5, stride=2, padding=1, pool=False),
        """
        super().__init__()
        ch = 64
        if dropouts:
            d1 = 0.05
            d2 = 0.1
            d3 = 0.15
            print("Dropout enabled in VerificationNetShallow")
        else:
            d1 = d2 = d3 = 0
        self.backbone = nn.Sequential(
            ConvBlock(input_channels, 4, ker_size=(8,2), stride=(2,1), padding=(1,0), pool=False,droprate=0),
            ResidualConvBlock(4, 4),
            ConvBlock(4, 16, ker_size=(8,2), stride=(2,1), padding=(1,0), pool=False,droprate=d1),
            ResidualConvBlock(16, 16),
            ConvBlock(16, 64, ker_size=(8,2), stride=(2,1), padding=(1,0), pool=False,droprate=d1),
            ResidualConvBlock(64, 64),

        )
        self.downsample = nn.Sequential(
            ConvBlock(ch, ch, ker_size=5, stride=2, padding=1, pool=False, droprate=d2),
            ConvBlock(ch, ch, ker_size=5, stride=2, padding=1, pool=False, droprate=d2),
            ConvBlock(ch, ch, ker_size=5, stride=2, padding=1, pool=False, droprate=d3),
            ConvBlock(ch, ch, ker_size=5, stride=2, padding=1, pool=False, droprate=d3)
        )
        self.MLP_layers = nn.Sequential(
            ResidualMLP(6912, output_dim)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.downsample(x)
        x = x.view(x.size(0), -1)
        x = self.MLP_layers(x)
        return x

class VerificationNetShallowTuned(nn.Module):
    def __init__(self,input_channels=1, output_dim=128):
        """
        ConvBlock(1, 4, ker_size=5, stride=2, padding=1, pool=False),
        ConvBlock(4, 8, ker_size=5, stride=2, padding=1, pool=False),
        ConvBlock(8, 16, ker_size=5, stride=2, padding=1, pool=False),
        ConvBlock(16, 32, ker_size=5, stride=2, padding=1, pool=False),
        ConvBlock(32, 64, ker_size=5, stride=2, padding=1, pool=False),
        ConvBlock(64, 128, ker_size=5, stride=2, padding=1, pool=False),
        ConvBlock(128, 256, ker_size=5, stride=2, padding=1, pool=False),
        """
        super().__init__()
        ch = 64
        self.backbone = nn.Sequential(
            ConvBlock(input_channels, 4, ker_size=(8,2), stride=(2,1), padding=(1,0), pool=False),
            ResidualConvBlock(4, 4),
            ConvBlock(4, 16, ker_size=(8,2), stride=(2,1), padding=(1,0), pool=False),
            ResidualConvBlock(16, 16),
            ConvBlock(16, 64, ker_size=(8,2), stride=(2,1), padding=(1,0), pool=False),
            ResidualConvBlock(64, 64),

        )
        self.downsample = nn.Sequential(
            ConvBlock(ch, ch, ker_size=3, stride=2, padding=1, pool=False),
            ConvBlock(ch, ch, ker_size=3, stride=2, padding=1, pool=False),
            ConvBlock(ch, ch, ker_size=3, stride=2, padding=1, pool=False),
            ConvBlock(ch, ch, ker_size=3, stride=2, padding=1, pool=False),
            ConvBlock(ch, ch, ker_size=3, stride=2, padding=1, pool=False),
            ConvBlock(ch, ch, ker_size=3, stride=2, padding=1, pool=False),
            ConvBlock(ch, ch, ker_size=3, stride=2, padding=1, pool=False),
            ConvBlock(ch, ch, ker_size=3, stride=2, padding=1, pool=False),
        )
        self.MLP_layers = nn.Sequential(
            ResidualMLP(128, output_dim)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.downsample(x)
        x = x.view(x.size(0), -1)
        x = self.MLP_layers(x)
        return x

class VerificationNetShallowOriginal(nn.Module):
    def __init__(self, input_channels=1, output_dim=4096):
        super().__init__()
        self.backbone = nn.Sequential(
            # TODO 256 channels everywhere and residual train with 2IDS -> AUC=1 WORKING :D, LR=3E-5
            ConvBlock(input_channels, 4, ker_size=5, stride=2, padding=1, pool=False),
            ConvBlock(4, 8, ker_size=5, stride=2, padding=1, pool=False),
            ConvBlock(8, 16, ker_size=5, stride=2, padding=1, pool=False),
            ConvBlock(16, 32, ker_size=5, stride=2, padding=1, pool=False),
            ConvBlock(32, 64, ker_size=5, stride=2, padding=1, pool=False),
            ConvBlock(64, 128, ker_size=5, stride=2, padding=1, pool=False),
            ConvBlock(128, 256, ker_size=5, stride=2, padding=1, pool=False),
        )
        self.MLP_layers = nn.Sequential(
            ResidualMLP(1536, output_dim),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.MLP_layers(x)
        return x


class VerificationNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            ConvBlock(1, 4, ker_size=3, stride=1, padding=1, pool=False),
            ConvBlock(4, 8, ker_size=3, stride=2, padding=1, pool=False),
            ConvBlock(8, 16, ker_size=3, stride=1, padding=1, pool=False),
            ConvBlock(16, 32, ker_size=3, stride=2, padding=1, pool=False),
            ConvBlock(32, 64, ker_size=3, stride=1, padding=1, pool=False),
            ConvBlock(64, 128, ker_size=3, stride=2, padding=1, pool=True),
            ConvBlock(128, 256, ker_size=3, stride=1, padding=1, pool=True),
            ConvBlock(256, 512, ker_size=3, stride=2, padding=1, pool=True),
        )
        self.MLP_layers = nn.Sequential(
            ResidualMLP(30720, 8096),
            ResidualMLP(8096, 2048),
            ResidualMLP(2048, 512),
            ResidualMLP(512, 128)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.MLP_layers(x)
        return x


class SiameseNetworkBCE(nn.Module):
    def __init__(self):
        super(SiameseNetworkBCE, self).__init__()
        self.feature_extractor = VerificationNet()
        self.fc = nn.Linear(128, 1)

    def forward(self, x1, x2):
        x1 = self.feature_extractor(x1)
        x2 = self.feature_extractor(x2)
        x = torch.abs(x1 - x2)
        x = self.fc(x)
        return x


class SiameseNetworkContrastive(nn.Module):
    def __init__(self):
        super(SiameseNetworkContrastive, self).__init__()
        self.feature_extractor = VerificationNet()

    def forward(self, x1, x2):
        x1 = self.feature_extractor(x1)
        x2 = self.feature_extractor(x2)
        return x1, x2



import torch
import torch.nn as nn
import torch.nn.functional as F

class VerificationNetWithMHAttention(nn.Module):
    def __init__(self,
                 input_channels=1,
                 output_dim=128,
                 attn_heads=4,
                 attn_dropout=0.1):
        super().__init__()
        # --- 1) CNN Backbone (jako předtím) ---
        self.backbone = nn.Sequential(
            ConvBlock(input_channels, 4,  ker_size=(8,2), stride=(2,1), padding=(1,0), pool=False),
            ResidualConvBlock(4,4),
            ConvBlock(4,16,  ker_size=(8,2), stride=(2,1), padding=(1,0), pool=False),
            ResidualConvBlock(16,16),
            ConvBlock(16,64, ker_size=(8,2), stride=(2,1), padding=(1,0), pool=False),
            ResidualConvBlock(64,64),
        )
        self.downsample = nn.Sequential(
            ConvBlock(64,64, ker_size=5, stride=2, padding=1, pool=False),
            ConvBlock(64,64, ker_size=5, stride=2, padding=1, pool=False),
            ConvBlock(64,64, ker_size=5, stride=2, padding=1, pool=False),
            ConvBlock(64,64, ker_size=5, stride=2, padding=1, pool=False),
        )

        # --- 2) Multi‐Head Attention ---
        # embed_dim musí odpovídat C (po posledním ConvBlock je to 64)
        self.embed_dim = 64
        self.attn = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=attn_heads,
            dropout=attn_dropout,
            batch_first=True   # takže vstup i výstup jako B×S×C
        )

        # --- 3) Head: aggreagce + MLP ---
        # Po attenu vezmeme průměr přes všechny tokeny
        self.mlp = nn.Sequential(
            nn.Linear(self.embed_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0)
        )

    def forward(self, x):
        # 1) extrakce CNN
        x = self.backbone(x)
        x = self.downsample(x)     # nyní x: [B, 64, H, W]

        # 2) příprava pro attention
        B, C, H, W = x.shape
        S = H * W
        # vyrovnat na sekvenci: B×S×C
        x = x.view(B, C, S).permute(0, 2, 1)

        # 3) self‐attention (query=key=value)
        # attn_out: [B, S, C]
        attn_out, _ = self.attn(x, x, x)

        # 4) agregujeme do jedné vektorky: průměr tokenů -> pruměr přes S
        pooled = attn_out.mean(dim=1)  # [B, C]

        # 5) finální MLP head
        out = self.mlp(pooled)         # [B, output_dim]
        return out
import torch
import torch.nn as nn
import torch.nn.functional as F

class VerificationNetLayeredAttention(nn.Module):
    def __init__(self,
                 input_channels=1,
                 output_dim=128,
                 attn_heads=4,
                 attn_dropout=0.1):
        super().__init__()
        # --- 1) CNN Backbone (beze změny) ---
        self.backbone = nn.Sequential(
            ConvBlock(input_channels, 4,  ker_size=(8,2), stride=(2,1), padding=(1,0), pool=False),
            ResidualConvBlock(4,4),
            ConvBlock(4,16,  ker_size=(8,2), stride=(2,1), padding=(1,0), pool=False),
            ResidualConvBlock(16,16),
            ConvBlock(16,64, ker_size=(8,2), stride=(2,1), padding=(1,0), pool=False),
            ResidualConvBlock(64,64),
        )

        # --- 2) Downsample + Attention v každé vrstvě ---
        downsample_convs = []
        downsample_attns = []
        for _ in range(4):
            # všechny downsample končí na 64 kanálech
            downsample_convs.append(
                ConvBlock(64, 64, ker_size=5, stride=2, padding=1, pool=False)
            )
            # embed_dim=64, batch_first=True pro pohodlné B×S×C
            downsample_attns.append(
                nn.MultiheadAttention(embed_dim=64,
                                      num_heads=attn_heads,
                                      dropout=attn_dropout,
                                      batch_first=True)
            )

        self.downsample_convs = nn.ModuleList(downsample_convs)
        self.downsample_attns = nn.ModuleList(downsample_attns)

        # --- 3) Pool+MLP head (stejné jako předtím) ---
        self.mlp = nn.Sequential(
            nn.Linear(64, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(attn_dropout)
        )

    def forward(self, x):
        # 1) základní CNN extrakce
        x = self.backbone(x)      # → [B, 64, H, W]

        # 2) postupné downsample + attention
        for conv, attn in zip(self.downsample_convs, self.downsample_attns):
            x = conv(x)           # → [B, 64, H', W']
            B, C, H, W = x.shape
            S = H * W
            # předefinujeme jako sekvenci tokenů
            x_seq = x.view(B, C, S).permute(0, 2, 1)  # [B, S, C]
            # self-attention Q=K=V
            attn_out, _ = attn(x_seq, x_seq, x_seq)  # [B, S, C]
            # zpět do tvaru mapy
            x = attn_out.permute(0, 2, 1).view(B, C, H, W)

        # 3) global pooling a MLP
        # průměr přes všechny tokeny (H×W)
        out = x.mean(dim=[2,3])    # [B, 64]
        out = self.mlp(out)        # [B, output_dim]
        return out

from math import sqrt, sin, cos, pi
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)            # [T, D]
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * -(pi / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe)                 # [T, D]

    def forward(self, x):
        # x: [B, T, D]
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0)

# ------------------------------------------------
# 3) Verification net s Full Transformer-Encoder
# ------------------------------------------------
class VerificationNetWithTransformer(nn.Module):
    def __init__(self, input_channels=1, output_dim=128,
                 nhead=4, num_layers=4, dim_feedforward=1024,
                 dropout=0, max_seq_len=1000):
        super().__init__()
        # 1) CNN backbone (stejné vrstvy)
        self.backbone = nn.Sequential(
            ConvBlock(input_channels, 4,  ker_size=(8,2), stride=(2,1), padding=(1,0), pool=False),
            ResidualConvBlock(4,4),
            ConvBlock(4,16, ker_size=(8,2), stride=(2,1), padding=(1,0), pool=False),
            ResidualConvBlock(16,16),
            ConvBlock(16,64,ker_size=(8,2), stride=(2,1), padding=(1,0), pool=False),
            ResidualConvBlock(64,64),
        )
        self.downsample = nn.Sequential(
            ConvBlock(64,64,ker_size=5, stride=2, padding=1, pool=False),
            ConvBlock(64,64,ker_size=5, stride=2, padding=1, pool=False),
            ConvBlock(64,64,ker_size=5, stride=2, padding=1, pool=False),
            ConvBlock(64,64,ker_size=5, stride=2, padding=1, pool=False),
        )
        # 2) PositionalEncoding + TransformerEncoder
        self.embed_dim = 64
        self.pos_encoder = PositionalEncoding(self.embed_dim, max_len=max_seq_len)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout or 0.1,  # try 0.1
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.dropout = nn.Dropout(dropout or 0.1)
        self.mlp = nn.Sequential(
            nn.Linear(self.embed_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout or 0.1)
        )

    def forward(self, x):
        x = self.backbone(x);
        x = self.downsample(x)
        B, C, H, W = x.shape
        x = x.view(B, C, H * W).permute(0, 2, 1)  # [B, S, C]

        x = self.pos_encoder(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # [B, S+1, C]

        x = self.transformer(x)
        pooled = x[:, 0]  # take CLS

        return self.mlp(self.dropout(pooled))

import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_

class PatchEmbedding(nn.Module):
    """
    Split image into patches and embed.
    """
    def __init__(self, in_channels=1, embed_dim=768, patch_size=(16,16), img_size=(128,256)):
        super().__init__()
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size,
                              stride=patch_size)

    def forward(self, x:
        torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        x = self.proj(x)  # [B, embed_dim, H/ps, W/ps]
        B, E, Gh, Gw = x.shape
        x = x.flatten(2)  # [B, E, num_patches]
        x = x.transpose(1, 2)  # [B, num_patches, E]
        return x

class VisionTransformer(nn.Module):
    """
    Enhanced Vision Transformer with pre-norm, layer norms, and dropout to improve stability and prevent NaNs.

    Args:
        img_size: tuple, image height and width
        patch_size: tuple, patch height and width
        input_channels: number of input channels
        output_dim: embedding output size or classification heads
        embed_dim: dimension of patch embeddings (d_model)
        depth: number of Transformer blocks
        num_heads: attention heads
        mlp_ratio: expansion ratio in FFN
        dropout: dropout rate applied after embeddings and in FFN
        attn_dropout: dropout rate inside attention
    """
    def __init__(
        self,
        img_size=(970, 316),
        patch_size=(32,32),
        input_channels=1,
        output_dim=128,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        dropout=0,
        attn_dropout=0,
    ):
        super().__init__()
        # 1) Patch embedding
        self.patch_embed = PatchEmbedding(input_channels, embed_dim, patch_size, img_size)
        # normalize patch embeddings
        self.patch_norm  = nn.LayerNorm(embed_dim)
        self.pos_drop    = nn.Dropout(dropout)

        # 2) Learnable CLS token & positional embeddings
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)

        # 3) Transformer Encoder (pre-norm for stability)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=attn_dropout,
            activation=F.gelu,
            batch_first=True,
            norm_first=True  # pre-norm stabilizes deep networks
        )
        # add final norm in encoder
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth,
            norm=nn.LayerNorm(embed_dim)
        )

        # 4) Head normalization + projection
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        B = x.size(0)
        # patch & project
        x = self.patch_embed(x)    # [B, N, D]
        x = self.patch_norm(x)     # stabilize embeddings
        x = self.pos_drop(x)
        # prepend CLS
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)          # [B, N+1, D]
        # add positional embeddings
        x = x + self.pos_embed
        x = self.pos_drop(x)
        # transformer encoding
        x = self.transformer(x)                # [B, N+1, D]
        # final norm + head
        x = self.norm(x)
        cls_out = x[:, 0]                      # [B, D]
        out = self.head(cls_out)               # [B, output_dim]
        return out



from torchvision.models import resnet18, resnet34, ResNet18_Weights, ResNet34_Weights

import torch
import torch.nn as nn
from torchvision.models import (
    resnet18, resnet34, resnet50, resnet101, resnet152,
    vgg16, inception_v3, googlenet
)
from torchvision.models import (
    ResNet18_Weights, ResNet34_Weights, ResNet50_Weights,
    ResNet101_Weights, ResNet152_Weights,
    VGG16_Weights, Inception_V3_Weights, GoogLeNet_Weights
)


def get_pretrained_backbone(name: str, input_channels: int = 3, input_size=(224, 224)):
    """
    Load a pretrained model, adapt input channels, remove classification head, and return:
        - Modified model
        - Feature dimension before final classification
        - Reference to the first conv layer
    """
    # Load model with weights
    if name == 'resnet18':
        weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights=weights)
    elif name == 'resnet34':
        weights = ResNet34_Weights.DEFAULT
        model = resnet34(weights=weights)
    elif name == 'resnet50':
        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights)
    elif name == 'resnet101':
        weights = ResNet101_Weights.DEFAULT
        model = resnet101(weights=weights)
    elif name == 'resnet152':
        weights = ResNet152_Weights.DEFAULT
        model = resnet152(weights=weights)
    elif name == 'vgg16':
        weights = VGG16_Weights.DEFAULT
        model = vgg16(weights=weights)
    elif name == 'inception_v3':
        weights = Inception_V3_Weights.DEFAULT
        model = inception_v3(weights=weights, aux_logits=True)
        model.aux_logits = False  # disables usage of aux logits
        model.AuxLogits = nn.Identity()  # remove the layer
    elif name == 'googlenet':
        weights = GoogLeNet_Weights.DEFAULT
        model = googlenet(weights=weights)  # aux_logits=True is required here
        model.aux_logits = False  # disable usage in forward pass
        model.aux1 = nn.Identity()
        model.aux2 = nn.Identity()
    else:
        raise ValueError(f"Unsupported model: {name}")

    # Modify input conv layer if needed
    if name.startswith("resnet"):
        conv_ref = model.conv1
    elif name == "googlenet":
        conv_ref = model.conv1
    elif name == "inception_v3":
        conv_ref = model.Conv2d_1a_3x3.conv
    elif name.startswith("vgg"):
        conv_ref = model.features[0]
    else:
        raise ValueError(f"Unsupported model for conv patching: {name}")

    if input_channels != 3:
        new_conv = nn.Conv2d(input_channels, conv_ref.out_channels,
                             kernel_size=conv_ref.kernel_size,
                             stride=conv_ref.stride,
                             padding=conv_ref.padding,
                             bias=(conv_ref.bias is not None))
        if name.startswith("resnet") or name in ["inception_v3", "googlenet"]:
            model.conv1 = new_conv
        elif name.startswith("vgg"):
            model.features[0] = new_conv
        conv_ref = new_conv  # update reference

    # Remove classification head
    if name.startswith("resnet") or name in ["inception_v3", "googlenet"]:
        in_features = model.fc.in_features
        model.fc = nn.Identity()
    elif name.startswith("vgg"):
        model.classifier = nn.Identity()

        # Dynamically infer feature size (e.g., 25088 for VGG16)
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, *input_size)
            out = model.features(dummy)
            in_features = out.view(1, -1).shape[1]

    return model, in_features, conv_ref

def build_feature_extractor(model, output_dim, input_channels, freeze_backbone,dropouts=False):
    if model == "VerificationNetShallow":
        feature_extractor = VerificationNetShallow(input_channels=input_channels, output_dim=output_dim,dropouts=dropouts)
        projection = nn.Identity()
    elif model == "VerificationNetShallowOriginal":
        feature_extractor = VerificationNetShallowOriginal(input_channels=input_channels, output_dim=output_dim)
        projection = nn.Identity()
    elif model == "VerificationNetWithMHAttention":
        feature_extractor = VerificationNetWithMHAttention(input_channels=input_channels, output_dim=output_dim)
        projection = nn.Identity()
    elif model == "VerificationNetWithTransformer":
        feature_extractor = VerificationNetWithTransformer(input_channels=input_channels, output_dim=output_dim)
        projection = nn.Identity()
    elif model == "VerificationNetLayeredAttention":
        feature_extractor = VerificationNetLayeredAttention(input_channels=input_channels, output_dim=output_dim)
        projection = nn.Identity()
    elif model == "VisionTransformer":
        feature_extractor = VisionTransformer(input_channels=input_channels, output_dim=output_dim)
        projection = nn.Identity()
    elif model == "VerificationNetShallowTuned":
        feature_extractor = VerificationNetShallowTuned(input_channels=input_channels, output_dim=output_dim)
        projection = nn.Identity()
    elif model in pretrained_nets:
        feature_extractor, in_features, conv1_ref = get_pretrained_backbone(model, input_channels)
        projection = nn.Sequential(
            nn.Linear(in_features, output_dim),
            nn.Tanh()
        )
        if freeze_backbone:
            for param in feature_extractor.parameters():
                param.requires_grad = False
            for param in conv1_ref.parameters():
                param.requires_grad = True
    else:
        raise ValueError("Model not supported")
    return feature_extractor, projection


class EncoderCosine(nn.Module):
    def __init__(self, model, output_dim=4096, input_channels=1, freeze_backbone=True,dropouts=False):
        super(EncoderCosine, self).__init__()
        self.feature_extractor, self.projection = build_feature_extractor(model, output_dim, input_channels,
                                                                          freeze_backbone,dropouts)
        self.normalize = True

    def forward(self, x):
        x = self.feature_extractor(x)
        if len(x.shape) > 2:
            x = torch.flatten(x, 1)
        x = self.projection(x)
        if self.normalize:
            x = x / torch.norm(x, dim=1, keepdim=True)
        return x


class EncoderDistance(nn.Module):
    def __init__(self, model, output_dim=4096, input_channels=1, freeze_backbone=True):
        super(EncoderDistance, self).__init__()
        self.feature_extractor, self.projection = build_feature_extractor(model, output_dim, input_channels,
                                                                          freeze_backbone)
        self.normalize = True

    def forward(self, x):
        x = self.feature_extractor(x)
        if len(x.shape) > 2:
            x = torch.flatten(x, 1)
        x = self.projection(x)
        if self.normalize:
            x = x / torch.norm(x, dim=1, keepdim=True)
        return x
