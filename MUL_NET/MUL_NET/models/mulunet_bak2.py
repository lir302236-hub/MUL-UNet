import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.init import _calculate_fan_in_and_fan_out
from timm.models.layers import to_2tuple, trunc_normal_
from .norm_layer import *

class ConvLayer(nn.Module):
    def __init__(self, net_depth, dim, kernel_size=3, gate_act=nn.Sigmoid):
        super().__init__()
        self.dim = dim

        self.net_depth = net_depth
        self.kernel_size = kernel_size

        self.Wv = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim, padding_mode='reflect')
        )

        self.Wg = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            gate_act() if gate_act in [nn.Sigmoid, nn.Tanh] else gate_act(inplace=False)
        )

        self.proj = nn.Conv2d(dim, dim, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            gain = (8 * self.net_depth) ** (
                    -1 / 4)
            fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
            std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
            trunc_normal_(m.weight, std=std)

            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, X):
        out = self.Wv(X) * self.Wg(X)
        out = self.proj(out)
        return out


class BasicBlock(nn.Module):
    def __init__(self, net_depth, dim, kernel_size=3, conv_layer=ConvLayer, norm_layer=nn.BatchNorm2d,
                 gate_act=nn.Sigmoid):
        super().__init__()
        self.norm = norm_layer(dim)
        self.conv = conv_layer(net_depth, dim, kernel_size, gate_act)

    def forward(self, x):
        identity = x
        x = self.norm(x)
        x = self.conv(x)
        x = identity + x
        return x


class BasicLayer(nn.Module):
    def __init__(self, net_depth, dim, depth, kernel_size=3, conv_layer=ConvLayer, norm_layer=nn.BatchNorm2d,
                 gate_act=nn.Sigmoid):
        super().__init__()
        self.dim = dim
        self.depth = depth
        # build blocks  MCCONV()
        self.blocks = nn.ModuleList([MCCONV(channel = dim) for i in range(depth)])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if kernel_size is None:
            kernel_size = patch_size

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=patch_size,
                              padding=(kernel_size - patch_size + 1) // 2, padding_mode='reflect')

    def forward(self, x):
        x = self.proj(x)
        return x


class PatchUnEmbed(nn.Module):
    def __init__(self, patch_size=4, out_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        self.out_chans = out_chans
        self.embed_dim = embed_dim

        if kernel_size is None:
            kernel_size = 1

        self.proj = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans * patch_size ** 2, kernel_size=kernel_size,
                      padding=kernel_size // 2, padding_mode='reflect'),
            nn.PixelShuffle(patch_size)
        )

    def forward(self, x):
        x = self.proj(x)
        return x


class SKFusion(nn.Module):
    def __init__(self, dim, height=2, reduction=8):
        super(SKFusion, self).__init__()
        self.height = height
        d = max(int(dim / reduction), 4)
        self.mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, d, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(d, dim * height, 1, bias=False)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, in_feats):
        B, C, H, W = in_feats[0].shape

        in_feats = torch.cat(in_feats, dim=1)
        in_feats = in_feats.view(B, self.height, C, H, W)

        feats_sum = torch.sum(in_feats, dim=1)
        attn = self.mlp(feats_sum)
        attn = self.softmax(attn.view(B, self.height, C, 1, 1))

        out = torch.sum(in_feats * attn, dim=1)
        return out


# 梯度提取
class grad_skip_connection(nn.Module):
    def __init__(self, dim, cuda_use=True):
        self.dim = dim
        super(grad_skip_connection, self).__init__()
        self.kernel = torch.Tensor(np.array([[1, 1, 1],
                                             [1, -8, 1],
                                             [1, 1, 1]])).expand(dim, 1, 3, 3) / 8

    def forward(self, x):
        return F.conv2d(x, self.kernel.to(x.device), stride=1, padding=1, groups=self.dim)


class MCCONV(nn.Module):
    def __init__(self, K1=1, K2=5, p1=0, p2=2, channel=64):
        super(MCCONV, self).__init__()
        self.BN = nn.BatchNorm2d(channel)
        self.conv1 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=K1, padding=p1, groups=channel,bias=False)
        self.conv2 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=K2, padding=p2, groups=channel,bias=False)
        self.conv_fusion = nn.Conv2d(in_channels=channel * 2, out_channels=channel, kernel_size=1, padding=0, bias=False)
        self.conv_start = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1, padding=0, bias=False),
            nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, padding=1, groups=channel, bias=False)
        )
        self.conv_end = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        x = self.BN(x)
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_end = self.conv_fusion(torch.cat((x_1, x_2), 1))
        x_start = self.conv_start(x)
        out = self.conv_end(torch.mul(x_start, F.sigmoid(x_end)))
        return out
    
    
class MULUNET(nn.Module):
    def __init__(self, kernel_size=5, base_dim=32, depths=[4, 4, 4, 4, 4, 4, 4], conv_layer=ConvLayer, norm_layer=nn.BatchNorm2d, gate_act=nn.Sigmoid, fusion_layer=SKFusion):
        super(MULUNET, self).__init__()
        assert len(depths) % 2 == 1
        stage_num = len(depths)
        half_num = stage_num // 2
        net_depth = sum(depths)
        embed_dims = [2 ** i * base_dim for i in range(half_num)]
        embed_dims = embed_dims + [2 ** half_num * base_dim] + embed_dims[::-1]
        self.patch_size = 2 ** (stage_num // 2)
        self.stage_num = stage_num
        self.half_num = half_num
        self.inconv = PatchEmbed(patch_size=1, in_chans=3, embed_dim=embed_dims[0], kernel_size=3)
        self.layers_restore_net = nn.ModuleList()
        # classify_net
        self.layers_classify_net = nn.ModuleList()
        # edge_net
        self.layers_edge_net = nn.ModuleList()
        # decoder_net
        self.layers_decoder = nn.ModuleList()
        self.downs_restore = nn.ModuleList()
        self.downs_classify = nn.ModuleList()
        self.downs_edge = nn.ModuleList()
        self.skips_restore = nn.ModuleList()
        self.skips_classify = nn.ModuleList()
        self.skips_edge = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.fusions = nn.ModuleList()
        self.grad_skip_connections = nn.ModuleList()
        self.sig = nn.Sigmoid()
        
        for i in range(self.half_num + 1):
            self.layers_restore_net.append(
                BasicLayer(dim=embed_dims[i], depth=depths[i], net_depth=net_depth, kernel_size=kernel_size,
                           conv_layer=conv_layer, norm_layer=norm_layer, gate_act=gate_act))

        for i in range(self.half_num + 1):
            self.layers_classify_net.append(
                BasicLayer(dim=embed_dims[i], depth=depths[i], net_depth=net_depth, kernel_size=kernel_size,
                           conv_layer=conv_layer, norm_layer=norm_layer, gate_act=gate_act))

        for i in range(self.half_num + 1):
            self.layers_edge_net.append(
                BasicLayer(dim=embed_dims[i], depth=depths[i], net_depth=net_depth, kernel_size=kernel_size,
                           conv_layer=conv_layer, norm_layer=norm_layer, gate_act=gate_act))
            
        for i in range(self.half_num + 1, self.stage_num):
            self.layers_decoder.append(
                BasicLayer(dim=embed_dims[i], depth=depths[i], net_depth=net_depth, kernel_size=kernel_size,
                           conv_layer=conv_layer, norm_layer=norm_layer, gate_act=gate_act))

        for i in range(self.half_num):
            self.downs_restore.append(PatchEmbed(patch_size=2, in_chans=embed_dims[i], embed_dim=embed_dims[i + 1]))
            self.downs_classify.append(PatchEmbed(patch_size=2, in_chans=embed_dims[i], embed_dim=embed_dims[i + 1]))
            self.downs_edge.append(PatchEmbed(patch_size=2, in_chans=embed_dims[i], embed_dim=embed_dims[i + 1]))

            self.skips_restore.append(nn.Conv2d(embed_dims[i], embed_dims[i], 1))
            self.skips_classify.append(nn.Conv2d(embed_dims[i], embed_dims[i], 1))
            self.skips_edge.append(nn.Conv2d(embed_dims[i], embed_dims[i], 1))
            self.grad_skip_connections.append(grad_skip_connection(dim=embed_dims[i]))
            self.ups.append(PatchUnEmbed(patch_size=2, out_chans=embed_dims[i], embed_dim=embed_dims[i + 1]))
            self.fusions.append(fusion_layer(embed_dims[i]))

        # output convolution
        self.outconv = PatchUnEmbed(patch_size=1, out_chans=3, embed_dim=embed_dims[-1], kernel_size=3)
        self.outconv_edge = nn.Sequential(
            nn.Conv2d(192, 3 * 8 ** 2, kernel_size=3, padding=3 // 2, padding_mode='reflect'),
            nn.PixelShuffle(8)
        )
        self.fusions_net_alpha = nn.Sequential(
            nn.Conv2d(192, 96, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(96, 192, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        self.fusions_net_beta = nn.Sequential(
            nn.Conv2d(192, 96, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(96, 192, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU()
        self.classify_1 = nn.Sequential(
            nn.Conv2d(192, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1)
        )
        self.classify_2 = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 3)
        )


    def forward(self, x):
        x_init = x.clone()
        feat = self.inconv(x)
        feat_edge_in = feat.clone()
        feat_restore_in = feat.clone()
        feat_classify_in = feat.clone()
        skips_edge = []
        skips_restore = []
        skips_classify = []

        for i in range(self.half_num):
            feat_restore_in = self.layers_restore_net[i](feat_restore_in)
            skips_restore.append(self.skips_restore[i](feat_restore_in))
            feat_restore_in = self.downs_restore[i](feat_restore_in)

        for i in range(self.half_num):
            feat_classify_in = self.layers_classify_net[i](feat_classify_in)
            skips_classify.append(self.skips_classify[i](feat_classify_in))
            feat_classify_in = self.downs_classify[i](feat_classify_in)

        for i in range(self.half_num):
            feat_edge_in = self.layers_edge_net[i](feat_edge_in)
            skips_edge.append(self.skips_edge[i](feat_edge_in))
            feat_edge_in = self.downs_edge[i](feat_edge_in)

        feat_restore_in = self.layers_restore_net[self.half_num](feat_restore_in)
        feat_classify_in = self.layers_classify_net[self.half_num](feat_classify_in)
        feat_edge_in = self.layers_edge_net[self.half_num](feat_edge_in)

        #融合三者的特征
        feat_alpha = self.fusions_net_alpha(feat_edge_in)
        feat_beta = self.fusions_net_beta(feat_classify_in)
        feat = self.relu((feat_restore_in * feat_alpha + feat_beta)+feat_restore_in)

        for i in range(self.half_num - 1, -1, -1):
            feat = self.ups[i](feat)
            feat = self.fusions[i]([feat, (skips_restore[i] * self.sig(self.grad_skip_connections[i](skips_edge[i]))
                                           + self.sig(self.grad_skip_connections[i](skips_classify[i])))+skips_restore[i]])
            feat = self.layers_decoder[self.stage_num - i - 5](feat)

        x = self.outconv(feat) + x_init
        edge = self.outconv_edge(feat_edge_in)
        out_class_step1 = self.classify_1(feat_classify_in)
        out_class_step1 = torch.mean(out_class_step1, dim=[2, 3])
        out_class = self.classify_2(out_class_step1)
        return x, edge, out_class


def mulunet():
    return MULUNET(kernel_size=5, base_dim=24, depths=[2, 4, 4, 6, 4, 4, 2], conv_layer=ConvLayer, norm_layer=nn.BatchNorm2d, gate_act=nn.Sigmoid, fusion_layer=SKFusion)
