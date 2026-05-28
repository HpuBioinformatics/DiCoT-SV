"""
reference
- https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py

Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from .common import FrozenBatchNorm2d
from ..core import register
import logging

# Constants for initialization
kaiming_normal_ = nn.init.kaiming_normal_
zeros_ = nn.init.zeros_
ones_ = nn.init.ones_

__all__ = ['HGNetv2']


class LearnableAffineBlock(nn.Module):   # 线性变换：y = scale * x + bias
    def __init__(
            self,
            scale_value=1.0,
            bias_value=0.0
    ):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor([scale_value]), requires_grad=True)
        self.bias = nn.Parameter(torch.tensor([bias_value]), requires_grad=True)

    def forward(self, x):
        return self.scale * x + self.bias


class ConvBNAct(nn.Module):    #卷积+归一化+激活处理+线性增强    激活和非线性增强做不做可选
    def __init__(
            self,
            in_chs,
            out_chs,
            kernel_size,
            stride=1,
            groups=1,
            padding='',
            use_act=True,
            use_lab=False
    ):
        super().__init__()
        self.use_act = use_act
        self.use_lab = use_lab
        if padding == 'same':
            self.conv = nn.Sequential(
                nn.ZeroPad2d([0, 1, 0, 1]),
                nn.Conv2d(
                    in_chs,
                    out_chs,
                    kernel_size,
                    stride,
                    groups=groups,
                    bias=False
                )
            )
        else:
            self.conv = nn.Conv2d(
                in_chs,
                out_chs,
                kernel_size,
                stride,
                padding=(kernel_size - 1) // 2,
                groups=groups,
                bias=False
            )
        self.bn = nn.BatchNorm2d(out_chs)
        if self.use_act:   #self.use_act为True
            self.act = nn.ReLU()
        else:
            self.act = nn.Identity()
        if self.use_act and self.use_lab:
            self.lab = LearnableAffineBlock()
        else:
            self.lab = nn.Identity()    #什么都不做

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.lab(x)
        return x

class DirectionalConvBNAct(nn.Module):
    """
    Directional convolution block (Scheme-1 compatible with pretrained HGNetV2):
    (1x3 conv + 3x1 conv) -> concat -> 1x1 fusion
    Channel is preserved after fusion.
    """
    def __init__(
        self,
        in_chs,
        out_chs,
        stride=1,
        use_act=True,
        use_lab=False,
    ):
        super().__init__()

        assert in_chs == out_chs, \
            "DirectionalConvBNAct (scheme-1) requires in_chs == out_chs"

        # directional convs (do NOT change channels)
        self.conv_h = nn.Conv2d(
            in_chs,
            in_chs,
            kernel_size=(1, 3),
            stride=stride,
            padding=(0, 1),
            bias=False
        )

        self.conv_v = nn.Conv2d(
            in_chs,
            in_chs,
            kernel_size=(3, 1),
            stride=stride,
            padding=(1, 0),
            bias=False
        )

        # fuse 2C -> C
        self.fuse = nn.Conv2d(
            in_chs * 2,
            out_chs,
            kernel_size=1,
            bias=False
        )

        self.bn = nn.BatchNorm2d(out_chs)

        self.act = nn.ReLU() if use_act else nn.Identity()
        self.lab = LearnableAffineBlock() if (use_act and use_lab) else nn.Identity()

    def forward(self, x):
        x_h = self.conv_h(x)
        x_v = self.conv_v(x)

        x = torch.cat([x_h, x_v], dim=1)
        x = self.fuse(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.lab(x)

        return x


class LightConvBNAct(nn.Module):   #深度可分离卷积（轻量化CNN）
    def __init__(
            self,
            in_chs,
            out_chs,
            kernel_size,
            groups=1,
            use_lab=False,
    ):
        super().__init__()
        self.conv1 = ConvBNAct(
            in_chs,
            out_chs,
            kernel_size=1,
            use_act=False,
            use_lab=use_lab,
        )
        self.conv2 = ConvBNAct(
            out_chs,
            out_chs,
            kernel_size=kernel_size,
            groups=out_chs,
            use_act=True,
            use_lab=use_lab,
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class StemBlock(nn.Module):  #快速降采样（×4）+ 并行提取粗粒度 + 细粒度特征 + 融合后统一通道数，送入 stage1
    # for HGNetv2
    def __init__(self, in_chs, mid_chs, out_chs, use_lab=False):
        super().__init__()
        self.stem1 = ConvBNAct(  
            in_chs,
            mid_chs,
            kernel_size=3,
            stride=2,
            use_lab=use_lab,
        )
        self.stem2a = ConvBNAct(   
            mid_chs,
            mid_chs // 2,
            kernel_size=2,
            stride=1,
            use_lab=use_lab,
        )
        self.stem2b = ConvBNAct(  
            mid_chs // 2,
            mid_chs,
            kernel_size=2,
            stride=1,
            use_lab=use_lab,
        )
        self.stem3 = ConvBNAct(
            mid_chs * 2,
            mid_chs,
            kernel_size=3,
            stride=2,
            use_lab=use_lab,
        )
        self.stem4 = ConvBNAct(
            mid_chs,
            out_chs,
            kernel_size=1,
            stride=1,
            use_lab=use_lab,
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, ceil_mode=True) 

    def forward(self, x):
        x = self.stem1(x)             #下采样
        x = F.pad(x, (0, 1, 0, 1))
        x2 = self.stem2a(x)           #2a2b细粒度分支 使用2*2小卷积，捕捉比 3×3 更局部的变化
        x2 = F.pad(x2, (0, 1, 0, 1))
        x2 = self.stem2b(x2)
        x1 = self.pool(x)             #粗粒度分支
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x



class EseModule(nn.Module):       #通道注意力
    def __init__(self, chs):
        super().__init__()
        self.conv = nn.Conv2d(
            chs,
            chs,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x
        x = x.mean((2, 3), keepdim=True)    #全局平均池化
        x = self.conv(x)
        x = self.sigmoid(x)
        return torch.mul(identity, x)

# class HG_Block(nn.Module):        # 逐层提取特征 → 累积所有中间表示 → 通道拼接 → 注意力聚合 → 可选残差增强稳定性
#     def __init__(
#             self,
#             in_chs,
#             mid_chs,
#             out_chs,
#             layer_num,
#             kernel_size=3,
#             residual=False,
#             light_block=False,
#             use_lab=False,
#             agg='ese',
#             drop_path=0.,
#     ):
#         super().__init__()
#         self.residual = residual

#         self.layers = nn.ModuleList() #
#         for i in range(layer_num):
#             if light_block:
#                 self.layers.append(
#                     LightConvBNAct(
#                         in_chs if i == 0 else mid_chs,
#                         mid_chs,
#                         kernel_size=kernel_size,
#                         use_lab=use_lab,
#                     )
#                 )
#             else:
#                 self.layers.append(
#                     ConvBNAct(
#                         in_chs if i == 0 else mid_chs,
#                         mid_chs,
#                         kernel_size=kernel_size,
#                         stride=1,
#                         use_lab=use_lab,
#                     )
#                 )

#         # feature aggregation
#         total_chs = in_chs + layer_num * mid_chs
#         if agg == 'se':
#             aggregation_squeeze_conv = ConvBNAct(
#                 total_chs,
#                 out_chs // 2,
#                 kernel_size=1,
#                 stride=1,
#                 use_lab=use_lab,
#             )
#             aggregation_excitation_conv = ConvBNAct(
#                 out_chs // 2,
#                 out_chs,
#                 kernel_size=1,
#                 stride=1,
#                 use_lab=use_lab,
#             )
#             self.aggregation = nn.Sequential(
#                 aggregation_squeeze_conv,
#                 aggregation_excitation_conv,
#             )
#         else:
#             aggregation_conv = ConvBNAct(
#                 total_chs,
#                 out_chs,
#                 kernel_size=1,
#                 stride=1,
#                 use_lab=use_lab,
#             )
#             att = EseModule(out_chs)
#             self.aggregation = nn.Sequential(
#                 aggregation_conv,
#                 att,
#             )

#         self.drop_path = nn.Dropout(drop_path) if drop_path else nn.Identity()

#     def forward(self, x):
#         identity = x
#         output = [x]
#         for layer in self.layers:
#             x = layer(x)
#             output.append(x)
#         x = torch.cat(output, dim=1)
#         x = self.aggregation(x)
#         if self.residual:
#             x = self.drop_path(x) + identity
#         return x

class HG_Block(nn.Module):        # 逐层提取特征 → 累积所有中间表示 → 通道拼接 → 注意力聚合 → 可选残差增强稳定性
    def __init__(
            self,
            in_chs,
            mid_chs,
            out_chs,
            layer_num,
            kernel_size=3,
            residual=False,
            light_block=False,
            use_lab=False,
            agg='ese',
            drop_path=0.,
    ):
        super().__init__()
        self.residual = residual

        self.layers = nn.ModuleList() #
        # for i in range(layer_num):
        #     if light_block:
        #         self.layers.append(
        #             LightConvBNAct(
        #                 in_chs if i == 0 else mid_chs,
        #                 mid_chs,
        #                 kernel_size=kernel_size,
        #                 use_lab=use_lab,
        #             )
        #         )
        #     else:
        #         self.layers.append(
        #             ConvBNAct(
        #                 in_chs if i == 0 else mid_chs,
        #                 mid_chs,
        #                 kernel_size=kernel_size,
        #                 stride=1,
        #                 use_lab=use_lab,
        #             )
        #         )
        for i in range(layer_num):
            in_c = in_chs if i == 0 else mid_chs

            if light_block:
                 self.layers.append(
                    LightConvBNAct(
                        in_c,
                        mid_chs,
                        kernel_size=kernel_size,
                        use_lab=use_lab,
                    )
                )
            else:
                # ✅ 只替换「3×3 + 不变通道」的位置
                if kernel_size == 3 and in_c == mid_chs:
                    self.layers.append(
                        DirectionalConvBNAct(
                            in_c,
                            mid_chs,
                            stride=1,
                            use_lab=use_lab,
                        )
                    )
                else:
                    self.layers.append(
                        ConvBNAct(
                            in_c,
                            mid_chs,
                            kernel_size=kernel_size,
                            stride=1,
                            use_lab=use_lab,
                        )
                    )       


        # feature aggregation
        total_chs = in_chs + layer_num * mid_chs
        if agg == 'se':
            aggregation_squeeze_conv = ConvBNAct(
                total_chs,
                out_chs // 2,
                kernel_size=1,
                stride=1,
                use_lab=use_lab,
            )
            aggregation_excitation_conv = ConvBNAct(
                out_chs // 2,
                out_chs,
                kernel_size=1,
                stride=1,
                use_lab=use_lab,
            )
            self.aggregation = nn.Sequential(
                aggregation_squeeze_conv,
                aggregation_excitation_conv,
            )
        else:
            aggregation_conv = ConvBNAct(
                total_chs,
                out_chs,
                kernel_size=1,
                stride=1,
                use_lab=use_lab,
            )
            att = EseModule(out_chs)
            self.aggregation = nn.Sequential(
                aggregation_conv,
                att,
            )

        self.drop_path = nn.Dropout(drop_path) if drop_path else nn.Identity()

    def forward(self, x):
        identity = x
        output = [x]
        for layer in self.layers:
            x = layer(x)
            output.append(x)
        x = torch.cat(output, dim=1)
        x = self.aggregation(x)
        if self.residual:
            x = self.drop_path(x) + identity
        return x
    

    


class HG_Stage(nn.Module):     #下采样+ stage的设计
    def __init__(
            self,
            in_chs,
            mid_chs,
            out_chs,
            block_num,
            layer_num,
            downsample=True,
            light_block=False,
            kernel_size=3,
            use_lab=False,
            agg='se',
            drop_path=0.,
    ):
        super().__init__()
        self.downsample = downsample
        if downsample:
            self.downsample = ConvBNAct(
                in_chs,
                in_chs,
                kernel_size=3,
                stride=2,
                groups=in_chs,
                use_act=False,
                use_lab=use_lab,
            )
        else:
            self.downsample = nn.Identity()

        blocks_list = []
        for i in range(block_num):
            blocks_list.append(
                HG_Block(
                    in_chs if i == 0 else out_chs,
                    mid_chs,
                    out_chs,
                    layer_num,
                    residual=False if i == 0 else True,
                    kernel_size=kernel_size,
                    light_block=light_block,
                    use_lab=use_lab,
                    agg=agg,
                    drop_path=drop_path[i] if isinstance(drop_path, (list, tuple)) else drop_path,
                )
            )
        self.blocks = nn.Sequential(*blocks_list)  #把多个子模块按顺序串起来

    def forward(self, x):
        x = self.downsample(x)
        x = self.blocks(x)
        return x



@register()
class HGNetv2(nn.Module):
    """
    HGNetV2
    Args:
        stem_channels: list. Number of channels for the stem block.
        stage_type: str. The stage configuration of HGNet. such as the number of channels, stride, etc.
        use_lab: boolean. Whether to use LearnableAffineBlock in network.
        lr_mult_list: list. Control the learning rate of different stages.
    Returns:
        model: nn.Layer. Specific HGNetV2 model depends on args.
    """

    arch_configs = {
        'B0': {
            'stem_channels': [3, 16, 16],
            'stage_config': {
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [16, 16, 64, 1, False, False, 3, 3],
                "stage2": [64, 32, 256, 1, True, False, 3, 3],
                "stage3": [256, 64, 512, 2, True, True, 5, 3],
                "stage4": [512, 128, 1024, 1, True, True, 5, 3],
            },
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B0_stage1.pth'
        },
        'B1': {
            'stem_channels': [3, 24, 32],
            'stage_config': {
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [32, 32, 64, 1, False, False, 3, 3],
                "stage2": [64, 48, 256, 1, True, False, 3, 3],
                "stage3": [256, 96, 512, 2, True, True, 5, 3],
                "stage4": [512, 192, 1024, 1, True, True, 5, 3],
            },
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B1_stage1.pth'
        },
        'B2': {
            'stem_channels': [3, 24, 32],
            'stage_config': {
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [32, 32, 96, 1, False, False, 3, 4],
                "stage2": [96, 64, 384, 1, True, False, 3, 4],
                "stage3": [384, 128, 768, 3, True, True, 5, 4],
                "stage4": [768, 256, 1536, 1, True, True, 5, 4],
            },
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B2_stage1.pth'
        },
        'B3': {
            'stem_channels': [3, 24, 32],
            'stage_config': {
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [32, 32, 128, 1, False, False, 3, 5],
                "stage2": [128, 64, 512, 1, True, False, 3, 5],
                "stage3": [512, 128, 1024, 3, True, True, 5, 5],
                "stage4": [1024, 256, 2048, 1, True, True, 5, 5],
            },
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B3_stage1.pth'
        },
        'B4': {
            'stem_channels': [3, 32, 48],
            'stage_config': {
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [48, 48, 128, 1, False, False, 3, 6],
                "stage2": [128, 96, 512, 1, True, False, 3, 6],
                "stage3": [512, 192, 1024, 3, True, True, 5, 6],
                "stage4": [1024, 384, 2048, 1, True, True, 5, 6],
            },
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B4_stage1.pth'
        },
        'B5': {
            'stem_channels': [3, 32, 64],
            'stage_config': {
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [64, 64, 128, 1, False, False, 3, 6],
                "stage2": [128, 128, 512, 2, True, False, 3, 6],
                "stage3": [512, 256, 1024, 5, True, True, 5, 6],
                "stage4": [1024, 512, 2048, 2, True, True, 5, 6],
            },
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B5_stage1.pth'
        },
        'B6': {
            'stem_channels': [3, 48, 96],
            'stage_config': {
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [96, 96, 192, 2, False, False, 3, 6],
                "stage2": [192, 192, 512, 3, True, False, 3, 6],
                "stage3": [512, 384, 1024, 6, True, True, 5, 6],
                "stage4": [1024, 768, 2048, 3, True, True, 5, 6],
            },
            'url': 'https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B6_stage1.pth'
        },
    }

    def __init__(self,
                 name,
                 use_lab=False,
                 return_idx=[1, 2, 3],       #返回哪些stage的输出
                 freeze_stem_only=True,
                 freeze_at=0,
                 freeze_norm=True,
                 pretrained=True,
                 local_model_dir='weight/hgnetv2/'):
        super().__init__()
        self.use_lab = use_lab
        self.return_idx = return_idx

        stem_channels = self.arch_configs[name]['stem_channels']
        stage_config = self.arch_configs[name]['stage_config']
        download_url = self.arch_configs[name]['url']

        self._out_strides = [4, 8, 16, 32]
        self._out_channels = [stage_config[k][2] for k in stage_config]

        # stem
        self.stem = StemBlock(
                in_chs=stem_channels[0],
                mid_chs=stem_channels[1],
                out_chs=stem_channels[2],
                use_lab=use_lab)

        # stages
        self.stages = nn.ModuleList()
        for i, k in enumerate(stage_config):
            in_channels, mid_channels, out_channels, block_num, downsample, light_block, kernel_size, layer_num = stage_config[
                k]
            self.stages.append(
                HG_Stage(
                    in_channels,
                    mid_channels,
                    out_channels,
                    block_num,
                    layer_num,
                    downsample,
                    light_block,
                    kernel_size,
                    use_lab))

        if freeze_at >= 0:
            self._freeze_parameters(self.stem)
            if not freeze_stem_only:
                for i in range(min(freeze_at + 1, len(self.stages))):
                    self._freeze_parameters(self.stages[i])

        if freeze_norm:
            self._freeze_norm(self)

        if pretrained:
            RED, GREEN, RESET = "\033[91m", "\033[92m", "\033[0m"
            try:
                model_path = local_model_dir + 'PPHGNetV2_' + name + '_stage1.pth'
                if os.path.exists(model_path):
                    state = torch.load(model_path, map_location='cpu')
                    print(f"Loaded stage1 {name} HGNetV2 from local file.")
                else:
                    # If the file doesn't exist locally, download from the URL
                    # if torch.distributed.get_rank() == 0:
                    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                        print(GREEN + "If the pretrained HGNetV2 can't be downloaded automatically. Please check your network connection." + RESET)
                        print(GREEN + "Please check your network connection. Or download the model manually from " + RESET + f"{download_url}" + GREEN + " to " + RESET + f"{local_model_dir}." + RESET)
                        state = torch.hub.load_state_dict_from_url(download_url, map_location='cpu', model_dir=local_model_dir)
                        torch.distributed.barrier()
                    else:
                        torch.distributed.barrier()
                        state = torch.load(local_model_dir)

                    print(f"Loaded stage1 {name} HGNetV2 from URL.")

                # self.load_state_dict(state)
                missing, unexpected = self.load_state_dict(state, strict=False)

                if (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0:
                    print('[HGNetV2] Load pretrained with strict=False')
                    print('  Missing keys:', len(missing))
                    print('  Unexpected keys:', len(unexpected))


            except (Exception, KeyboardInterrupt) as e:
                if torch.distributed.get_rank() == 0:
                    print(f"{str(e)}")
                    logging.error(RED + "CRITICAL WARNING: Failed to load pretrained HGNetV2 model" + RESET)
                    logging.error(GREEN + "Please check your network connection. Or download the model manually from " \
                                + RESET + f"{download_url}" + GREEN + " to " + RESET + f"{local_model_dir}." + RESET)
                exit()




    def _freeze_norm(self, m: nn.Module):
        if isinstance(m, nn.BatchNorm2d):
            m = FrozenBatchNorm2d(m.num_features)
        else:
            for name, child in m.named_children():
                _child = self._freeze_norm(child)
                if _child is not child:
                    setattr(m, name, _child)
        return m

    def _freeze_parameters(self, m: nn.Module):
        for p in m.parameters():
            p.requires_grad = False

    def forward(self, x):
        x = self.stem(x)
        outs = []
        for idx, stage in enumerate(self.stages):
            x = stage(x)
            if idx in self.return_idx:
                outs.append(x)
        return outs
