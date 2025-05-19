#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
===============================================================================
IMU ResNetEncoder Model Script
-------------------------------------------------------------------------------
Author      : Joseph Yep
Email       : yewentai126@gmail.com
Edited      : 2025-05-19
Description : Defines a modular ResNetEncoder architecture for IMU data. Includes:
              - weight loading utility for checkpoints (supports distributed models)
              - anti-aliased downsampling (1D)
              - residual building blocks (1D)
              - sequential stage construction (make_layer)
              - optional feature freezing for transfer learning
              - final Encoder wrapper that flattens extracted features
===============================================================================
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def load_weights(weight_path, model, device="cpu", name_start_idx=2, is_dist=False):
    # only need to change weights name when the
    # model is trained in a distributed manner

    pretrained_dict = torch.load(weight_path, map_location=device, weights_only=True)
    pretrained_dict_v2 = copy.deepcopy(pretrained_dict)  # v2 has the right para names

    if is_dist:
        for key in pretrained_dict:
            para_names = key.split(".")
            new_key = ".".join(para_names[name_start_idx:])
            pretrained_dict_v2[new_key] = pretrained_dict_v2.pop(key)

    model_dict = model.state_dict()

    # 1. filter out unnecessary keys such as the final linear layers
    #    we don't want linear layer weights either
    pretrained_dict = {
        k: v for k, v in pretrained_dict_v2.items() if k in model_dict and k.split(".")[0] != "classifier"
    }

    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)

    # 3. load the new state dict
    model.load_state_dict(model_dict)


class Downsample(nn.Module):
    r"""Downsampling layer that applies anti-aliasing filters.
    For example, order=0 corresponds to a box filter (or average downsampling
    -- this is the same as AvgPool in Pytorch), order=1 to a triangle filter
    (or linear downsampling), order=2 to cubic downsampling, and so on.
    See https://richzhang.github.io/antialiased-cnns/ for more details.
    """

    def __init__(self, channels=None, factor=2, order=1):
        super(Downsample, self).__init__()
        assert factor > 1, "Downsampling factor must be > 1"
        self.stride = factor
        self.channels = channels
        self.order = order

        # Figure out padding and check params make sense
        # The padding is given by order*(factor-1)/2
        # so order*(factor-1) must be divisible by 2
        total_padding = order * (factor - 1)
        assert total_padding % 2 == 0, (
            "Misspecified downsampling parameters."
            "Downsampling factor and order must be such "
            "that order*(factor-1) is divisible by 2"
        )
        self.padding = int(order * (factor - 1) / 2)

        box_kernel = np.ones(factor)
        kernel = np.ones(factor)
        for _ in range(order):
            kernel = np.convolve(kernel, box_kernel)
        kernel /= np.sum(kernel)
        kernel = torch.Tensor(kernel)
        self.register_buffer("kernel", kernel[None, None, :].repeat((channels, 1, 1)))

    def forward(self, x):
        return F.conv1d(
            x,
            self.kernel,
            stride=self.stride,
            padding=self.padding,
            groups=x.shape[1],
        )


class ResBlock(nn.Module):
    r""" Basic bulding block in Resnets:

       bn-relu-conv-bn-relu-conv
      /                         \
    x --------------------------(+)->

    """

    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, padding=2):

        super(ResBlock, self).__init__()

        self.bn1 = nn.BatchNorm1d(in_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=False,
            padding_mode="circular",
        )
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=False,
            padding_mode="circular",
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        x = self.relu(self.bn1(x))
        x = self.conv1(x)
        x = self.relu(self.bn2(x))
        x = self.conv2(x)

        x = x + identity

        return x


class ResNet(nn.Module):
    """The general form of the architecture can be described as follows:

    x->[Conv-[ResBlock]^m-BN-ReLU-Down]^n->y

    In other words:

            bn-relu-conv-bn-relu-conv                        bn-
           /                         \                      /
    x->conv --------------------------(+)-bn-relu-down-> conv ----

    """

    def __init__(
        self,
        n_channels=3,
    ):
        super(ResNet, self).__init__()

        cgf = [
            (64, 5, 2, 5, 2, 2),
            (128, 5, 2, 5, 2, 2),
            (256, 5, 2, 5, 5, 1),
            (512, 5, 2, 5, 5, 1),
            (1024, 5, 0, 5, 3, 1),
        ]
        in_channels = n_channels
        feature_extractor = nn.Sequential()
        for i, layer_params in enumerate(cgf):
            (
                out_channels,
                conv_kernel_size,
                n_resblocks,
                resblock_kernel_size,
                downfactor,
                downorder,
            ) = layer_params
            feature_extractor.add_module(
                f"layer{i+1}",
                ResNet.make_layer(
                    in_channels,
                    out_channels,
                    conv_kernel_size,
                    n_resblocks,
                    resblock_kernel_size,
                    downfactor,
                    downorder,
                ),
            )
            in_channels = out_channels

        self.feature_extractor = feature_extractor

    @staticmethod
    def make_layer(
        in_channels,
        out_channels,
        conv_kernel_size,
        n_resblocks,
        resblock_kernel_size,
        downfactor,
        downorder=1,
    ):
        r""" Basic layer in Resnets:

        x->[Conv-[ResBlock]^m-BN-ReLU-Down]->

        In other words:

                bn-relu-conv-bn-relu-conv
               /                         \
        x->conv --------------------------(+)-bn-relu-down->

        """

        # Check kernel sizes make sense (only odd numbers are supported)
        assert conv_kernel_size % 2, "Only odd number for conv_kernel_size supported"
        assert resblock_kernel_size % 2, "Only odd number for resblock_kernel_size supported"

        # Figure out correct paddings
        conv_padding = int((conv_kernel_size - 1) / 2)
        resblock_padding = int((resblock_kernel_size - 1) / 2)

        modules = [
            nn.Conv1d(
                in_channels,
                out_channels,
                conv_kernel_size,
                1,
                conv_padding,
                bias=False,
                padding_mode="circular",
            )
        ]

        for i in range(n_resblocks):
            modules.append(
                ResBlock(
                    out_channels,
                    out_channels,
                    resblock_kernel_size,
                    1,
                    resblock_padding,
                )
            )

        modules.append(nn.BatchNorm1d(out_channels))
        modules.append(nn.ReLU(True))
        modules.append(Downsample(out_channels, downfactor, downorder))

        return nn.Sequential(*modules)


class ResNetEncoder(nn.Module):
    def __init__(self, weight_path=None, n_channels=3, device="cpu", freeze=False):
        """
        Loads the pre-trained ResNetEncoder model, removes its classifier head, and
        outputs flattened features from the feature extractor.

        Parameters:
            weight_path (str): Path to the pre-trained weights.
            n_channels (int): Number of input channels.
            device (str): Device for loading the weights.
            freeze (bool): If True, freeze encoder weights.
        """
        super(ResNetEncoder, self).__init__()
        # Create the ResNetEncoder model with is_eva=True to use the two-layer FC head in pre-training.
        # (The classifier head will be discarded.)
        self.resnet = ResNet(
            n_channels=n_channels,
        )
        # Load pre-trained weights. The load_weights function adapts the parameter names if needed.
        if weight_path is not None:
            load_weights(weight_path, self.resnet, device=device, is_dist=True, name_start_idx=1)

        # Freeze encoder parameters if requested.
        if freeze:
            for param in self.resnet.parameters():
                param.requires_grad = False

        # Save the feature extractor, which is all layers before the classifier.
        self.feature_extractor = self.resnet.feature_extractor

        self.out_features = 1024

    def forward(self, x):
        # x is expected to be of shape [B, channels, seq_len]
        feats = self.feature_extractor(x)
        feats = feats.view(x.shape[0], -1)  # flatten
        return feats


class MLPClassifier(nn.Module):
    def __init__(self, output_size, input_size=1024, nn_size=512):
        super(MLPClassifier, self).__init__()
        self.linear1 = torch.nn.Linear(input_size, nn_size)
        self.linear2 = torch.nn.Linear(nn_size, output_size)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x
