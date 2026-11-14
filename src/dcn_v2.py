from __future__ import absolute_import, division, print_function

import torch
from torch import nn
from torchvision.ops import deform_conv2d


class DCN(nn.Module):
    """DCNv2-compatible wrapper backed by torchvision.ops."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation=1,
        deformable_groups=1,
    ):
        super(DCN, self).__init__()
        if isinstance(kernel_size, tuple):
            kh, kw = kernel_size
        else:
            kh = kw = kernel_size

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.deformable_groups = deformable_groups
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kh, kw))
        self.bias = nn.Parameter(torch.empty(out_channels))
        self.conv_offset_mask = nn.Conv2d(
            in_channels,
            deformable_groups * 3 * kh * kw,
            kernel_size=(kh, kw),
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=True,
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=1)
        nn.init.zeros_(self.bias)
        nn.init.zeros_(self.conv_offset_mask.weight)
        nn.init.zeros_(self.conv_offset_mask.bias)

    def forward(self, x):
        offset_mask = self.conv_offset_mask(x)
        offset_channels = offset_mask.shape[1] * 2 // 3
        offset = offset_mask[:, :offset_channels]
        mask = offset_mask[:, offset_channels:].sigmoid()
        return deform_conv2d(
            x,
            offset,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            mask=mask,
        )
