# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


# Preliminary code example for the paper  "On Learning Sets of Symmetric Elements",
# Maron et al. ICML 2020: basic Deep Sets for Symmetric Elements (DSS) layer for sets of images
# Full implementation will be released soon.
import torch
import torch.nn as nn


class Conv2dDeepSym(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size,padding=0,use_max=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_max = use_max
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv_s = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.bns = nn.BatchNorm2d(num_features=out_channels)
        torch.nn.init.xavier_normal_(self.conv.weight)
        torch.nn.init.xavier_normal_(self.conv_s.weight)

    def forward(self, x):
        b, n, c, h, w = x.size()
        x1 = self.bn(self.conv(x.view(n * b, c, h, w)))
        if self.use_max:
            x2 = self.bns(self.conv_s(torch.max(x, dim=1, keepdim=False)[0]))
        else:
            x2 = self.bns(self.conv_s(torch.sum(x, dim=1, keepdim=False)))
        x2 = x2.view(b, 1, h, w, self.out_channels).repeat(1, n, 1, 1, 1).view(b * n, self.out_channels, h, w)
        x = x1 + x2
        x = x.view(b, n, self.out_channels, h, w)
        return x
