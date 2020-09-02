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
