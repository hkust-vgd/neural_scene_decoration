from math import log2

import numpy as np
import torch
from torch import nn


class Rezero(nn.Module):
    def __init__(self, fn):
        super(Rezero, self).__init__()
        self.fn = fn
        self.g = nn.Parameter(torch.tensor(1e-3))

    def forward(self, x):
        return self.g * self.fn(x)


class Residual(nn.Module):
    def __init__(self, fn):
        super(Residual, self).__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class SumBranches(nn.Module):
    def __init__(self, branches):
        super(SumBranches, self).__init__()
        self.branches = nn.ModuleList(branches)

    def forward(self, x):
        return sum(map(lambda fn: fn(x), self.branches))


class LabelsDownsampler(nn.Module):
    def __init__(self, layer_ids, image_size, label_mode='point'):
        super(LabelsDownsampler, self).__init__()
        layer_ids = sorted(layer_ids)[::-1]
        self.layers = layer_ids

        resolution = int(log2(image_size))
        layer_ids = np.array([resolution, *layer_ids])
        window_sizes = 2 ** (layer_ids[:-1] - layer_ids[1:])

        self.pool_layers = nn.ModuleList([])

        # if label_mode == 'point':
        #     for w in window_sizes:
        #         self.pool_layers.append(nn.MaxPool2d(kernel_size=w))
        # elif label_mode == 'box' or label_mode == 'semantic':
        for w in window_sizes:
            self.pool_layers.append(nn.AvgPool2d(kernel_size=w))

    def forward(self, x):
        labels = []

        for i, pool in zip(self.layers, self.pool_layers):
            x = pool(x)
            labels.append(x)

        return labels
