from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn

from .misc import _list


class ShardedSequential(nn.Sequential):
    """Wraps given modules similarly to nn.Sequential, but across GPUs."""

    def __init__(self, *args):
        super(ShardedSequential, self).__init__(*args)

    def forward(self, input):
        for i, module in enumerate(self._modules.values()):
            params = list(module.parameters())
            if len(params) > 0:
                input = input.to(params[0].device)
            input = module(input)
        return input


def mlp_block(
    num_inputs,
    layer_sizes,
    batch_norm=False,
    activate_last=False,
    name="mlp_block_{}",
):
    """
    Creates a block of linear layers with ReLUs.

    num_inputs (``int``)

        Number of elements in input.

    layer_sizes(``list``)

        Output sizes (integer) for each perceptron layer.

    name (``string``):

        Prefix used for naming all layers in this block.

    """
    layers = [(name.format(0), nn.Linear(num_inputs, layer_sizes[0]))]
    if len(layer_sizes) == 1 and activate_last:
        layers.append(((name + "_relu").format(0), nn.ReLU()))

    for i in range(1, len(layer_sizes)):
        layers.append((name.format(i), nn.Linear(layer_sizes[i - 1], layer_sizes[i])))
        if i != (len(layer_sizes) - 1) or activate_last:
            layers.append(((name + "_relu").format(i), nn.ReLU()))
        if batch_norm:
            layers.append(((name + "_bn").format(i), nn.BatchNorm1d(layer_sizes[i])))

    return nn.Sequential(OrderedDict(layers))


def conv2d_block(
    num_inputs,
    fmap_nums,
    kernel_sizes=None,
    batch_norm=True,
    activate_last=True,
    stride=1,
    padding=0,
    groups=1,
    dilations=None,
    devices=None,
    name="conv_block_{}",
):
    """
    Creates a block of 2D convolutions with specified kernel sizes and ReLUs.

    num_inputs (``int``):

        Number of input channels to convolve.

    fmap_nums (``list``):

        List of number of feature maps (integer) for each layer.

    kernel_sizes (``list``):

        List of kernel sizes (integer or tuple) for each layer.

    stride (``int`` or ``list``):

        Stride(s) for each layer.

    padding (``int`` or ``list``):

        Padding(s) for each layer.

    name (``string``):

        Prefix used for naming all layers in this block.

    """
    if kernel_sizes is None:
        kernel_sizes = [3 for i in range(len(fmap_nums))]
    if dilations is None:
        dilations = [1 for i in range(len(fmap_nums))]
    stride = _list(stride)
    padding = _list(padding)
    groups = _list(groups)
    if devices != None:
        devices = _list(devices)

    layers = [
        (
            name.format(0),
            nn.Conv2d(
                num_inputs,
                fmap_nums[0],
                kernel_sizes[0],
                stride[0],
                padding[0],
                dilation=dilations[0],
                groups=groups[0],
            ),
        )
    ]
    if devices != None:
        layers[-1][1].to(devices[0])
    if len(fmap_nums) == 1 and activate_last:
        layers.append(((name + "_relu").format(0), nn.LeakyReLU()))
        if devices != None:
            layers[-1][1].to(devices[0])

    for i in range(1, len(kernel_sizes)):
        layers.append(
            (
                name.format(i),
                nn.Conv2d(
                    fmap_nums[i - 1],
                    fmap_nums[i],
                    kernel_sizes[i],
                    stride=stride[i % len(stride)],
                    padding=padding[i % len(padding)],
                    dilation=dilations[i % len(dilations)],
                    groups=groups[i % len(groups)],
                ),
            )
        )
        if devices != None:
            layers[-1][1].to(devices[i % len(devices)])

        if i != (len(kernel_sizes) - 1) or activate_last:
            layers.append(((name + "_relu").format(i), nn.LeakyReLU()))
            if devices != None:
                layers[-1][1].to(devices[i % len(devices)])

        if batch_norm:
            layers.append(((name + "_bn").format(i), nn.BatchNorm2d(fmap_nums[i])))
            if devices != None:
                layers[-1][1].to(devices[i % len(devices)])
    if devices != None:
        return ShardedSequential(OrderedDict(layers))
    else:
        return nn.Sequential(OrderedDict(layers))


def calculate_shape(module, input_shape):
    """Calculates output shape of given module with specified input shape."""
    tmp = module(torch.ones(1, *input_shape, requires_grad=True))
    return tmp.size()[1:]


def calculate_numel(module, input_shape):
    """
    Calculates output flattened size of given module with specified input
    shape.
    """
    shape = calculate_shape(module, input_shape)
    return int(np.prod(shape))
