from fouriernet import *
import torch.nn as nn

from snapshotscope.utils import *
from snapshotscope.utils import _list, _pair, _triple

from .unet import UNet3DVolumeReconstructor, UNet2D as UNet2DVolumeReconstructor


def UNet2D(
    in_channels,
    out_channels,
    f_maps=64,
    layer_order="crb",
    num_levels=4,
    conv_padding=1,
    input_scaling_mode="batchnorm",
    quantile=0.5,
    quantile_scale=1.0,
    device="cpu",
    **kwargs,
):
    # create image batch norm
    if input_scaling_mode in ["scaling", "norming"]:
        layers = []
    elif input_scaling_mode in ["batchnorm"]:
        if not kwargs['imbn_momentum']:
            imbn_momentum = 0.1
        imbn = nn.BatchNorm3d(1, momentum=imbn_momentum).to(device)
        layers = [imbn]
    else:
        raise TypeError(
            "Argument input_scaling_mode must be one of 'scaling', 'norming', or 'batchnorm'"
        )
    # create unet layers
    layers += [
        UNet2DVolumeReconstructor(
            in_channels,
            out_channels,
            final_sigmoid=False,
            f_maps=f_maps,
            layer_order=layer_order,
            num_levels=num_levels,
            is_segmentation=False,
            conv_padding=conv_padding,
            **kwargs,
        ).to(device),
        nn.ReLU().to(device),
    ]
    # construct and return sequential module
    if input_scaling_mode == "scaling":
        reconstruct = InputScalingSequential(quantile, quantile_scale, *layers)
    elif input_scaling_mode == "norming":
        reconstruct = InputNormingSequential((0, 2, 3, 4), *layers)
    else:
        reconstruct = nn.Sequential(*layers)
    return reconstruct


def UNet3D(
    in_channels,
    out_channels,
    num_planes,
    f_maps=64,
    layer_order="crb",
    num_levels=4,
    conv_padding=1,
    input_scaling_mode="batchnorm",
    quantile=0.5,
    quantile_scale=1.0,
    device="cpu",
    **kwargs,
):
    # create image batch norm
    if input_scaling_mode in ["scaling", "norming"]:
        layers = []
    elif input_scaling_mode in ["batchnorm"]:
        if not kwargs['imbn_momentum']:
            imbn_momentum = 0.1
        imbn = nn.BatchNorm3d(1, momentum=imbn_momentum).to(device)
        layers = [imbn]
    else:
        raise TypeError(
            "Argument input_scaling_mode must be one of 'scaling', 'norming', or 'batchnorm'"
        )
    # create unet layers
    layers += [
        UNet3DVolumeReconstructor(
            in_channels,
            out_channels,
            num_planes,
            final_sigmoid=False,
            f_maps=f_maps,
            layer_order=layer_order,
            num_levels=num_levels,
            is_segmentation=False,
            conv_padding=conv_padding,
            **kwargs,
        ).to(device),
        nn.ReLU().to(device),
    ]
    # construct and return sequential module
    if input_scaling_mode == "scaling":
        reconstruct = InputScalingSequential(quantile, quantile_scale, *layers)
    elif input_scaling_mode == "norming":
        reconstruct = InputNormingSequential((0, 2, 3, 4), *layers)
    else:
        reconstruct = nn.Sequential(*layers)
    return reconstruct

