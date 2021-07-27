import math
import random
from functools import partial

import numpy as np
import scipy
import scipy.ndimage
import skimage
import skimage.transform
import skimage.util
import torch
import torch.nn.functional as F

from snapshotscope.utils import (
    _pair,
    _triple,
    cabs,
    cmul,
    ctensor_from_phase_angle,
    phase_angle_from_ctensor,
    gaussian_blur_3d,
    real,
    real_to_complex,
)


def create_augmentations(specs):
    augmentations = []
    template = "partial({name}, **{args})"
    for spec in specs:
        augmentation = eval(template.format(name=spec["name"], args=spec["args"]))
        augmentations.append(augmentation)
    return augmentations


def compose_augmentations(augmentations):
    def augment(x):
        out = x
        logs = []
        for augment in augmentations:
            out = augment(out)
            if isinstance(out, tuple):
                logs.append(out[1])
                out = out[0]
        if len(logs) == 0:
            return out
        else:
            return out, logs

    return augment


def interpolate_shape(vol, in_resolution, out_resolution, anti_aliasing=True):
    assert len(in_resolution) == len(out_resolution), "Sizes must match"
    target_shape = tuple(
        s * i / o for s, i, o in zip(vol.shape, in_resolution, out_resolution)
    )
    return skimage.transform.resize(
        vol,
        target_shape,
        mode="reflect",
        anti_aliasing=anti_aliasing,
        preserve_range=True,
    ).astype(vol.dtype)


def pool(vol, plane_downsample, downsample):
    vol = torch.from_numpy(vol)
    vol = F.avg_pool3d(
        vol.unsqueeze(0).unsqueeze(0),
        kernel_size=(plane_downsample, downsample, downsample),
        divisor_override=1,
    )
    vol = vol.squeeze(0).squeeze(0)
    return vol.numpy()


def downsample(vol, downsample=1):
    vol = torch.from_numpy(vol)
    is2d = len(vol.shape) == 2
    if is2d:
        vol = vol.unsqueeze(0)
    vol = F.avg_pool2d(
        vol.unsqueeze(1),
        kernel_size=(downsample, downsample),
        divisor_override=1,
    )
    vol = vol.squeeze(1)
    if is2d:
        vol = vol.squeeze(0)
    return vol.numpy()


def image_line_mask(im, percent=0.5):
    """
    Masks image to certain scan line region, e.g. percent=0.5 means center 50%
    of rows in the image will be kept and the remaining rows set to 0.
    """
    start_index = int((percent / 2) * im.shape[-2])
    end_index = int(start_index + (percent * im.shape[-2]))
    im[:start_index, :] *= 0
    im[end_index:, :] *= 0
    return im


def cylinder_crop(
    vol, radius, center=None, max_depth=101, valid_ratio=0.75, probability=1.0
):
    if (probability < 1.0) and (random.random() >= probability):
        return vol
    vol = torch.tensor(vol)
    if center == None:
        center = [
            random.randint(
                int((1 - valid_ratio) / 2 * vol.shape[d]),
                int((1 + valid_ratio) / 2 * vol.shape[d]),
            )
            for d in [1, 2]
        ]
    elif center == "center":
        center = [int(d / 2) for d in vol.shape[1:]]
    vol = vol[: min(max_depth, vol.shape[0])]
    y, x = torch.meshgrid(
        [
            torch.arange(s, dtype=torch.float32, device=vol.device)
            for s in (2 * radius + 1, 2 * radius + 1)
        ]
    )
    y = y - radius
    x = x - radius
    dist = (y.pow(2) + x.pow(2)).sqrt()
    circle = (dist <= radius).unsqueeze(0).to(torch.float32)
    shape = [
        min(center[-2] + radius + 1, vol.shape[1]) - max(center[-2] - radius, 0),
        min(center[-1] + radius + 1, vol.shape[2]) - max(center[-1] - radius, 0),
    ]
    cylinder = torch.zeros(vol.shape[0], shape[0], shape[1])
    circle = circle[:, : shape[0], : shape[1]]
    mask = circle.expand(*cylinder.shape)
    cylinder[:, :, :] = vol[
        :,
        max(center[-2] - radius, 0) : min(center[-2] + radius + 1, vol.shape[1]),
        max(center[-1] - radius, 0) : min(center[-1] + radius + 1, vol.shape[2]),
    ]
    cylinder *= mask
    return cylinder.squeeze().numpy()


def cylinder_mask(shape, radius):
    y, x = torch.meshgrid(
        [torch.linspace(-int(s / 2), int(s / 2), steps=s) for s in shape[1:]]
    )
    dist = (y.pow(2) + x.pow(2)).sqrt()
    circle = (dist <= radius).unsqueeze(0).to(torch.float32)
    mask = circle.expand(*shape)
    return mask


def quantile(vol, q):
    """
    Returns the array but with the maximum value set to the qth quantile
    (percentile).
    """
    return np.clip(vol, 0, np.quantile(vol, q))


def pixel_shuffle(vol):
    """
    Returns an array of the same shape and same entries in a randomly shuffled
    order, across all dimensions.
    """
    return np.reshape(np.random.permutation(np.ravel(vol)), vol.shape)


def phase_shuffle(vol):
    """
    Returns an array of the same shape as the input, but where the phase has
    been shuffled. We compute a Fourier transform of the input, then shuffle
    the phase component while leaving the amplitude component untouched. We
    then compute the inverse Fourier transform of the phase shuffled volume,
    and return the result.
    """
    vol = torch.from_numpy(vol)
    fourier_vol = torch.fft(real_to_complex(vol), 3)
    phase = torch.zeros_like(vol)
    phase.uniform_(-math.pi, math.pi)
    fourier_amplitude = cabs(fourier_vol)
    randomized_phase_vol = cmul(
        real_to_complex(fourier_amplitude), ctensor_from_phase_angle(phase)
    )
    reconstructed_vol = cabs(torch.ifft(randomized_phase_vol, 3))
    return reconstructed_vol.numpy()


def pad_volume(vol, target_shape, mode="constant", estimate=False, **kwargs):
    background = 0
    if estimate:
        background = np.mean(vol[0:10, 0:10, 0:10])
    return np.pad(
        vol,
        [
            (int((t - s) / 2), int((t - s + 1) / 2))
            for s, t in zip(vol.shape, target_shape)
        ],
        mode=mode,
        constant_values=background,
        **kwargs,
    )


def stack_and_crop(vol, probability=0.5):
    """
    Stacks given volume on top of itself and takes a random crop in z of the
    same number of planes as the given volume from this stacked volume.
    """
    if random.random() < probability:
        stacked = np.concatenate((vol, vol), axis=0)
        start = random.randint(0, vol.shape[0] - 1)
        return stacked[start : start + vol.shape[0]].copy()
    else:
        return vol


def vertical_jitter(vol, additional_planes=0, top_bottom_ratio=0.5):
    """
    Adds an additional_planes number of empty planes above or below a volume
    with a ratio of planes added to the top to planes added to the bottom
    specified by top_bottom_ratio. These additional planes serve to jitter the
    original volume in its vertical/z (0th) axis. If top_bottom_ratio is a
    tuple of floats, then the top to bottom ratio is chosen uniformly randomly
    within the given range.
    """
    top_bottom_ratio = np.random.uniform(*_pair(top_bottom_ratio))
    top_planes = int(top_bottom_ratio * additional_planes)
    bottom_planes = int(additional_planes - top_planes)
    return skimage.util.pad(vol, [(top_planes, bottom_planes), (0, 0), (0, 0)])


def horizontal_jitter(vol, jitter_amount=(0, 50), direction=("left", "right")):
    """
    Adds additional empty space to volume by by shifting volume in xy in the
    specified direction by jitter_amount and filling the empty space in the
    opposite direction by an estimated background value. If jitter_amount is a
    tuple, then jitter_amount is chosen uniformly randomly within the given
    range. If direction is a tuple of ('left', 'right'), the direction is
    chosen uniformly at random.
    """
    jitter_amount = random.randint(*_pair(jitter_amount))
    direction = random.choice(_pair(direction))
    background = np.mean(vol[0:10, 0:10, 0:10])
    vol = vol.copy()
    if direction is "left":
        vol[:, :, : vol.shape[2] - jitter_amount] = vol[:, :, jitter_amount:]
        vol[:, :, vol.shape[2] - jitter_amount : vol.shape[2]] = background
    else:
        vol[:, :, jitter_amount:] = vol[:, :, : vol.shape[2] - jitter_amount]
        vol[:, :, :jitter_amount] = background
    return vol


def slice_crop(vol, num_planes, center="center", probability=1.0):
    if (probability < 1.0) and (random.random() >= probability):
        return vol
    if center == "center":
        center = int(vol.shape[0] / 2)
    else:
        return NotImplemented
    start = center - int(num_planes / 2)
    end = start + num_planes
    vol = vol.copy()
    vol[0:start] = 0
    vol[end + 1 :] = 0
    return vol


def squash_range(vol, target_range, input_range=None):
    if input_range == None:
        imin = vol.min()
        imax = vol.max()
        input_range = (imin, imax)
    return (target_range[1] - target_range[0]) * (
        (vol - input_range[0]) / float(input_range[1] - input_range[0])
    ) + target_range[0]


def adjust_brightness(
    vol, background, brightness, shift=0, scale=1, blur_sigma=None, log=False
):
    if hasattr(brightness, "__iter__"):
        background = random.uniform(background[0], background[-1])
    if hasattr(brightness, "__iter__"):
        brightness = random.uniform(brightness[0], brightness[-1])
    if blur_sigma is not None:
        blur_sigma = _triple(blur_sigma)
        blur_kernel_size = tuple(int(3 * bs) for bs in blur_sigma)
        blur_kernel_size = tuple(ks if ks % 2 else ks + 1 for ks in blur_kernel_size)
        scale_map = gaussian_blur_3d(
            torch.from_numpy(vol), sigma=blur_sigma, kernel_size=blur_kernel_size
        )
        scale_map = scale_map.detach().cpu().numpy()
        scale_map -= scale_map.min()
        scale_map /= scale_map.max()
        background *= scale_map
    vol = vol - shift
    vol = vol * scale
    vol = vol + background
    vol = vol * brightness
    if log:
        return (vol, f"background: {background} brightness: {brightness}")
    return vol


def flip_planes(vol, num_flips="random", allowed_flips=[0, 1, 2, 3]):
    if num_flips == "random":
        num_flips = random.choice(allowed_flips)
    return np.rot90(vol, axes=(1, 2), k=num_flips).copy()


def reflect_planes(vol, probability=0.5):
    if random.random() < probability:
        vol = np.flip(vol, axis=2).copy()
    return vol


def flip_volume(vol, probability=0.5):
    if random.random() < probability:
        vol = np.flip(vol, axis=0).copy()
    return vol


def rotate_yaw(vol, angle):
    if hasattr(angle, "__iter__"):
        angle = random.uniform(angle[0], angle[-1])
    background = np.mean(vol[0:10, 0:10, 0:10])
    return scipy.ndimage.rotate(
        vol,
        angle,
        axes=(1, 2),
        reshape=False,
        mode="constant",
        cval=background,
    ).clip(background)


def rotate_pitch(vol, angle):
    if hasattr(angle, "__iter__"):
        angle = random.uniform(angle[0], angle[-1])
    background = np.mean(vol[0:10, 0:10, 0:10])
    return scipy.ndimage.rotate(
        vol,
        angle,
        axes=(0, 1),
        reshape=False,
        mode="constant",
        cval=background,
    ).clip(background)


def rotate_roll(vol, angle):
    if hasattr(angle, "__iter__"):
        angle = random.uniform(angle[0], angle[-1])
    background = np.mean(vol[0:10, 0:10, 0:10])
    return scipy.ndimage.rotate(
        vol,
        angle,
        axes=(0, 2),
        reshape=False,
        mode="constant",
        cval=background,
    ).clip(background)
