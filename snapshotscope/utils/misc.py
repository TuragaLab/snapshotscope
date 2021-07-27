import math
from itertools import repeat

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .complex import fftconvn


def pearson_corr(x, y):
    x = x - x.mean()
    y = y - y.mean()
    corr = (
        torch.sum(x * y)
        * torch.rsqrt(torch.sum(x ** 2))
        * torch.rsqrt(torch.sum(y ** 2))
    )
    return corr


def rotate_indices(indices, shape, pitch, yaw, roll):
    angles = torch.tensor([pitch, yaw, roll], dtype=torch.float32)
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    rotation_z = torch.tensor(
        [[cos[0], -sin[0], 0], [sin[0], cos[0], 0], [0, 0, 1]],
        dtype=torch.float32,
        device=indices.device,
    )
    rotation_y = torch.tensor(
        [[cos[1], 0, sin[1]], [0, 1, 0], [-sin[1], 0, cos[1]]],
        dtype=torch.float32,
        device=indices.device,
    )
    rotation_x = torch.tensor(
        [[1, 0, 0], [0, cos[2], -sin[2]], [0, sin[2], cos[2]]],
        dtype=torch.float32,
        device=indices.device,
    )
    rotation = torch.matmul(torch.matmul(rotation_z, rotation_y), rotation_x)
    indices = indices.clone().to(torch.float32)
    tmp = indices[0].clone()
    indices[0] = indices[2]
    indices[2] = tmp
    indices -= torch.tensor(
        [[shape[2] / 2.0], [shape[1] / 2.0], [shape[0] / 2.0]],
        device=indices.device,
    )
    indices = torch.matmul(rotation, indices.to(torch.float32))
    indices += torch.tensor(
        [[shape[2] / 2.0], [shape[1] / 2.0], [shape[0] / 2.0]],
        device=indices.device,
    )
    indices = indices.to(torch.long)
    tmp = indices[0].clone()
    indices[0] = indices[2]
    indices[2] = tmp
    for i, s in enumerate(shape):
        indices[i] = torch.clamp(indices[i], 0, s - 1)
    return indices


def gaussian_kernel_2d(sigma, kernel_size, device="cpu"):
    kernel_size = _pair(kernel_size)
    y, x = torch.meshgrid(
        (
            torch.arange(kernel_size[0]).float(),
            torch.arange(kernel_size[1]).float(),
        )
    )
    y_mean = (kernel_size[0] - 1) / 2.0
    x_mean = (kernel_size[1] - 1) / 2.0
    kernel = 1
    kernel *= (
        1
        / (sigma * math.sqrt(2 * math.pi))
        * torch.exp(-(((y - y_mean) / (2 * sigma)) ** 2))
    )
    kernel *= (
        1
        / (sigma * math.sqrt(2 * math.pi))
        * torch.exp(-(((x - x_mean) / (2 * sigma)) ** 2))
    )
    kernel = kernel / torch.sum(kernel)
    kernel = kernel.to(device)
    return kernel


def gaussian_blur_2d(vol2d, sigma=0.1, kernel_size=(3, 3), kernel=None, device="cpu"):
    if kernel is None:
        kernel_size = _pair(kernel_size)
        kernel = gaussian_kernel_2d(sigma, kernel_size, device=device)
        kernel = kernel.expand(vol2d.shape[1], vol2d.shape[1], *kernel.shape)
    kernel_size = kernel.shape[2:]
    padding = tuple(math.floor(kernel_size[d] / 2.0) for d in [0, 0, 1, 1])
    return F.conv2d(
        F.pad(vol2d, padding, mode="reflect"), kernel, groups=kernel.shape[0]
    ).squeeze()


def gaussian_kernel_3d(
    sigma=1.0, kernel_size=(21, 21, 21), pixel_size=1.0, device="cpu"
):
    """Creates a 3D Gaussian kernel of specified size and pixel size."""
    kernel_size = _triple(kernel_size)
    sigma = _triple(sigma)
    z, y, x = torch.meshgrid(
        [
            torch.linspace(
                -(kernel_size[d] - 1) / 2,
                (kernel_size[d] - 1) / 2,
                steps=kernel_size[d],
            )
            * pixel_size
            for d in range(3)
        ]
    )
    kernel = 1
    for d, s in zip([z, y, x], sigma):
        d_mean = d[tuple(int(d.shape[i] / 2) for i in range(3))]
        kernel *= (1 / (s * math.sqrt(2 * math.pi))) * torch.exp(
            -(((d - d_mean) / (2 * s)) ** 2)
        )
    kernel = kernel / torch.sum(kernel)
    return kernel.to(device)


def gaussian_blur_3d(
    vol3d,
    sigma=1.0,
    kernel_size=(21, 21, 21),
    pixel_size=1.0,
    kernel=None,
    device="cpu",
):
    """Performs 3D blur using the specified kernel and FFT convolution."""
    if kernel is None:
        kernel = gaussian_kernel_3d(sigma, kernel_size, pixel_size, device)
    return fftconvn(vol3d, kernel, shape="same")


def create_fourier_plane(num_pixels, dk):
    """Creates the coordinates for a Fourier plane."""
    k = (
        np.linspace(
            -(num_pixels - 1) / 2,
            (num_pixels - 1) / 2,
            num=num_pixels,
            endpoint=True,
        )
        * dk
    )
    return np.meshgrid(k, k)


def split_defocus_range(defocus_range, num_chunks):
    """Returns a list of chunked arrays spanning the original defocus_range."""
    chunk_size = int(len(defocus_range) / num_chunks)
    if len(defocus_range) % num_chunks != 0:
        chunk_size += 1
    sections = [i * chunk_size for i in range(1, num_chunks)]
    defocus_ranges = np.split(defocus_range, sections)
    return defocus_ranges


def combinatorial(n, k):
    """Calculates the combination C(n, k), i.e. n choose k."""
    return math.factorial(n) / (math.factorial(k) * math.factorial(n - k))


def _list(x, repetitions=1):
    if hasattr(x, "__iter__") and not isinstance(x, str):
        return x
    else:
        return [
            x,
        ] * repetitions


def _ntuple(n):
    """Creates a function enforcing ``x`` to be a tuple of ``n`` elements."""

    def parse(x):
        if isinstance(x, tuple):
            return x
        return tuple(repeat(x, n))

    return parse


_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
