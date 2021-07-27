import copy
import math
import random

import numpy as np
import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.ndimage.filters import gaussian_filter
from scipy.signal import fftconvolve
from scipy.special import expit as sigmoid
from skimage.morphology import erosion

from learnedLFM.utils import *

NP_DTYPE = np.complex64
T_DTYPE = torch.float32
# NP_DTYPE=np.complex128
# T_DTYPE=torch.float64


def poissonlike_gaussian_noise(im):
    im = im + (im.sqrt()) * torch.randn_like(im)
    im.clamp_(min=0.0)
    return im


def sparse_point_sources(
    sample_size,
    sparsity=1e-3,
    brightness=500,
    background=1e-2,
    device="cpu",
    dtype=torch.float32,
):
    sample = (
        torch.rand(sample_size, device=device, requires_grad=False) < sparsity
    ).to(dtype)
    sample = sample * torch.randn(sample_size, device=device, requires_grad=False).abs()
    sample = (
        (sample + background * (0.5 + torch.rand([])))
        * brightness
        * (0.5 + torch.rand([]))
    )
    return sample


def sparse_point_sources_coo(
    sample_size,
    locs=None,
    num_sources=5000,
    brightness=500,
    background=1e-2,
    device="cpu",
    dtype=torch.float32,
):
    if locs is None:
        locs = []
        for dim_max in list(sample_size):
            locs.append(torch.rand(num_sources, device=device) * dim_max)
    else:
        num_sources = locs[0].size(0)
    vals = torch.ones(num_sources)
    sample = torch.sparse_coo_tensor(
        torch.stack(locs), vals, size=sample_size, device=device
    ).to_dense()
    sample = (sample + background) * brightness
    return sample, locs


def jitter_point_sources(sample_size, locs, sigmas=[10.0, 10.0, 1.0]):
    num_sources = locs[0].size(0)
    for loc, dim_max, sig in zip(locs, list(sample_size), list(sigmas)):
        loc += torch.randn_like(loc) * sig
        loc.clamp_(min=0, max=dim_max - 1)
    return locs


def sparse_balls_scipy(
    sample_size,
    ball_radius=2.5,
    xy_spacing=0.1,
    z_spacing=3.0,
    sparsity=1e-5,
    brightness=500,
    background=1e-2,
    device="cpu",
    dtype=torch.float32,
):
    z, y, x = np.meshgrid(
        np.arange(-ball_radius, ball_radius + 1e-3, step=z_spacing),
        np.arange(-ball_radius, ball_radius + 1e-3, step=xy_spacing),
        np.arange(-ball_radius, ball_radius + 1e-3, step=xy_spacing),
    )
    ball = np.sqrt(x ** 2 + y ** 2 + z ** 2) < ball_radius
    ballsize = ball.shape
    sample = np.random.rand(*sample_size) < sparsity
    sample = fftconvolve(sample, ball, mode="same")
    sample = torch.tensor(
        (sample + background) * brightness, dtype=dtype, device=device
    )
    return sample


def sparse_balls(
    sample_size,
    ball_radius=5.0,
    xy_spacing=0.1,
    z_spacing=3.0,
    sparsity=1e-5,
    brightness=500,
    background=1e-2,
    separate=False,
    device="cpu",
    dtype=torch.float32,
):
    z, y, x = torch.meshgrid(
        (
            torch.arange(
                -ball_radius, ball_radius + 1e-3, step=z_spacing, device=device
            ),
            torch.arange(
                -ball_radius,
                ball_radius + 1e-3,
                step=xy_spacing,
                device=device,
            ),
            torch.arange(
                -ball_radius,
                ball_radius + 1e-3,
                step=xy_spacing,
                device=device,
            ),
        )
    )
    if type(sample_size) is not torch.Size:
        sample_size = torch.Size(sample_size)

    ball = ((x.pow(2) + y.pow(2) + z.pow(2)).sqrt() < ball_radius).to(dtype)
    sample = (
        torch.rand(sample_size, device=device, requires_grad=False) < sparsity
    ).to(dtype)
    sample = sample * torch.randn(sample_size, device=device, requires_grad=False).abs()
    sample = fftconvn(sample, ball, ndims=3, shape="same")
    if separate:
        return sample

    sample = (
        (sample + background * (0.5 + torch.rand([])))
        * brightness
        * (0.5 + torch.rand([]))
    )
    return sample


def sparse_shells(
    sample_size,
    shell_radius=5.0,
    xy_spacing=0.1,
    z_spacing=3.0,
    sparsity=1e-5,
    brightness=500,
    background=1e-2,
    separate=False,
    device="cpu",
    dtype=torch.float32,
):
    z, y, x = torch.meshgrid(
        (
            torch.arange(
                -shell_radius,
                shell_radius + 1e-3,
                step=z_spacing,
                device=device,
            ),
            torch.arange(
                -shell_radius,
                shell_radius + 1e-3,
                step=xy_spacing,
                device=device,
            ),
            torch.arange(
                -shell_radius,
                shell_radius + 1e-3,
                step=xy_spacing,
                device=device,
            ),
        )
    )
    if type(sample_size) is not torch.Size:
        sample_size = torch.Size(sample_size)

    distances = (x.pow(2) + y.pow(2) + z.pow(2)).sqrt()
    shell = (
        (
            (distances < shell_radius) + (distances > (shell_radius - 2 * xy_spacing))
            == 2
        )
    ).to(dtype)
    sample = (
        torch.rand(sample_size, device=device, requires_grad=False) < sparsity
    ).to(dtype)
    sample = sample * torch.randn(sample_size, device=device, requires_grad=False).abs()
    sample = fftconvn(sample, shell, ndims=3, shape="same")
    if separate:
        return sample

    sample = (
        (sample + background * (0.5 + torch.rand([])))
        * brightness
        * (0.5 + torch.rand([]))
    )
    return sample


def sparse_filled_shells(
    sample_size,
    shell_radius=5.0,
    shell_filling=0.1,
    xy_spacing=0.1,
    z_spacing=3.0,
    sparsity=1e-5,
    brightness=500,
    background=1e-2,
    separate=False,
    device="cpu",
    dtype=torch.float32,
):
    z, y, x = torch.meshgrid(
        (
            torch.arange(
                -shell_radius,
                shell_radius + 1e-3,
                step=z_spacing,
                device=device,
            ),
            torch.arange(
                -shell_radius,
                shell_radius + 1e-3,
                step=xy_spacing,
                device=device,
            ),
            torch.arange(
                -shell_radius,
                shell_radius + 1e-3,
                step=xy_spacing,
                device=device,
            ),
        )
    )
    if type(sample_size) is not torch.Size:
        sample_size = torch.Size(sample_size)

    distances = (x.pow(2) + y.pow(2) + z.pow(2)).sqrt()
    shell = (
        (
            (distances < shell_radius) + (distances > (shell_radius - 2 * xy_spacing))
            == 2
        )
    ).to(dtype)
    ball = ((x.pow(2) + y.pow(2) + z.pow(2)).sqrt() < shell_radius).to(dtype)
    sample = (
        torch.rand(sample_size, device=device, requires_grad=False) < sparsity
    ).to(dtype)
    filling = fftconvn(sample * shell_filling, ball, ndims=3, shape="same")
    sample = sample * torch.randn(sample_size, device=device, requires_grad=False).abs()
    sample = fftconvn(sample, shell, ndims=3, shape="same")
    sample += filling
    if separate:
        return sample
    sample = (
        (sample + background * (0.5 + torch.rand([])))
        * brightness
        * (0.5 + torch.rand([]))
    )
    return sample


def sparse_rotated_balls(
    volume_size,
    sample_size,
    ball_radius=5.0,
    pitch=0,
    yaw=0,
    roll=0,
    xy_spacing=0.1,
    z_spacing=3.0,
    sparsity=1e-5,
    brightness=500,
    background=1e-2,
    separate=False,
    device="cpu",
    dtype=torch.float32,
):
    z, y, x = torch.meshgrid(
        (
            torch.arange(
                -ball_radius, ball_radius + 1e-3, step=z_spacing, device=device
            ),
            torch.arange(
                -ball_radius,
                ball_radius + 1e-3,
                step=xy_spacing,
                device=device,
            ),
            torch.arange(
                -ball_radius,
                ball_radius + 1e-3,
                step=xy_spacing,
                device=device,
            ),
        )
    )
    if type(sample_size) is not torch.Size:
        sample_size = torch.Size(sample_size)

    ball = ((x.pow(2) + y.pow(2) + z.pow(2)).sqrt() < ball_radius).to(dtype)
    sample = (
        torch.rand(sample_size, device=device, requires_grad=False) < sparsity
    ).to(dtype)
    sample = F.pad(
        sample,
        (
            int((volume_size[2] - sample_size[2]) / 2),
            int((volume_size[2] - sample_size[2] + 1) / 2),
            int((volume_size[1] - sample_size[1]) / 2),
            int((volume_size[1] - sample_size[1] + 1) / 2),
            int((volume_size[0] - sample_size[0]) / 2),
            int((volume_size[0] - sample_size[0] + 1) / 2),
        ),
    )
    sample_indices = torch.nonzero(sample)
    rotated_indices = rotate_indices(sample_indices.t(), sample_size, pitch, yaw, roll)
    sample.zero_()
    sample[tuple(rotated_indices)] = 1.0
    sample = sample * torch.randn(volume_size, device=device, requires_grad=False).abs()
    sample = fftconvn(sample, ball, ndims=3, shape="same")

    if separate:
        return sample
    sample = (
        (sample + background * (0.5 + torch.rand([])))
        * brightness
        * (0.5 + torch.rand([]))
    )
    return sample


def sparse_rotated_shells(
    volume_size,
    sample_size,
    shell_radius=5.0,
    pitch=0,
    yaw=0,
    roll=0,
    xy_spacing=0.1,
    z_spacing=3.0,
    sparsity=1e-5,
    brightness=500,
    background=1e-2,
    separate=False,
    device="cpu",
    dtype=torch.float32,
):
    z, y, x = torch.meshgrid(
        (
            torch.arange(
                -shell_radius,
                shell_radius + 1e-3,
                step=z_spacing,
                device=device,
            ),
            torch.arange(
                -shell_radius,
                shell_radius + 1e-3,
                step=xy_spacing,
                device=device,
            ),
            torch.arange(
                -shell_radius,
                shell_radius + 1e-3,
                step=xy_spacing,
                device=device,
            ),
        )
    )
    if type(sample_size) is not torch.Size:
        sample_size = torch.Size(sample_size)

    distances = (x.pow(2) + y.pow(2) + z.pow(2)).sqrt()
    shell = (
        (
            (distances < shell_radius) + (distances > (shell_radius - 2 * xy_spacing))
            == 2
        )
    ).to(dtype)
    sample = (
        torch.rand(sample_size, device=device, requires_grad=False) < sparsity
    ).to(dtype)
    sample = F.pad(
        sample,
        (
            int((volume_size[2] - sample_size[2]) / 2),
            int((volume_size[2] - sample_size[2] + 1) / 2),
            int((volume_size[1] - sample_size[1]) / 2),
            int((volume_size[1] - sample_size[1] + 1) / 2),
            int((volume_size[0] - sample_size[0]) / 2),
            int((volume_size[0] - sample_size[0] + 1) / 2),
        ),
    )
    sample_indices = torch.nonzero(sample)
    rotated_indices = rotate_indices(sample_indices.t(), volume_size, pitch, yaw, roll)
    sample.zero_()
    sample[tuple(rotated_indices)] = 1.0
    sample = sample * torch.randn(volume_size, device=device, requires_grad=False).abs()
    sample = fftconvn(sample, shell, ndims=3, shape="same")

    if separate:
        return sample
    sample = (
        (sample + background * (0.5 + torch.rand([])))
        * brightness
        * (0.5 + torch.rand([]))
    )
    return sample


def random_binary_shells(
    sample_size,
    shell_radius=5,
    xy_spacing=0.1,
    z_spacing=3.0,
    num_shells=20,
    device="cpu",
    dtype=torch.float32,
):
    z, y, x = torch.meshgrid(
        (
            torch.arange(
                -shell_radius,
                shell_radius + 1e-3,
                step=z_spacing,
                device=device,
            ),
            torch.arange(
                -shell_radius,
                shell_radius + 1e-3,
                step=xy_spacing,
                device=device,
            ),
            torch.arange(
                -shell_radius,
                shell_radius + 1e-3,
                step=xy_spacing,
                device=device,
            ),
        )
    )
    if type(sample_size) is not torch.Size:
        sample_size = torch.Size(sample_size)
    distances = (x.pow(2) + y.pow(2) + z.pow(2)).sqrt()
    shell = (
        (
            (distances < shell_radius) + (distances > (shell_radius - 2 * xy_spacing))
            == 2
        )
    ).to(dtype)
    raveled_points = np.random.choice(torch.prod(torch.tensor(sample_size)), num_shells)
    unraveled_points = np.array(np.unravel_index(raveled_points, sample_size))
    samples = torch.zeros((num_shells,) + sample_size, dtype=dtype, device=device)
    for i in range(num_shells):
        samples[i][tuple(unraveled_points[:, i])] = 1
        samples[i] = fftconvn(samples[i], shell, ndims=3, shape="same")
    return samples


def grid_binary_balls(
    sample_size,
    ball_radius=5,
    xy_spacing=0.1,
    z_spacing=3.0,
    num_planes=3,
    num_rows=3,
    num_columns=3,
    plane_space_change=0,
    row_space_change=0,
    device="cpu",
    dtype=torch.float32,
):
    z, y, x = torch.meshgrid(
        (
            torch.arange(
                -ball_radius, ball_radius + 1e-3, step=z_spacing, device=device
            ),
            torch.arange(
                -ball_radius,
                ball_radius + 1e-3,
                step=xy_spacing,
                device=device,
            ),
            torch.arange(
                -ball_radius,
                ball_radius + 1e-3,
                step=xy_spacing,
                device=device,
            ),
        )
    )
    if type(sample_size) is not torch.Size:
        sample_size = torch.Size(sample_size)
    distances = (x.pow(2) + y.pow(2) + z.pow(2)).sqrt()
    ball = ((x.pow(2) + y.pow(2) + z.pow(2)).sqrt() < ball_radius).to(dtype)
    num_balls = num_planes * num_rows * num_columns

    plane_pad = 0
    row_pad = 0
    ball_counter = 0
    points = np.zeros((num_balls, 3), dtype=np.uint16)
    point = np.array([int(s // (2 ** (1 / 3)) + 1) for s in ball.size()])
    for p in range(num_planes):
        point[1] = int(ball.size(1) // (2 ** (1 / 3)) + 1) + row_pad
        for r in range(num_rows):
            point[2] = int(ball.size(2) // (2 ** (1 / 3)) + 1)
            for c in range(num_columns):
                points[ball_counter] = point
                point[2] += int(2 * ball_radius / xy_spacing)
                ball_counter += 1
            point[1] += int(2 * ball_radius / xy_spacing) + row_pad
            row_pad += row_space_change
        point[0] += int(2 * ball_radius / z_spacing) + plane_pad
        plane_pad += plane_space_change

    if center_mean:
        sample_center = np.array(sample_size) // 2
        points_center = points.mean(axis=0)
        difference = np.uint16(sample_center - points_center)
        points += difference

    samples = torch.zeros((num_balls,) + sample_size, dtype=dtype, device=device)
    for i in range(num_balls):
        samples[i][tuple(points[i])] = 1
        current_ball = fftconvn(samples[i], ball, ndims=3, shape="same")
        samples[i] = current_ball

    samples[samples < 0.95] = 0.0
    return samples


def grid_binary_shells(
    sample_size,
    shell_radius=5,
    xy_spacing=0.1,
    z_spacing=3.0,
    num_planes=3,
    num_rows=3,
    num_columns=3,
    plane_space_change=0,
    row_space_change=0,
    center_mean=False,
    device="cpu",
    dtype=torch.float32,
):
    z, y, x = torch.meshgrid(
        (
            torch.arange(
                -shell_radius,
                shell_radius + 1e-3,
                step=z_spacing,
                device=device,
            ),
            torch.arange(
                -shell_radius,
                shell_radius + 1e-3,
                step=xy_spacing,
                device=device,
            ),
            torch.arange(
                -shell_radius,
                shell_radius + 1e-3,
                step=xy_spacing,
                device=device,
            ),
        )
    )
    if type(sample_size) is not torch.Size:
        sample_size = torch.Size(sample_size)

    distances = (x.pow(2) + y.pow(2) + z.pow(2)).sqrt()
    shell = (
        (
            (distances < shell_radius) + (distances > (shell_radius - 2 * xy_spacing))
            == 2
        )
    ).to(dtype)
    num_shells = num_planes * num_rows * num_columns

    plane_pad = 0
    row_pad = 0
    shell_counter = 0
    points = np.zeros((num_shells, 3), dtype=np.uint16)
    point = np.array([int(s // (2 ** (1 / 3)) + 1) for s in shell.size()])
    for p in range(num_planes):
        point[1] = int(shell.size(1) // (2 ** (1 / 3)) + 1) + row_pad
        for r in range(num_rows):
            point[2] = int(shell.size(2) // (2 ** (1 / 3)) + 1)
            for c in range(num_columns):
                points[shell_counter] = point
                point[2] += int(2 * shell_radius / xy_spacing)
                shell_counter += 1
            point[1] += int(2 * shell_radius / xy_spacing) + row_pad
            row_pad += row_space_change
        point[0] += int(2 * shell_radius / z_spacing) + plane_pad
        plane_pad += plane_space_change

    if center_mean:
        sample_center = np.array(sample_size) // 2
        points_center = points.mean(axis=0)
        difference = np.uint16(sample_center - points_center)
        points += difference

    samples = torch.zeros((num_shells,) + sample_size, dtype=dtype, device=device)
    for i in range(num_shells):
        samples[i][tuple(points[i])] = 1
        current_shell = fftconvn(samples[i], shell, ndims=3, shape="same")
        samples[i] = current_shell

    samples[samples < 0.95] = 0.0
    return samples


def shift_sample(sample, shift, value=0):
    assert shift.shape == (3,), "Shift must be flat list with 3 elements"
    pad = []

    if shift[2] > 0:
        pad += [shift[2], 0]
    elif shift[2] < 0:
        pad += [0, -shift[2]]
    else:
        pad += [0, 0]

    if shift[1] > 0:
        pad += [0, shift[1]]
    elif shift[1] < 0:
        pad += [-shift[1], 0]
    else:
        pad += [0, 0]

    if shift[0] > 0:
        pad += [0, shift[0]]
    elif shift[0] < 0:
        pad += [-shift[0], 0]
    else:
        pad += [0, 0]

    pad = tuple(pad)
    padded = F.pad(sample, pad, value=value)

    slices = []
    if shift[0] > 0:
        slices.append(slice(shift[0], sample.shape[0] + shift[0]))
    elif shift[0] <= 0:
        slices.append(slice(0, sample.shape[0]))

    if shift[1] > 0:
        slices.append(slice(shift[1], sample.shape[1] + shift[1]))
    elif shift[1] <= 0:
        slices.append(slice(0, sample.shape[1]))

    if shift[2] >= 0:
        slices.append(slice(0, sample.shape[2]))
    elif shift[2] < 0:
        slices.append(slice(-shift[2], sample.shape[2] - shift[2]))

    shifted_sample = padded[slices[0], slices[1], slices[2]]

    return shifted_sample


def apply_series_intensities(samples, intensities, random=True):
    samples = torch.tensordot(intensities, samples, dims=1)
    return samples


def apply_series_brightness(series, background=1e-2, brightness=500, random=True):
    num_frames = series.size(0)
    if random:
        backgrounds = background * (
            0.5 + torch.rand([num_frames], device=series.device)
        )
        backgrounds = backgrounds[:, None, None, None]
    else:
        backgrounds = background
    if random:
        brightness_values = brightness * (
            0.5 + torch.rand([num_frames], device=series.device)
        )
        brightness_values = brightness_values[:, None, None, None]
    else:
        brightness_values = brightness
    series = (series + backgrounds) * brightness_values
    return series


def apply_traces_brightness(traces, background=1e-2, brightness=500):
    num_frames = traces.size(0)
    backgrounds = background * (0.5 + torch.rand([num_frames], device=traces.device))
    backgrounds = backgrounds[:, None]
    brightness_values = brightness * (
        0.5 + torch.rand([num_frames], device=traces.device)
    )
    brightness_values = brightness_values[:, None]
    traces = (traces + backgrounds) * brightness_values
    return traces


def apply_volume_brightness(volume, background=1e-2, brightness=500, random=True):
    # background can also be a 2d image
    if isinstance(background, float):
        if random:
            background = background * (0.5 + torch.rand([], device=volume.device))
    if random:
        brightness = brightness * (0.5 + torch.rand([], device=volume.device))
    volume = (volume + background) * brightness
    return volume


def decaying_background_mask(shape, sigma=7, device=torch.device("cpu")):
    background = np.ones(shape)
    background = gaussian_filter(background, sigma, mode="constant")
    background = torch.from_numpy(background).to(torch.float32).to(device)
    return background


def shells_scf_series(
    shell_samples,
    brightness=500,
    background=1e-2,
    num_shells=20,
    timebins=10,
    fps=1,
    firing_rate=0.3,
    low_params=None,
    high_params=None,
    scale_trace=False,
    dtype=torch.float32,
):
    if low_params == None:
        low_params = {
            "gamma": 2.75,
            "alpha": -1.25,
            "beta": 0.0,
            "sigma": 0.01,
        }
    if high_params == None:
        high_params = {
            "gamma": 4.6,
            "alpha": -0.43,
            "beta": 0.0,
            "sigma": 0.02,
        }
    trace_dict = linear_scf(
        num_shells, timebins, fps, firing_rate, low_params, high_params
    )
    shell_intensities = torch.from_numpy(trace_dict["traces"].T)
    shell_intensities = shell_intensities.to(dtype)
    shell_intensities = shell_intensities.to(shell_samples.device)
    min_intensity = shell_intensities.min(1)[0].unsqueeze(1)
    shell_intensities -= min_intensity
    shell_intensities += min_intensity.abs() * 0.1
    shell_series = apply_series_intensities(shell_samples, shell_intensities)
    shell_series = apply_series_brightness(
        shell_series, background=background, brightness=brightness
    )
    if scale_trace:
        shell_intensities = apply_traces_brightness(
            shell_intensities, background=background, brightness=brightness
        )
    return shell_series, shell_intensities


def linear_scf(
    num_cells, timebins, fps, firing_rate, low_params=None, high_params=None
):
    """
    Implementation of a simple linear model of fluorescence dynamics. Returns
    the generated data as a dictionary containing the traces, spikes, params,
    fps and spike_fps.
    """

    firing_probability = firing_rate / fps

    traces = []
    spikes = []
    true_params = []

    if low_params == None:
        low_params = {
            "gamma": 2.75,
            "alpha": -0.45,
            "beta": 0.05,
            "sigma": 0.01,
        }
    if high_params == None:
        high_params = {
            "gamma": 4.6,
            "alpha": 1.23,
            "beta": 0.15,
            "sigma": 0.02,
        }

    for _ in range(num_cells):
        init_parameters = {}

        for p in low_params:
            if low_params[p] < high_params[p]:
                # generate random params between specified range
                init_parameters.update(
                    {
                        p: np.random.randint(
                            1000 * low_params[p], 1000 * high_params[p]
                        )
                        / 1000
                    }
                )
            else:
                init_parameters.update({p: low_params[p]})

        # generate spikes according to the respective firing probability
        sim_spikes = np.random.binomial(1, firing_probability, (timebins,))
        sim_spikes = np.array(sim_spikes).astype(np.float32)
        sim_fluor_temp = np.zeros(sim_spikes.shape, dtype=np.float32)
        g = init_parameters["gamma"]
        C = 0.0  # C_0
        for t in range(1, timebins):
            C = linear_scf_step(sim_spikes[t], C, g)
            sim_fluor_temp[t] = (
                softp(init_parameters["alpha"]) * C + init_parameters["beta"]
            )
        noise = np.random.normal(size=(sim_spikes.shape)) * init_parameters["sigma"]
        sim_fluor = sim_fluor_temp + noise

        traces.append(copy.deepcopy(sim_fluor))
        spikes.append(copy.deepcopy(sim_spikes))
        true_params.append(copy.deepcopy(init_parameters))

    return {
        "traces": np.array(traces).clip(min=0),
        "spikes": np.array(spikes),
        "params": true_params,
        "fps": [fps for _ in range(num_cells)],
        "spike_fps": 60,
    }


def linear_scf_step(s, C_prev, g):
    """Single step of SCF model"""
    return s + sigmoid(g) * C_prev


def softp(x):
    """Softplus function"""
    return np.log(1 + np.exp(x))
