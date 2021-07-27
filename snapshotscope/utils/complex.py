import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

NP_DTYPE = np.complex64
T_DTYPE = torch.float32


def cart2pol(x, y):
    rho = torch.sqrt(x ** 2 + y ** 2)
    theta = torch.atan2(y, x)
    return (rho, theta)


def pol2cart(rho, theta):
    x = rho * torch.cos(theta)
    y = rho * torch.sin(theta)
    return x, y


def real_to_complex(input):
    output = F.pad(input.unsqueeze(-1), (0, 1))
    return output


def ctensor_from_phase_angle(input):
    real = input.cos().unsqueeze(-1)
    imag = input.sin().unsqueeze(-1)
    output = torch.cat([real, imag], input.ndimension())
    return output


def ctensor_from_amplitude_and_phase_angle(amplitude, phase):
    real = amplitude.unsqueeze(-1) * phase.cos().unsqueeze(-1)
    imag = amplitude.unsqueeze(-1) * phase.sin().unsqueeze(-1)
    output = torch.cat([real, imag], phase.ndimension())
    return output


def phase_angle_from_ctensor(input, phase_range=[-math.pi, math.pi]):
    phase = torch.atan2(input[..., 1], input[..., 0])
    phase = (phase - (-math.pi)) * (phase_range[1] - phase_range[0]) / (
        2 * math.pi
    ) + phase_range[0]
    return phase


def ctensor_from_numpy(input, dtype=T_DTYPE, device="cpu"):
    real = torch.tensor(input.real, dtype=dtype, device=device).unsqueeze(-1)
    imag = torch.tensor(input.imag, dtype=dtype, device=device).unsqueeze(-1)
    output = torch.cat([real, imag], np.ndim(input))
    return output


def tuple_from_numpy(input, dtype=T_DTYPE, device="cpu"):
    real = torch.tensor(input.real, dtype=dtype, device=device)
    imag = torch.tensor(input.imag, dtype=dtype, device=device)
    output = (real, imag)
    return output


def ctensor_from_tuple(tensor_tuple):
    ndim = tensor_tuple[0].ndimension()
    ctensor = torch.cat(
        (tensor_tuple[0].unsqueeze(-1), tensor_tuple[1].unsqueeze(-1)), ndim
    )
    return ctensor


def ctensor_to_tuple(ctensor):
    ndim = ctensor.ndimension()
    tensor_tuple = torch.split(ctensor, (1, 1), ndim - 1)
    tensor_tuple = (
        tensor_tuple[0].squeeze(ndim - 1),
        tensor_tuple[1].squeeze(ndim - 1),
    )
    return tensor_tuple


def cmul(x, y, xy=None):
    if xy is None:
        xy = torch.zeros_like(x)
    x2 = x.view(-1, 2)
    y2 = y.view(-1, 2)
    xy2 = xy.view(-1, 2)
    xy2[:, 0] = torch.mul(x2[:, 0], y2[:, 0]) - torch.mul(x2[:, 1], y2[:, 1])
    xy2[:, 1] = torch.mul(x2[:, 0], y2[:, 1]) + torch.mul(x2[:, 1], y2[:, 0])
    return xy


def real(x):
    return x.index_select(-1, torch.tensor(0, device=x.device)).squeeze(-1)


def imag(x):
    return x.index_select(-1, torch.tensor(1, device=x.device)).squeeze(-1)


def cmul2(x, y, xy=None):
    xr, xi = real(x), imag(x)
    yr, yi = real(y), imag(y)
    xyr = torch.mul(xr, yr) - torch.mul(xi, yi)
    xyi = torch.mul(xr, yi) + torch.mul(xi, yr)
    xy = torch.stack([xyr, xyi], -1, out=xy)
    return xy


def cdiv(x, y, xy=None):
    if xy is None:
        xy = torch.zeros_like(x)
    x2 = x.view(-1, 2)
    y2 = y.view(-1, 2)
    xy2 = xy.view(-1, 2)
    xy2[:, 0] = torch.mul(x2[:, 0], y2[:, 0]) - torch.mul(x2[:, 1], -y2[:, 1])
    xy2[:, 1] = torch.mul(x2[:, 0], -y2[:, 1]) + torch.mul(x2[:, 1], y2[:, 0])
    xy2 /= y2.pow(2).sum(-1).unsqueeze(-1)
    return xy


def cmulconjy(x, y, xy=None):
    """x*conj(y)"""
    if xy is None:
        xy = torch.zeros_like(x)
    x2 = x.view(-1, 2)
    y2 = y.view(-1, 2)
    xy2 = xy.view(-1, 2)
    xy2[:, 0] = torch.mul(x2[:, 0], y2[:, 0]) - torch.mul(x2[:, 1], -y2[:, 1])
    xy2[:, 1] = torch.mul(x2[:, 0], -y2[:, 1]) + torch.mul(x2[:, 1], y2[:, 0])
    return xy


def cmulconjx(x, y, xy=None):
    """x*conj(y)"""
    if xy is None:
        xy = torch.zeros_like(x)
    x2 = x.view(-1, 2)
    y2 = y.view(-1, 2)
    xy2 = xy.view(-1, 2)
    xy2[:, 0] = torch.mul(x2[:, 0], y2[:, 0]) - torch.mul(-x2[:, 1], y2[:, 1])
    xy2[:, 1] = torch.mul(x2[:, 0], y2[:, 1]) + torch.mul(-x2[:, 1], y2[:, 0])
    return xy


def cabs2(x):
    xabs2 = x.pow(2).sum(-1)
    return xabs2


def cabs(x):
    xabs = x.pow(2).sum(-1).sqrt()
    return xabs


def cmul_tuple(x, y):
    z_r = torch.mul(x[0], y[0]) - torch.mul(x[1], y[1])
    z_i = torch.mul(x[0], y[1]) + torch.mul(x[1], y[0])
    return (z_r, z_i)


def cabs_tuple(x):
    return (x[0].pow(2) + x[1].pow(2)).sqrt()


def fftshift(input, dim=1):
    split_size_right = int(np.floor(input.size(dim) / 2))
    split_sizes = (input.size(dim) - split_size_right, split_size_right)
    pos, neg = torch.split(input, split_sizes, dim=dim)
    input = torch.cat([neg, pos], dim=dim)
    return input


def ifftshift(input, dim=1):
    split_size_left = int(np.floor(input.size(dim) / 2))
    split_sizes = (split_size_left, input.size(dim) - split_size_left)
    pos, neg = torch.split(input, split_sizes, dim=dim)
    input = torch.cat([neg, pos], dim=dim)
    return input


def fftshift2d(input):
    for dim in [0, 1]:
        input = fftshift(input, dim=dim)
    return input


def ifftshift2d(input):
    for dim in [0, 1]:
        input = ifftshift(input, dim=dim)
    return input


def fftshiftn(input):
    for dim in range(len(input.shape)):
        input = fftshift(input, dim=dim)
    return input


def ifftshiftn(input):
    for dim in range(len(input.shape)):
        input = ifftshift(input, dim=dim)
    return input


def cfftconv2d(a, b, fa=None, fb=None, shape="full", fftsize=None):

    asize = a.size()
    bsize = b.size()

    if fftsize == None:
        fftsize = [asize[-3] + bsize[-3] - 1, asize[-2] + bsize[-2] - 1]

    # use cached fa, fb if available
    if fa is None:
        fa = torch.fft(
            F.pad(
                a,
                (0, 0, 0, fftsize[-1] - asize[-2], 0, fftsize[-2] - asize[-3]),
            ),
            2,
        )
    if fb is None:
        fb = torch.fft(
            F.pad(
                b,
                (0, 0, 0, fftsize[-1] - bsize[-2], 0, fftsize[-2] - bsize[-3]),
            ),
            2,
        )

    # complex fft convolution
    ab = torch.ifft(cmul(fa, fb), 2)

    # crop based on shape
    if shape in "same":
        cropsize = [fftsize[-2] - asize[-3], fftsize[-1] - asize[-2]]
        cropsizeL = [int(c / 2) for c in cropsize]
        cropsizeR = [int((c + 1) / 2) for c in cropsize]
        ab = F.pad(
            ab,
            (0, 0, 0, -cropsize[-1], 0, -cropsize[-2]),
        )
    elif shape in "valid":
        cropsize = [
            fftsize[-2] - asize[-3] + bsize[-3] - 1,
            fftsize[-1] - asize[-2] + bsize[-2] - 1,
        ]
        cropsizeL = [int(c / 2) for c in cropsize]
        cropsizeR = [int((c) / 2) for c in cropsize]
        ab = F.pad(
            ab,
            (0, 0, -cropsizeL[-1], -cropsizeR[-1], -cropsizeL[-2], -cropsizeR[-2]),
        )

    return ab


def fftconv2d(a, b, fa=None, fb=None, shape="full", fftsize=None):

    asize = a.size()
    bsize = b.size()

    if fftsize == None:
        fftsize = [asize[-2] + bsize[-2] - 1, asize[-1] + bsize[-1] - 1]

    # zero pad real signal and add a channel for the imaginary part
    # use cached fa, fb if available
    if fa is None:
        fa = torch.fft(
            F.pad(
                a.unsqueeze(-1),
                (0, 1, 0, fftsize[-1] - asize[-1], 0, fftsize[-2] - asize[-2]),
            ),
            2,
        )
    if fb is None:
        fb = torch.fft(
            F.pad(
                b.unsqueeze(-1),
                (0, 1, 0, fftsize[-1] - bsize[-1], 0, fftsize[-2] - bsize[-2]),
            ),
            2,
        )

    # fft convolution, keep real part, remove imaginary part
    ab = (
        torch.ifft(cmul(fa, fb), 2)
        .index_select(-1, torch.tensor(0, device=fa.device))
        .squeeze(-1)
    )

    # crop based on shape
    if shape in "same":
        cropsize = [fftsize[-2] - asize[-2], fftsize[-1] - asize[-1]]
        cropsizeL = [int(c / 2) for c in cropsize]
        cropsizeR = [int((c + 1) / 2) for c in cropsize]
        ab = F.pad(
            ab,
            (-cropsizeL[-1], -cropsizeR[-1], -cropsizeL[-2], -cropsizeR[-2]),
        )
    elif shape in "valid":
        cropsize = [
            fftsize[-2] - asize[-2] + bsize[-2] - 1,
            fftsize[-1] - asize[-1] + bsize[-1] - 1,
        ]
        cropsizeL = [int(c / 2) for c in cropsize]
        cropsizeR = [int((c) / 2) for c in cropsize]
        ab = F.pad(
            ab,
            (-cropsizeL[-1], -cropsizeR[-1], -cropsizeL[-2], -cropsizeR[-2]),
        )

    return ab


def fftconvn(a, b, fa=None, fb=None, ndims=3, shape="full", fftsize=None):

    asize = list(a.size())[::-1][:ndims]
    bsize = list(b.size())[::-1][:ndims]

    if fftsize == None:
        fftsize = [asz + bsz - 1 for asz, bsz in zip(asize, bsize)]

    # zero pad real signal and add a channel for the imaginary part
    # use cached fa, fb if available
    if fa is None:
        padsize = (0, 1)
        for fsz, asz in zip(fftsize, asize):
            padsize += (0, fsz - asz)
        fa = torch.fft(F.pad(a.unsqueeze(-1), padsize), ndims)
    if fb is None:
        padsize = (0, 1)
        for fsz, bsz in zip(fftsize, bsize):
            padsize += (0, fsz - bsz)
        fb = torch.fft(F.pad(b.unsqueeze(-1), padsize), ndims)

    # fft convolution, keep real part, remove imaginary part
    ab = (
        torch.ifft(cmul(fa, fb), ndims)
        .index_select(-1, torch.tensor(0, device=fa.device))
        .squeeze(-1)
    )

    # crop based on shape
    if shape in "same":
        cropsize = [fsz - asz for fsz, asz in zip(fftsize, asize)]
        padsize = ()
        for c in cropsize:
            padsize += (-int(c / 2), -int((c + 1) / 2))
        ab = F.pad(ab, padsize)
    elif shape in "valid":
        cropsize = [fsz - asz + bsz - 1 for fsz, asz, bsz in zip(fftsize, asize, bsize)]
        padsize = ()
        for c in cropsize:
            padsize += (-int(c / 2), -int((c + 1) / 2))
        ab = F.pad(ab, padsize)

    return ab


def fftcrosscorr2d(a, b, fa=None, fb=None, shape="full", fftsize=None):

    asize = a.size()
    bsize = b.size()

    if fftsize == None:
        fftsize = [asize[-2] + bsize[-2] - 1, asize[-1] + bsize[-1] - 1]

    # zero pad real signal and add a channel for the imaginary part
    # use cached fa, fb if available
    if fa is None:
        fa = torch.fft(
            F.pad(
                a.unsqueeze(-1),
                (0, 1, 0, fftsize[-1] - asize[-1], 0, fftsize[-2] - asize[-2]),
            ),
            2,
        )
    if fb is None:
        fb = torch.fft(
            F.pad(
                b.unsqueeze(-1),
                (0, 1, 0, fftsize[-1] - bsize[-1], 0, fftsize[-2] - bsize[-2]),
            ),
            2,
        )

    # fft cross correlation, keep real part, remove imaginary part
    ab = (
        torch.ifft(cmulconjy(fa, fb), 2)
        .index_select(-1, torch.tensor(0, device=fa.device))
        .squeeze(-1)
    )
    for dim in list(range(ab.ndimension()))[-2:]:
        if ab.size(dim) > 1:
            ab = fftshift(ab, dim)

    # crop based on shape
    if shape in "same":
        cropsize = [fftsize[-2] - asize[-2], fftsize[1] - asize[-1]]
        cropsizeL = [int(c / 2) for c in cropsize]
        cropsizeR = [int((c + 1) / 2) for c in cropsize]
        ab = F.pad(
            ab,
            (-cropsizeL[-1], -cropsizeR[-1], -cropsizeL[-2], -cropsizeR[-2]),
        )
    elif shape in "valid":
        cropsize = [
            fftsize[-2] - asize[-2] + bsize[-2] - 1,
            fftsize[-1] - asize[-1] + bsize[-1] - 1,
        ]
        cropsizeL = [int(c / 2) for c in cropsize]
        cropsizeR = [int((c) / 2) for c in cropsize]
        ab = F.pad(
            ab,
            (-cropsizeL[-1], -cropsizeR[-1], -cropsizeL[-2], -cropsizeR[-2]),
        )

    return ab


def center_crop2d(im, out_shape):
    shape = (im.shape[-2], im.shape[-1])
    crop_shape = [shape[-2] - out_shape[-2], shape[1] - out_shape[-1]]
    crop_shape_left = [int(c / 2) for c in crop_shape]
    crop_shape_right = [int((c + 1) / 2) for c in crop_shape]
    return F.pad(
        im,
        (
            -crop_shape_left[-1],
            -crop_shape_right[-1],
            -crop_shape_left[-2],
            -crop_shape_right[-2],
        ),
    )


def center_crop2d_complex(im, out_shape):
    shape = (im.shape[-3], im.shape[-2])
    crop_shape = [shape[-2] - out_shape[-2], shape[1] - out_shape[-1]]
    crop_shape_left = [int(c / 2) for c in crop_shape]
    crop_shape_right = [int((c + 1) / 2) for c in crop_shape]
    return F.pad(
        im,
        (
            0,
            0,
            -crop_shape_left[-1],
            -crop_shape_right[-1],
            -crop_shape_left[-2],
            -crop_shape_right[-2],
        ),
    )
