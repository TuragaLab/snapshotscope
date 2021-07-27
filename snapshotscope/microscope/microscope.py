import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F

from snapshotscope.optical_elements import microlenses as ul
from snapshotscope.optical_elements import phase_masks as pm
from snapshotscope.optical_elements import propagation as prop
from snapshotscope.utils import *
from snapshotscope.utils import _list

NP_DTYPE = np.complex64
T_DTYPE = torch.float32


class PupilFunction4FMicroscope(nn.Module):
    """
    Simulated 4f microscope with a phase mask in the pupil/fourier plane. Can
    be made chromatic by passing multiple wavelengths and desired combination
    ratio. Can be used either standalone or in parallel with other
    microscopes via ``snapshotscope.utils.parallel_apply``. Returns a simulated
    image from the camera (without shot noise) of an input 3D volume.

    Args:
        wavelength: A float or list of floats representing the wavelength(s)
        of light used to illuminate the microscope in microns.

        ratios: A list of floats representing the relative weighting of each
        wavelength of light.

        NA: A float in range [0, 1] represnting the NA (numerical aperture)
        of the pupil of the microscope. n_immersion: A float representing the
        refractive index of the immersion medium of the microscope.

        phase_mask: A phase mask module from snapshotscope.phase_masks. Defines
        the phase mask for the microscope.

        pixel_size: A float representing the xy pixel size (resolution) in
        microns of the microscope imaging plane.

        num_pixels: An integer representing the number of pixels on the
        microscope camera.

        pad: An integer representing the number of pixels to pad the simulation
        with in the x and y dimensions. This will be cropped away by pad // 2
        on both sides in each dimension after the PSF is calculated for
        imaging. Defaults to a value of 0, representing no padding.

        taper_width: A number representing the width of the tapering to apply
        to the PSF after cropping during imaging. The purpose of this is to
        fade the cropped PSF to 0 near the edges to avoid hard edge artifacts.
        A value of 0 (the default) would mean a simple crop without any
        tapering. A value of 10 applies a small amount of tapering at the
        default number of pixels and pixel size. Tapering is only performed if
        pad > 0, otherwise this argument is ignored.

        downsample: An integer representing the downsampling factor for PSF
        calculation/imaging. The PSF will be downscaled by this factor before
        imaging, so inputs for imaging should be matched to the desired
        downsampling factor.

        device: A string or torch.device representing the device to use for
        the parameters and buffers of this microscope module.

    Shape:
        - Input: :math:`(D, H, W)` (a 3D volume in ZYX order)
        - Output: :math:`(H, W)` (same as a single plane from the input)
    """

    def __init__(
        self,
        wavelength=[0.532],
        ratios=[1.0],
        NA=0.8,
        n_immersion=1.33,
        pixel_size=0.325,
        num_pixels=2560,
        pad=0,
        taper_width=0,
        downsample=1,
        device="cpu",
    ):
        super(PupilFunction4FMicroscope, self).__init__()

        self.wavelength = _list(wavelength)
        self.ratios = _list(ratios)
        self.NA = NA
        self.n_immersion = n_immersion
        self.pixel_size = pixel_size
        self.num_pixels = num_pixels + pad
        self.pad = pad
        self.taper_width = taper_width
        self.downsample = downsample
        self.device = device

        # sampling period, microns
        self.dx = self.pixel_size
        # spatial sampling frequency, inverse microns
        self.fS = 1 / self.dx
        # spacing between discrete frequency coordinates, inverse microns
        self.dk = 1 / (self.pixel_size * self.num_pixels)
        # radius of the pupil, inverse microns
        self.fNA = [self.NA / w for w in self.wavelength]

        # create the image plane coordinates
        self.x = (
            np.linspace(
                -(self.num_pixels - 1) / 2,
                (self.num_pixels - 1) / 2,
                num=self.num_pixels,
                endpoint=True,
            )
            * self.dx
        )
        self.xy_range = [self.x[0], self.x[-1]]
        self.psf_size = [self.num_pixels, self.num_pixels]
        self.fourier_psf_size = [
            2 * self.num_pixels - 1,
            2 * self.num_pixels - 1,
            2,
        ]
        if self.downsample > 1:
            self.downsampled_psf_size = [
                int(d / self.downsample) for d in self.psf_size
            ]
            self.downsampled_fourier_psf_size = [
                2 * self.downsampled_psf_size[0] - 1,
                2 * self.downsampled_psf_size[1] - 1,
                2,
            ]
        # create the Fourier plane
        self.k = (
            np.linspace(
                -(self.num_pixels - 1) / 2,
                (self.num_pixels - 1) / 2,
                num=self.num_pixels,
                endpoint=True,
            )
            * self.dk
        )
        self.k_range = [self.k[0], self.k[-1]]
        kx, ky = np.meshgrid(self.k, self.k)
        self.register_buffer("kx", torch.tensor(kx, dtype=T_DTYPE, device=self.device))
        self.register_buffer("ky", torch.tensor(ky, dtype=T_DTYPE, device=self.device))
        # create the pupil, which is defined by the numerical aperture
        pupil_mask = [
            1.0 * (np.sqrt(kx ** 2 + ky ** 2) <= self.fNA[i])
            for i in range(len(self.wavelength))
        ]
        pupil_power = [(m ** 2).sum() * self.dk ** 2 for m in pupil_mask]
        pupil = [
            pupil_mask[i] / np.sqrt(pupil_power[i]) for i in range(len(self.wavelength))
        ]
        pupil = np.stack(pupil)
        self.register_buffer("pupil", ctensor_from_numpy(pupil, device=self.device))
        # make the defocus kernel (radians)
        defocus_phase_angle = [
            2
            * np.pi
            * np.sqrt(
                ((self.n_immersion / self.wavelength[i]) ** 2 - (kx ** 2 + ky ** 2))
                * pupil_mask[i]
            )
            for i in range(len(self.wavelength))
        ]
        defocus_phase_angle = np.stack(defocus_phase_angle)
        self.register_buffer(
            "defocus_phase_angle",
            torch.tensor(defocus_phase_angle, dtype=T_DTYPE, device=self.device),
        )
        # make the plane
        if self.pad > 0:
            distance_input = np.pad(
                np.ones(
                    (
                        self.num_pixels - self.pad - 2,
                        self.num_pixels - self.pad - 2,
                    )
                ),
                1,
            )
            distance = scipy.ndimage.distance_transform_edt(distance_input)
            self.taper = 2 * (
                torch.sigmoid(torch.from_numpy(distance) / self.taper_width) - 0.5
            ).to(self.device)

    def compute_psf(self, phase_mask, defocus_z):
        # calculate psf
        # split necessary variables over wavelengths
        pupils = torch.split(self.pupil, 1)
        defocus_phase_angles = torch.split(self.defocus_phase_angle, 1)
        psf = 0
        for i in range(len(self.wavelength)):
            # apply pupil mask to phase mask, scaled by wavelength
            # scale phase by ratio of wavelength i / wavelength 0
            pupil_phase = cmul(
                pupils[i].squeeze(0),
                phase_mask * self.wavelength[0] / self.wavelength[i],
            )
            defocus = ctensor_from_phase_angle(
                defocus_phase_angles[i].squeeze(0).mul(defocus_z)
            )
            pupil_phase_defocus = cmul(pupil_phase, defocus)
            # TODO(dip): call this as a batch
            im_field = (self.dk ** 2) * fftshift2d(
                torch.fft(ifftshift2d(pupil_phase_defocus), 2)
            ).unsqueeze(0)
            # psf is weighted sum of fields
            # from pupils/defocusing of different wavelengths
            weighted_im_field = (self.ratios[i] * cabs2(im_field)).squeeze()
            psf += weighted_im_field.squeeze()
        return psf

    def image(self, phase_mask_angle, sample, defocus_zs):

        # TODO(dip): take complex phase mask as input, do everything in complex domain
        # TODO(dip): rename phase_mask_angle to pupil_function
        phase_mask = ctensor_from_phase_angle(phase_mask_angle)
        im = 0
        psf = []
        # TODO(dip): move the psf calculation loop into compute_psf
        for zidx, defocus_z in enumerate(defocus_zs):
            psf.append(self.compute_psf(phase_mask, defocus_z))
        psf = torch.stack(psf)
        # store psf
        self.psf = psf
        # crop and downsample psf for imaging
        crop = int(self.pad / 2)
        if crop > 0:
            psf = psf[:, crop:-crop, crop:-crop]
            psf = psf * self.taper.expand_as(psf)
            self.cropped_psf = psf
        if self.downsample > 1:
            psf = F.avg_pool2d(
                psf.unsqueeze(1),
                kernel_size=self.downsample,
                divisor_override=1,
            ).squeeze(1)
        im = fftconv2d(sample, psf, shape="same").sum(0)
        return im

    def forward(self, *args, **kwargs):
        return self.image(*args, **kwargs)


class PremadePSFMicroscope(nn.Module):
    """
    Dummy microscope made to hold a precomputed PSF and cache the Fourier
    transform of the given PSF for use in imaging or backprojection
    calculations as needed for iterative reconstruction algorithms.
    """

    def __init__(self, psf, pad=0, taper_width=0, downsample=1):
        # set up cropping parameters
        self.pad = pad
        self.taper_width = taper_width
        # cache psf
        self.psf = psf
        self.num_pixels = self.psf.shape[-1]
        self.downsample = downsample
        # crop and downsample psf for imaging
        crop = int(self.pad / 2)
        if crop > 0:
            # create taper plane
            distance_input = np.pad(
                np.ones(
                    (
                        self.num_pixels - self.pad - 2,
                        self.num_pixels - self.pad - 2,
                    )
                ),
                1,
            )
            distance = scipy.ndimage.distance_transform_edt(distance_input)
            self.taper = 2 * (
                torch.sigmoid(torch.from_numpy(distance) / self.taper_width) - 0.5
            ).to(self.psf.device)
            self.taper = self.taper.to(torch.float32)
            cropped_psf = self.psf[:, crop:-crop, crop:-crop]
            cropped_psf = cropped_psf * self.taper.expand_as(cropped_psf)
            self.psf = cropped_psf
        if self.downsample > 1:
            self.psf = F.avg_pool2d(
                self.psf.unsqueeze(1),
                kernel_size=self.downsample,
                divisor_override=1,
            ).squeeze(1)
        # calculate fourier transform of psf and cache result
        psfsize = self.psf.shape
        fftsize = [d * 2 - 1 for d in psfsize]
        pad_size = (
            0,
            1,
            0,
            fftsize[-1] - psfsize[-1],
            0,
            fftsize[-2] - psfsize[-2],
        )
        fourier_psf = torch.fft(F.pad(self.psf.unsqueeze(-1), pad_size), 2)
        self.fourier_psf = fourier_psf
        # calculate psf sum and cache result
        psfsum = fftcrosscorr2d(
            torch.ones_like(self.psf),
            self.psf,
            fb=self.fourier_psf,
            shape="same",
        )
        self.psfsum = psfsum

    def image(self, sample):
        return fftconv2d(sample, self.psf, fb=self.fourier_psf, shape="same").sum(0)

    def backproject(self, image):
        recon = fftcrosscorr2d(
            image.unsqueeze(0).expand_as(self.psf),
            self.psf,
            fb=self.fourier_psf,
            shape="same",
        )
        recon = recon / self.psfsum
        return recon

    def forward(self, *args, **kwargs):
        return self.image(*args, **kwargs)


def compute_power(psf, dk):
    return (psf ** 2).sum(dim=(1, 2)) * dk ** 2
