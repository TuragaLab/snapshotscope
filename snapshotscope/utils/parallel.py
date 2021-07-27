import threading

import torch
import torch.cuda.comm
import torch.nn as nn
from torch._utils import ExceptionWrapper
from torch.cuda._utils import _get_device_index

from .misc import _single


def scatter(inputs, devices):
    return nn.parallel.scatter(inputs, [_get_device_index(d) for d in devices])


def gather(outputs, device):
    return nn.parallel.gather(outputs, _get_device_index(device))


def replicate(module, devices):
    return nn.parallel.replicate(module, [_get_device_index(d) for d in devices])


def parallel_apply(modules, inputs, devices, requires_grad=True):
    """A version of parallel_apply that works with functions that call FFT."""
    lock = threading.Lock()
    results = {}

    def _worker(i, module, scattered_input, device, requires_grad):
        try:
            if device.type != "cpu":
                torch.backends.cuda.cufft_plan_cache[i].clear()
            if isinstance(module, nn.Module):
                module.to(device)
            scattered_input = [
                s.to(device) if isinstance(s, torch.Tensor) else s
                for s in scattered_input
            ]
            if requires_grad:
                result = module(*scattered_input)
            else:
                with torch.no_grad():
                    result = module(*scattered_input)
            with lock:
                results[i] = result
        except Exception:
            with lock:
                results[i] = ExceptionWrapper(where=f"in module {i} on device {device}")

    if len(modules) > 1:
        threads = [
            threading.Thread(
                target=_worker,
                args=(
                    i,
                    module,
                    _single(scattered_input),
                    device,
                    requires_grad,
                ),
            )
            for i, (module, scattered_input, device) in enumerate(
                zip(modules, inputs, devices)
            )
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    else:
        _worker(0, modules[0], _single(inputs[0]), devices[0], requires_grad)

    outputs = []
    for i in range(len(modules)):
        result = results[i]
        if isinstance(result, ExceptionWrapper):
            result.reraise()
        outputs.append(result)
    return outputs


def sparse_chunk_recon_and_loss(deconv_list, zidxs, chunkidxs, im, sam, loss_function):
    """
    Reconstructs planes at given defocuses and calculates the given loss
    function between the reconstruction and the ground truth sample. Returns a
    tuple containing the 3D reconstruction and the loss.

    Args:
        deconv_list: An nn.ModuleList of reconstruction networks per plane.

        zidxs: Array-like/list of indices into the deconv_list, specifying
        which planes of defocus to reconstruct at.

        chunkidxs: Array-like/list of indices into the sample, specifying
        indices of which planes within the current chunked sample to compare
        against in loss_function.

        im: The 2D image (N=1, C, H W) which will be used to reconstruct from.

        sam: The ground truth 3D sample (D, H, W) used to calculate the
        loss_function.

        loss_function: The loss function that should be calculated between the
        reconstruction and the given sample.
    """
    recon_planes = []
    # reconstruct planes with grad
    for zidx in zidxs:
        # remove batch dimension
        recon_planes.append(deconv_list[zidx](im)[0])
    # return chunk as tensor
    recon_planes = torch.cat(recon_planes)
    # calculate loss for selected planes
    chunk_loss = loss_function(sam[chunkidxs], recon_planes)
    return (recon_planes, chunk_loss)


def sparse_chunk_image(mic, phase_mask_angle, sam, zs, chunkidxs):
    """
    Calculates the PSF and simulates imaging in the microscope for the given
    sample at the specified planes of defocus. Returns a 2D (H, W) image.

    Args:
        mic: The microscope used for imaging.

        phase_mask_angle: The phase mask in radians used to calculate the PSF
        for imaging.

        sam: The 3D sample (D, H, W) to be imaged by the microscope.

        zs: Array-like/list of defocus values in microns specifying the planes
        of defocus the microscope should calculate the PSF and image at.

        chunkidxs: Array-like/list of indices into the sample, specifying
        indices of which planes within the current chunked sample to image with
        the microscope.
    """
    return mic(phase_mask_angle, sam[chunkidxs], zs)
