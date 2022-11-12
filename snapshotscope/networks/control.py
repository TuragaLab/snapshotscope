import datetime
import logging
import math
import os
import subprocess
import time

import numpy as np
import torch
import torch.cuda.comm
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda._utils import _get_device_index

import snapshotscope.data.augment as augment
import snapshotscope.data.simulate_data as simdata
import snapshotscope.networks.losses.lpips as lpips
import snapshotscope.microscope as m
import snapshotscope.utils as utils


def train_psf(
    micdeconv,
    opt,
    dataloader,
    defocus_range,
    num_grad_im_planes,
    num_grad_recon_planes,
    devices,
    losses,
    high_pass_losses,
    regularized_losses,
    num_iterations,
    lr_scheduler=None,
    single_decoder=False,
    input_3d=False,
    im_function=None,
    regularize_lost_power=0,
    high_pass_kernel_size=11,
    low_pass_weight=0,
    validate_losses=[],
    validate_args=None,
    it=0,
):
    """
    Trains a PSF for volume reconstruction given a potentially multi-GPU
    decoder and microscope architecture. Supports selective gradients for
    certain planes during imaging and reconstruction.
    """
    # define constants
    num_scopes = len(micdeconv["mics"])
    chunk_size = int(len(defocus_range) / num_scopes)
    if len(defocus_range) % num_scopes != 0:
        chunk_size += 1
    sections = [i * chunk_size for i in range(1, num_scopes)]
    defocus_ranges = np.split(defocus_range, sections)
    zidxs = np.arange(len(defocus_range), dtype=np.uint64)
    chunked_zidxs = np.split(zidxs, sections)
    chunk_sizes = [len(zidxs) for zidxs in chunked_zidxs]
    placeholders = "placeholder_deconvs" in micdeconv
    if placeholders:
        deconvs = micdeconv["placeholder_deconvs"]
    else:
        deconvs = micdeconv["deconvs"]
    scheduler = lr_scheduler is not None

    # define logging messages
    log_string = "[{}] iter: {}, loss: {}, norm(phase mask): {}"
    profile_string = "[{}] {}"

    # create mse loss
    mse = nn.MSELoss()

    # create high pass filter if specified
    if high_pass_kernel_size > 0:
        with torch.no_grad():
            ks = high_pass_kernel_size
            delta_kernel = torch.zeros(ks, ks, device="cpu")
            delta_kernel[int(ks / 2), int(ks / 2)] = 1.0
            low_pass_kernel = utils.gaussian_kernel_2d(
                int(ks / 2) + 1, (ks, ks), device="cpu"
            )
            high_pass_kernel = delta_kernel - low_pass_kernel
            high_pass_kernel = high_pass_kernel.expand(
                num_grad_recon_planes, *high_pass_kernel.shape
            )
    else:
        high_pass_kernel = None

    # initialize end time
    end_time = None

    # train loop
    while it < num_iterations:
        for data in dataloader:
            start_time = time.perf_counter()
            sam = data[0]  # remove batch dimension
            if end_time is None:
                logging.info(
                    profile_string.format(start_time, "done loading new sample")
                )
            else:
                logging.info(
                    profile_string.format(
                        (start_time - end_time), "done loading new sample"
                    )
                )
            # clear gradients
            opt.zero_grad()
            # calculate phase mask with mirror if applicable
            phase_mask_angle = micdeconv["phase_mask"]()
            if "mirror" in micdeconv:
                phase_mask_angle = phase_mask_angle + micdeconv["mirror"]()
            # select image planes with gradient
            grad_im_chunk_idxs = select_chunk_indices(
                num_scopes, chunk_sizes, num_grad_im_planes
            )
            grad_im_zs = take_chunk_indices(defocus_ranges, grad_im_chunk_idxs)
            # select image planes without gradient
            nograd_im_chunk_idxs = chunk_indices_diff(grad_im_chunk_idxs, chunk_sizes)
            nograd_im_zs = take_chunk_indices(defocus_ranges, nograd_im_chunk_idxs)
            # image sample planes without grad in parallel on multiple gpu
            do_nograd_im = any(len(idxs) > 0 for idxs in nograd_im_chunk_idxs)
            if do_nograd_im:
                nograd_im_sams = selective_scatter(sam, nograd_im_chunk_idxs, devices)
                nograd_ims = utils.parallel_apply(
                    micdeconv["mics"],
                    [
                        (phase_mask_angle, sam, zs)
                        for (sam, zs) in zip(nograd_im_sams, nograd_im_zs)
                    ],
                    devices,
                    requires_grad=False,
                )
                nograd_ims = [im.unsqueeze(0) for im in nograd_ims]
                nograd_ims = utils.gather(nograd_ims, devices[0])
            # image sample planes with grad in parallel on multiple gpu
            grad_im_sams = selective_scatter(sam, grad_im_chunk_idxs, devices)
            grad_ims = utils.parallel_apply(
                micdeconv["mics"],
                [
                    (phase_mask_angle, sam, zs)
                    for (sam, zs) in zip(grad_im_sams, grad_im_zs)
                ],
                devices,
                requires_grad=True,
            )
            grad_ims = [im.unsqueeze(0) for im in grad_ims]
            grad_ims = utils.gather(grad_ims, devices[0])
            if do_nograd_im:
                ims = torch.cat([grad_ims, nograd_ims])
            else:
                ims = grad_ims
            im = torch.sum(ims, axis=0)
            # add shot noise to image
            im = simdata.poissonlike_gaussian_noise(im)
            if im_function is not None:
                # apply function to image if defined, e.g. masking image
                im = im_function(im)
            # reshape to add batch and channel dimensions for convolution
            if input_3d:
                im = im.view(1, 1, 1, *im.shape)
            else:
                im = im.view(1, 1, *im.shape)
            image_time = time.perf_counter()
            logging.info(
                profile_string.format((image_time - start_time), "done imaging sample")
            )
            # select gradient reconstruction planes
            recon_chunk_idxs = select_chunk_indices(
                num_scopes, chunk_sizes, num_grad_recon_planes
            )
            recon_chunk_zidxs = take_chunk_indices(chunked_zidxs, recon_chunk_idxs)
            selected_sams = selective_scatter(sam, recon_chunk_idxs, devices)
            # calculate normalization factor(s)
            # apply high pass to sample if specified
            if low_pass_weight > 0 or high_pass_kernel is None:
                # calculate normalization factor only
                normalization = torch.mean(utils.gather(selected_sams, devices[0]) ** 2)
            if high_pass_kernel is not None:
                # high pass filter sams and then calculate normalization factor
                high_pass_selected_sams = utils.parallel_apply(
                    [high_pass_filter for d in devices],
                    [(sam, high_pass_kernel) for sam in selected_sams],
                    devices,
                )
                high_pass_normalization = torch.mean(
                    utils.gather(high_pass_selected_sams, devices[0]) ** 2
                )
            else:
                # placeholder arguments for parallel loss calculation
                # if we don't need high passed samples
                high_pass_selected_sams = [None for sam in selected_sams]
            if placeholders:
                # copy params into placeholders on multiple gpus
                copy_deconv_params_to_placeholder(
                    num_scopes, micdeconv, recon_chunk_zidxs
                )
            # reconstruct planes with selective gradients in parallel on multiple gpus
            # calculate distributed loss in same step
            scattered_recons, scattered_losses = zip(
                *utils.parallel_apply(
                    [chunk_recon_and_loss for device in devices],
                    [
                        (
                            deconv,
                            im,
                            sam,
                            high_pass_sam,
                            mse,
                            low_pass_weight,
                            high_pass_kernel,
                            single_decoder,
                            input_3d,
                        )
                        for (deconv, sam, high_pass_sam) in zip(
                            deconvs, selected_sams, high_pass_selected_sams
                        )
                    ],
                    devices,
                )
            )
            recon_time = time.perf_counter()
            logging.info(
                profile_string.format(
                    (recon_time - image_time),
                    "done reconstructing sample and performing distributed loss backwards",
                )
            )
            # normalize loss
            if high_pass_kernel is not None:
                scattered_high_pass_loss = [
                    l["high_pass_loss"].to(devices[0]) for l in scattered_losses
                ]
                high_pass_loss = torch.cuda.comm.reduce_add(
                    scattered_high_pass_loss, destination=_get_device_index(devices[0])
                ) / len(scattered_high_pass_loss)
                high_pass_loss = high_pass_loss / high_pass_normalization
            if low_pass_weight > 0 or high_pass_kernel is None:
                scattered_loss = [l["loss"].to(devices[0]) for l in scattered_losses]
                loss = torch.cuda.comm.reduce_add(
                    scattered_loss, destination=_get_device_index(devices[0])
                ) / len(scattered_loss)
                loss = loss / normalization
            if high_pass_kernel is not None:
                unregularized_loss = high_pass_loss
            else:
                unregularized_loss = loss
            if low_pass_weight > 0:
                unregularized_loss = unregularized_loss + (low_pass_weight * loss)
            if regularize_lost_power > 0:
                lost_powers = utils.parallel_apply(
                    [chunk_lost_power for device in devices],
                    [(mic.psf, mic.cropped_psf, mic.dk) for mic in micdeconv["mics"]],
                    devices,
                )
                lost_power = torch.sum(
                    torch.stack([power.to(devices[0]) for power in lost_powers])
                )
                regularized_loss = unregularized_loss + (
                    regularize_lost_power * lost_power
                )
            else:
                regularized_loss = unregularized_loss
            loss_time = time.perf_counter()
            logging.info(
                profile_string.format((loss_time - recon_time), "done averaging loss")
            )
            # backpropagate gradients
            regularized_loss.backward()
            backward_time = time.perf_counter()
            logging.info(
                profile_string.format((backward_time - loss_time), "done with backward")
            )
            # update weights
            opt.step()
            if scheduler:
                # advance learning rate schedule
                lr_scheduler.step()
            if placeholders:
                # copy params back from placeholders on multiple gpus
                copy_placeholder_params_to_deconv(
                    num_scopes, micdeconv, recon_chunk_zidxs
                )
            opt_time = time.perf_counter()
            logging.info(
                profile_string.format(
                    (opt_time - backward_time), "done with optimizer step"
                )
            )
            # checkpoint/snapshot
            pmnorm = phase_mask_angle.pow(2).sum().detach().cpu().item()
            # TODO(dip): call these by MSE/more specific names
            regularized_losses.append(regularized_loss.detach().cpu().item())
            if high_pass_kernel is not None:
                high_pass_losses.append(high_pass_loss.detach().cpu().item())
            if low_pass_weight > 0 or high_pass_kernel is None:
                losses.append(loss.detach().cpu().item())
            logging.info(
                log_string.format(
                    datetime.datetime.now(), it, regularized_losses[-1], pmnorm
                )
            )
            # checkpoint
            if it % 10000 == 0 and it != 0 and validate_args is not None:
                per_sample_val_losses = test_recon(micdeconv, **validate_args)
                validate_losses.append(per_sample_val_losses)
            if it % 500 == 0 and it != 0:
                logging.info("checkpointing...")
                checkpoint(
                    micdeconv,
                    opt,
                    {
                        "mses": losses,
                        "high_pass_mses": high_pass_losses,
                        "regularized_losses": regularized_losses,
                        "validate_high_pass_mses": validate_losses,
                    },
                    it,
                )
            if (it % 10000 == 0) or (it == num_iterations - 1):
                logging.info("snapshotting...")
                with torch.no_grad():
                    mic = micdeconv["mics"][0]
                    phase_mask = utils.ctensor_from_phase_angle(phase_mask_angle)
                    phase_mask = phase_mask.to(devices[0])
                    psf = torch.stack(
                        [
                            mic.compute_psf(phase_mask, z).detach().cpu()
                            for z in defocus_range
                        ]
                    )
                    if not single_decoder:
                        full_recon = torch.stack(
                            [
                                micdeconv["deconvs"][z](im.cpu()).detach().cpu()
                                for (zidxs, d) in zip(chunked_zidxs, devices)
                                for z in zidxs
                            ]
                        )
                    else:
                        full_recon = torch.cuda.comm.gather(
                            scattered_recons, destination=-1
                        )
                    phase_mask_angle_snapshot = micdeconv["phase_mask"]()
                    if "mirror" in micdeconv:
                        mirror_phase_angle_snapshot = micdeconv["mirror"]()
                    else:
                        mirror_phase_angle_snapshot = None
                snapshot(
                    micdeconv,
                    opt,
                    {
                        "mses": losses,
                        "high_pass_mses": high_pass_losses,
                        "regularized_losses": regularized_losses,
                        "validate_high_pass_mses": validate_losses,
                    },
                    it,
                    sam,
                    full_recon,
                    im,
                    psf=psf,
                    phase_mask_angle=phase_mask_angle_snapshot,
                    mirror_phase=mirror_phase_angle_snapshot,
                )

            # update iteration count and check for end
            it += 1
            end_time = time.perf_counter()
            logging.info(
                profile_string.format((end_time - start_time), "done with loop")
            )
            if it >= num_iterations:
                break


def train_recon(
    micdeconv,
    opt,
    dataloader,
    defocus_range,
    num_grad_recon_planes,
    devices,
    losses,
    high_pass_losses,
    regularized_losses,
    num_iterations,
    lr_scheduler=None,
    single_decoder=False,
    input_3d=False,
    im_function=None,
    overlap=0,
    psf=None,
    high_pass_kernel_size=11,
    low_pass_weight=0,
    snapshot_interval=10000,
    validate_losses=[],
    validate_args=None,
    it=0,
):
    """
    Trains a PSF for volume reconstruction given a potentially
    multi-GPU decoder and microscope architecture. Supports selective
    gradients for certain planes during imaging and
    reconstruction. Also supports overlapping of planes reconstructed
    on different GPUs to ameliorate artifacts at the boundaries of
    chunks.
    """
    # define constants
    num_scopes = len(micdeconv["mics"])
    defocus_ranges, chunked_zidxs, chunk_sizes = create_chunk_indices(
        defocus_range, num_scopes
    )
    (
        overlapped_defocus_ranges,
        overlapped_chunked_zidxs,
        overlapped_chunk_sizes,
    ) = create_chunk_indices(defocus_range, num_scopes, overlap=overlap)
    placeholders = "placeholder_deconvs" in micdeconv
    if placeholders:
        deconvs = micdeconv["placeholder_deconvs"]
    else:
        deconvs = micdeconv["deconvs"]
    scheduler = lr_scheduler is not None

    # define logging messages
    log_string = "[{}] iter: {}, loss: {}, norm(phase mask): {}"
    profile_string = "[{}] {}"

    # create mse loss
    mse = nn.MSELoss()

    # create high pass filter if specified
    if high_pass_kernel_size > 0:
        with torch.no_grad():
            ks = high_pass_kernel_size
            delta_kernel = torch.zeros(ks, ks, device="cpu")
            delta_kernel[int(ks / 2), int(ks / 2)] = 1.0
            low_pass_kernel = utils.gaussian_kernel_2d(
                int(ks / 2) + 1, (ks, ks), device="cpu"
            )
            high_pass_kernel = delta_kernel - low_pass_kernel
            high_pass_kernel = high_pass_kernel.expand(
                num_grad_recon_planes, *high_pass_kernel.shape
            )
    else:
        high_pass_kernel = None

    # calculate phase mask and PSF and cache result
    with torch.no_grad():
        if psf is None:
            phase_mask_angle = micdeconv["phase_mask"]()
            if "mirror" in micdeconv:
                phase_mask_angle = phase_mask_angle + micdeconv["mirror"]()
            phase_mask = utils.ctensor_from_phase_angle(phase_mask_angle.to(devices[0]))
            # calculate psf and cache result
            mic = micdeconv["mics"][0]
            psf = torch.stack(
                [mic.compute_psf(phase_mask, z).detach().cpu() for z in defocus_range]
            )
            # crop and downsample psf for imaging
            crop = int(mic.pad / 2)
            if crop > 0:
                psf = psf[:, crop:-crop, crop:-crop]
                psf = psf * mic.taper.cpu().expand_as(psf)
            if mic.downsample > 1:
                psf = F.avg_pool2d(
                    psf.unsqueeze(1), kernel_size=mic.downsample, divisor_override=1
                ).squeeze(1)
    snapshot_psf = psf.detach().cpu()

    # initialize end time
    end_time = None

    # train loop
    snapshot_phase_mask_angle = micdeconv["phase_mask"]()
    snapshot_mirror_phase_angle = (
        None if "mirror" not in micdeconv else micdeconv["mirror"]()
    )
    while it < num_iterations:
        for data in dataloader:
            start_time = time.perf_counter()
            sam = data[0]  # remove batch dimension
            if end_time is None:
                logging.info(
                    profile_string.format(start_time, "done loading new sample")
                )
            else:
                logging.info(
                    profile_string.format(
                        (start_time - end_time), "done loading new sample"
                    )
                )
            # clear gradients
            opt.zero_grad()
            # calculate phase mask only
            # select image planes without gradient
            nograd_im_chunk_idxs = select_chunk_indices(
                num_scopes, chunk_sizes, chunk_sizes[0]
            )
            nograd_im_zs = take_chunk_indices(defocus_ranges, nograd_im_chunk_idxs)
            # image sample planes without grad in parallel on multiple gpu
            nograd_im_sams = selective_scatter(sam, nograd_im_chunk_idxs, devices)
            nograd_im_psfs = selective_scatter(psf, nograd_im_chunk_idxs, devices)
            nograd_ims = utils.parallel_apply(
                [chunk_image for d in devices],
                list(zip(nograd_im_sams, nograd_im_psfs)),
                devices,
                requires_grad=False,
            )
            with torch.no_grad():
                nograd_ims = [ni.unsqueeze(0) for ni in nograd_ims]
                nograd_ims = utils.gather(nograd_ims, devices[0])
                im = torch.sum(nograd_ims, axis=0)
                # add shot noise to image
                im = simdata.poissonlike_gaussian_noise(im)
                if im_function is not None:
                    # apply image function if specified
                    im = im_function(im)
                # reshape to add batch and channel dimensions for convolution
                if input_3d:
                    im = im.view(1, 1, 1, *im.shape)
                else:
                    im = im.view(1, 1, *im.shape)
            image_time = time.perf_counter()
            logging.info(
                profile_string.format((image_time - start_time), "done imaging sample")
            )
            # select gradient reconstruction planes
            recon_chunk_idxs = select_chunk_indices(
                num_scopes, overlapped_chunk_sizes, num_grad_recon_planes
            )
            recon_chunk_zidxs = take_chunk_indices(
                overlapped_chunked_zidxs, recon_chunk_idxs
            )
            selected_sams = selective_scatter(sam, recon_chunk_idxs, devices)
            # calculate normalization factor(s)
            # apply high pass to sample if specified
            if low_pass_weight > 0 or high_pass_kernel is None:
                # calculate normalization factor only
                normalization = torch.mean(utils.gather(selected_sams, devices[0]) ** 2)
            if high_pass_kernel is not None:
                # high pass filter sams and then calculate normalization factor
                high_pass_selected_sams = utils.parallel_apply(
                    [high_pass_filter for d in devices],
                    [(sam, high_pass_kernel) for sam in selected_sams],
                    devices,
                )
                high_pass_normalization = torch.mean(
                    utils.gather(high_pass_selected_sams, devices[0]) ** 2
                )
            else:
                # placeholder arguments for parallel loss calculation
                # if we don't need high passed samples
                high_pass_selected_sams = [None for sam in selected_sams]
            if placeholders:
                # copy params into placeholders on multiple gpus
                copy_deconv_params_to_placeholder(
                    num_scopes, micdeconv, recon_chunk_zidxs
                )
            # reconstruct planes with selective gradients in parallel on multiple gpus
            # calculate distributed loss in same step
            scattered_recons, scattered_losses = zip(
                *utils.parallel_apply(
                    [chunk_recon_and_loss for device in devices],
                    [
                        (
                            deconv,
                            im,
                            sam,
                            high_pass_sam,
                            mse,
                            low_pass_weight,
                            high_pass_kernel,
                            single_decoder,
                            input_3d,
                        )
                        for (deconv, sam, high_pass_sam) in zip(
                            deconvs, selected_sams, high_pass_selected_sams
                        )
                    ],
                    devices,
                )
            )
            recon_time = time.perf_counter()
            logging.info(
                profile_string.format(
                    (recon_time - image_time),
                    "done reconstructing sample and calculating distributed loss",
                )
            )
            # normalize loss
            if high_pass_kernel is not None:
                scattered_high_pass_loss = [
                    l["high_pass_loss"].to(devices[0]) for l in scattered_losses
                ]
                high_pass_loss = torch.stack(scattered_high_pass_loss).sum() / len(
                    scattered_high_pass_loss
                )
                high_pass_loss = high_pass_loss / high_pass_normalization
            if low_pass_weight > 0 or high_pass_kernel is None:
                scattered_loss = [l["loss"].to(devices[0]) for l in scattered_losses]
                loss = torch.stack(scattered_loss).sum() / len(scattered_loss)
                loss = loss / normalization
            if high_pass_kernel is not None:
                unregularized_loss = high_pass_loss
            else:
                unregularized_loss = loss
            if low_pass_weight > 0:
                unregularized_loss = unregularized_loss + (low_pass_weight * loss)
            # currently decoder only training doesn't use regularizers
            regularized_loss = unregularized_loss
            loss_time = time.perf_counter()
            logging.info(
                profile_string.format((loss_time - recon_time), "done averaging loss")
            )
            # backpropagate gradients
            regularized_loss.backward()
            backward_time = time.perf_counter()
            logging.info(
                profile_string.format((backward_time - loss_time), "done with backward")
            )
            # update weights
            opt.step()
            if scheduler:
                # advance learning rate schedule
                lr_scheduler.step()
            if placeholders:
                # copy params back from placeholders on multiple gpus
                copy_placeholder_params_to_deconv(
                    num_scopes, micdeconv, recon_chunk_zidxs
                )
            opt_time = time.perf_counter()
            logging.info(
                profile_string.format(
                    (opt_time - backward_time), "done with optimizer step"
                )
            )
            # checkpoint/snapshot
            pmnorm = snapshot_phase_mask_angle.pow(2).sum().detach().cpu().item()
            # TODO(dip): call these by MSE/more specific names
            regularized_losses.append(regularized_loss.detach().cpu().item())
            if high_pass_kernel is not None:
                high_pass_losses.append(high_pass_loss.detach().cpu().item())
            if low_pass_weight > 0 or high_pass_kernel is None:
                losses.append(loss.detach().cpu().item())
            logging.info(
                log_string.format(
                    datetime.datetime.now(), it, regularized_losses[-1], pmnorm
                )
            )
            # checkpoint
            if it % 10000 == 0 and it != 0 and validate_args is not None:
                per_sample_val_losses = test_recon(micdeconv, **validate_args)
                validate_losses.append(per_sample_val_losses)
            if it % 500 == 0 and it != 0:
                logging.info("checkpointing...")
                checkpoint(
                    micdeconv,
                    opt,
                    {
                        "mses": losses,
                        "high_pass_mses": high_pass_losses,
                        "regularized_losses": regularized_losses,
                        "validate_high_pass_mses": validate_losses,
                    },
                    it,
                )
            if (it % snapshot_interval == 0) or (it == num_iterations - 1):
                logging.info("snapshotting...")
                with torch.no_grad():
                    if not single_decoder:
                        scattered_full_recons = [
                            micdeconv["deconvs"][z](im.cpu()).detach().cpu()
                            for (zidxs, d) in zip(chunked_zidxs, devices)
                            for z in zidxs
                        ]
                        full_recon = gather_volumes(
                            scattered_full_recons, device="cpu", overlap=overlap
                        )
                    else:
                        full_recon = gather_volumes(
                            scattered_recons, device="cpu", overlap=overlap
                        )
                snapshot(
                    micdeconv,
                    opt,
                    {
                        "mses": losses,
                        "high_pass_mses": high_pass_losses,
                        "regularized_losses": regularized_losses,
                        "validate_high_pass_mses": validate_losses,
                    },
                    it,
                    sam,
                    full_recon,
                    im,
                    psf=snapshot_psf,
                    phase_mask_angle=snapshot_phase_mask_angle,
                    mirror_phase=snapshot_mirror_phase_angle,
                )

            # update iteration count and check for end
            it += 1
            end_time = time.perf_counter()
            logging.info(
                profile_string.format((end_time - start_time), "done with loop")
            )
            if it >= num_iterations:
                break


def test_recon(
    micdeconv,
    dataloader,
    defocus_range,
    target_shape,
    devices,
    extents,
    num_grad_recon_planes=None,
    single_decoder=True,
    input_3d=False,
    im_function=None,
    high_pass_kernel_size=11,
    aperture_radius=None,
    psf=None,
    save_dir="./",
):
    # define constants
    num_scopes = len(devices)
    chunk_size = int(len(defocus_range) / num_scopes)
    if len(defocus_range) % num_scopes != 0:
        chunk_size += 1
    if num_grad_recon_planes is None:
        num_grad_recon_planes = chunk_size
    sections = [i * chunk_size for i in range(1, num_scopes)]
    defocus_ranges = np.split(defocus_range, sections)
    zidxs = np.arange(len(defocus_range), dtype=np.uint64)
    chunked_zidxs = np.split(zidxs, sections)
    chunk_sizes = [len(zidxs) for zidxs in chunked_zidxs]
    placeholders = "placeholder_deconvs" in micdeconv
    if placeholders:
        deconvs = micdeconv["placeholder_deconvs"]
    else:
        deconvs = micdeconv["deconvs"]

    # define logging messages
    log_string = "[{}] sample index: {}, loss: {} corr: {}\n"
    profile_string = "[{}] {}\n"
    log_fname = os.path.join(save_dir, "out.log")

    # create mse loss
    mse = nn.MSELoss()

    # create high pass filter if specified
    if high_pass_kernel_size > 0:
        with torch.no_grad():
            ks = high_pass_kernel_size
            delta_kernel = torch.zeros(ks, ks, device="cpu")
            delta_kernel[int(ks / 2), int(ks / 2)] = 1.0
            low_pass_kernel = utils.gaussian_kernel_2d(
                int(ks / 2) + 1, (ks, ks), device="cpu"
            )
            high_pass_kernel = delta_kernel - low_pass_kernel
            high_pass_kernel = high_pass_kernel.expand(
                num_grad_recon_planes, *high_pass_kernel.shape
            )
    else:
        high_pass_kernel = None

    # calculate phase mask and PSF from SLM only and cache result
    with torch.no_grad():
        if psf is None:
            phase_mask_angle = micdeconv["phase_mask"]()
            mirror_phase_mask_angle = phase_mask_angle
            if "mirror" in micdeconv:
                mirror_phase_mask_angle = mirror_phase_mask_angle + micdeconv[
                    "mirror"
                ]().to(mirror_phase_mask_angle.device)
            phase_mask = utils.ctensor_from_phase_angle(mirror_phase_mask_angle).to(
                devices[0]
            )
            # calculate psf and cache result
            mic = micdeconv["mics"][0]
            psf = torch.stack([mic.compute_psf(phase_mask, z) for z in defocus_range])
            # crop and downsample psf for imaging
            crop = int(mic.pad / 2)
            if crop > 0:
                psf = psf[:, crop:-crop, crop:-crop]
                psf = psf * mic.taper.expand_as(psf)
            psf = F.avg_pool2d(
                psf.unsqueeze(1), kernel_size=mic.downsample, divisor_override=1
            ).squeeze(1)
        print("psf sum:", psf.sum())

    # initialize end time
    end_time = None

    # initialize counter
    counter = 0

    # initialize log file
    f = open(log_fname, "w")

    # select image planes
    nograd_im_chunk_idxs = select_chunk_indices(
        num_scopes, chunk_sizes, num_per_chunk=None
    )
    nograd_im_zs = take_chunk_indices(defocus_ranges, nograd_im_chunk_idxs)
    recon_chunk_idxs = select_chunk_indices(
        num_scopes, chunk_sizes, num_per_chunk=num_grad_recon_planes
    )
    recon_chunk_zidxs = take_chunk_indices(chunked_zidxs, recon_chunk_idxs)

    # scatter psf using selected planes
    nograd_im_psfs = selective_scatter(psf, nograd_im_chunk_idxs, devices)

    # initialize losses and correlations
    losses, corrs = [], []

    # test loop
    for data in dataloader:
        location = 0
        for extent in extents[counter]:
            with torch.no_grad():
                start_time = time.perf_counter()
                # remove batch dimension
                sam = data[0]
                # select location
                sam = sam[
                    extent[0][0] : extent[0][1],
                    extent[1][0] : extent[1][1],
                    extent[2][0] : extent[2][1],
                ]
                # select location
                # apply aperture
                if aperture_radius is not None:
                    sam = augment.cylinder_crop(
                        sam, aperture_radius, center="center", max_depth=sam.shape[0]
                    )
                    sam = torch.from_numpy(augment.pad_volume(sam, target_shape))
                else:
                    sam = torch.from_numpy(
                        augment.pad_volume(sam.numpy(), target_shape)
                    )
                assert all(
                    [s == t for s, t in zip(sam.shape, target_shape)]
                ), "Shape must match"
                if end_time is None:
                    f.write(
                        profile_string.format(start_time, "done loading new sample")
                    )
                else:
                    f.write(
                        profile_string.format(
                            (start_time - end_time), "done loading new sample"
                        )
                    )
                # image sample planes without grad in parallel on multiple gpu
                nograd_im_sams = selective_scatter(sam, nograd_im_chunk_idxs, devices)
                nograd_ims = utils.parallel_apply(
                    [utils.fftconv2d for d in devices],
                    [
                        (sam, psf, None, None, "same")
                        for (sam, psf) in zip(nograd_im_sams, nograd_im_psfs)
                    ],
                    devices,
                    requires_grad=False,
                )
                nograd_ims = utils.gather(nograd_ims, devices[0])
                im = torch.sum(nograd_ims, axis=0)
                # add shot noise to image
                im = simdata.poissonlike_gaussian_noise(im)
                if im_function is not None:
                    # apply image function if specified
                    im = im_function(im)
                # reshape to add batch and channel dimensions for convolution
                if input_3d:
                    im = im.view(1, 1, 1, *im.shape)
                else:
                    im = im.view(1, 1, *im.shape)
                image_time = time.perf_counter()
                f.write(
                    profile_string.format(
                        (image_time - start_time), "done imaging sample"
                    )
                )
                # select gradient reconstruction planes
                selected_sams = selective_scatter(sam, recon_chunk_idxs, devices)
                # calculate normalization factor
                # apply high pass to sample if specified
                if high_pass_kernel is not None:
                    # high pass filter sams
                    selected_sams = utils.parallel_apply(
                        [high_pass_filter for d in devices],
                        [(sam, high_pass_kernel) for sam in selected_sams],
                        devices,
                        requires_grad=False,
                    )
                    normalization = torch.mean(
                        utils.gather(selected_sams, devices[0]) ** 2
                    )
                else:
                    # calculate normalization factor only
                    normalization = torch.mean(
                        utils.gather(selected_sams, devices[0]) ** 2
                    )
                if placeholders:
                    # copy params into placeholders on multiple gpus
                    copy_deconv_params_to_placeholder(
                        num_scopes, micdeconv, recon_chunk_zidxs
                    )
                # reconstruct planes with selective gradients in parallel on multiple gpus
                # calculate distributed loss in same step
                # TODO(dip): update this function for new call to chunk_recon_and_loss
                # currently using a stopgap of reused arguments to make this work
                scattered_recons, distributed_loss = zip(
                    *utils.parallel_apply(
                        [chunk_recon_and_loss for device in devices],
                        [
                            (
                                deconv,
                                im,
                                sam,
                                sam,
                                mse,
                                0,
                                high_pass_kernel,
                                single_decoder,
                                input_3d,
                            )
                            for (deconv, sam) in zip(deconvs, selected_sams)
                        ],
                        devices,
                        requires_grad=False,
                    )
                )
                recon = utils.gather(scattered_recons, device=devices[0])
                recon_time = time.perf_counter()
                ignored_recons = zip(
                    *utils.parallel_apply(
                        [chunk_recon for device in devices],
                        [(deconv, im, single_decoder, input_3d) for deconv in deconvs],
                        devices,
                        requires_grad=False,
                    )
                )
                ignored_recon_time = time.perf_counter()
                for s in selected_sams:
                    del s
                f.write(
                    profile_string.format(
                        (recon_time - image_time),
                        "done reconstructing sample and performing distributed loss backwards",
                    )
                )
                f.write(
                    profile_string.format(
                        (ignored_recon_time - recon_time),
                        "done only reconstructing for timing",
                    )
                )
                # normalize loss
                # TODO(dip): update the way we use the loss and variable names here
                distributed_loss = [dl["high_pass_loss"] for dl in distributed_loss]
                loss = torch.cuda.comm.reduce_add(
                    distributed_loss, destination=_get_device_index(devices[0])
                ) / len(distributed_loss)
                # loss = (loss / normalization).detach().cpu().item()
                losses.append(loss)
                # calculate correlation
                corr_sam = utils.gather(selected_sams, device=devices[0])
                corr = utils.pearson_corr(recon, corr_sam).detach().cpu().item()
                corrs.append(corr)
                loss_time = time.perf_counter()
                f.write(
                    profile_string.format(
                        (loss_time - recon_time), "done averaging loss"
                    )
                )
                # save results
                f.write(log_string.format(datetime.datetime.now(), counter, loss, corr))
                torch.save(
                    recon.detach().cpu(), f"{save_dir}/recon{counter}_{location}.pt"
                )
                torch.save(sam.detach().cpu(), f"{save_dir}/sam{counter}_{location}.pt")
                torch.save(im.detach().cpu(), f"{save_dir}/im{counter}_{location}.pt")
                location += 1
                end_time = time.perf_counter()
                f.write(
                    profile_string.format((end_time - start_time), "done with loop")
                )
        # update data counter
        counter += 1
    f.close()
    del psf
    return losses


def structure_recon(
    micdeconv,
    dataloader,
    defocus_range,
    devices,
    single_decoder=True,
    input_3d=False,
    save_dir="./",
):
    # define constants
    num_scopes = len(micdeconv["mics"])
    chunk_size = int(len(defocus_range) / num_scopes)
    if len(defocus_range) % num_scopes != 0:
        chunk_size += 1
    sections = [i * chunk_size for i in range(1, num_scopes)]
    defocus_ranges = np.split(defocus_range, sections)
    zidxs = np.arange(len(defocus_range), dtype=np.uint64)
    chunked_zidxs = np.split(zidxs, sections)
    chunk_sizes = [len(zidxs) for zidxs in chunked_zidxs]
    placeholders = "placeholder_deconvs" in micdeconv
    if placeholders:
        deconvs = micdeconv["placeholder_deconvs"].train(False)
    else:
        deconvs = micdeconv["deconvs"].train(False)

    # define logging messages
    log_string = "[{}] sample index: {}\n"
    profile_string = "[{}] {}\n"
    log_fname = os.path.join(save_dir, "out.log")

    # create mse loss
    mse = nn.MSELoss()

    # initialize end time
    end_time = None

    # initialize counter
    counter = 0

    # initialize log file
    f = open(log_fname, "w")

    # select reconstruction indices
    recon_chunk_idxs = select_chunk_indices(num_scopes, chunk_sizes, chunk_sizes[0])
    recon_chunk_zidxs = take_chunk_indices(chunked_zidxs, recon_chunk_idxs)

    # test loop
    for counter, data in enumerate(dataloader):
        start_time = time.perf_counter()
        im = data
        if end_time is None:
            f.write(profile_string.format(start_time, "done loading new sample"))
        else:
            f.write(
                profile_string.format(
                    (start_time - end_time), "done loading new sample"
                )
            )
        # reshape in case of 3d
        # batch and channel dimensions added by dataloader
        if input_3d:
            im = im.unsqueeze(2)
        image_time = time.perf_counter()
        f.write(profile_string.format((image_time - start_time), "done imaging sample"))
        if placeholders:
            # copy params into placeholders on multiple gpus
            copy_deconv_params_to_placeholder(num_scopes, micdeconv, recon_chunk_zidxs)
        # reconstruct planes with selective gradients in parallel on multiple gpus
        # calculate distributed loss in same step
        scattered_recons = utils.parallel_apply(
            [chunk_recon for device in devices],
            [(deconv, im, single_decoder, input_3d) for deconv in deconvs],
            devices,
            requires_grad=False,
        )
        recon = utils.gather(scattered_recons, device=devices[0])
        recon_time = time.perf_counter()
        logging.info(
            profile_string.format(
                (recon_time - image_time), "done reconstructing sample"
            )
        )
        # save results
        f.write(log_string.format(datetime.datetime.now(), counter))
        torch.save(recon.detach().cpu(), f"{save_dir}/recon{counter}.pt")
        torch.save(im.detach().cpu(), f"{save_dir}/im{counter}.pt")
        end_time = time.perf_counter()
        f.write(profile_string.format((end_time - start_time), "done with loop"))
    f.close()


def train_rgb_recon(
    deconv,
    opt,
    dataloader,
    devices,
    mses,
    num_iterations,
    lr_scheduler=None,
    lpips_weight=0,
    lpips_step_milestones=[],
    lpips_step_size=0.1,
    lpips_losses=[],
    snapshot_interval=10000,
    validate_mses=[],
    validate_args=None,
    it=0,
):
    """
    Trains a reconstruction network for DLMD.
    """
    # define constants
    scheduler = lr_scheduler is not None

    # define logging messages
    log_string = "[{}] iter: {}, loss: {}"
    profile_string = "[{}] {}"

    # create mse loss
    mse = nn.MSELoss()

    # create perceptual loss
    perceptual = lpips.LPIPS().to(devices[0])

    # initialize end time
    end_time = None

    # train loop
    while it < num_iterations:
        for batch in dataloader:
            start_time = time.perf_counter()
            diffused_images = batch["image"].to(devices[0])
            ground_truth_images = batch["label"].to(devices[0])
            if end_time is None:
                logging.info(
                    profile_string.format(start_time, "done loading new sample")
                )
            else:
                logging.info(
                    profile_string.format(
                        (start_time - end_time), "done loading new sample"
                    )
                )
            # clear gradients
            opt.zero_grad()
            # reconstruct images
            predicted_images = deconv(diffused_images)
            recon_time = time.perf_counter()
            logging.info(
                profile_string.format(
                    (recon_time - start_time),
                    "done reconstructing sample",
                )
            )
            # calculate loss
            mse_loss = mse(predicted_images, ground_truth_images)
            if lpips_weight > 0:
                lpips_loss = perceptual(predicted_images, ground_truth_images)
                loss = ((1.0 - lpips_weight) * mse_loss) + (lpips_weight * lpips_loss)
            else:
                loss = mse_loss
            loss_time = time.perf_counter()
            logging.info(
                profile_string.format(
                    (loss_time - recon_time),
                    "done calculating loss",
                )
            )
            # backpropagate gradients
            loss.backward()
            backward_time = time.perf_counter()
            logging.info(
                profile_string.format((backward_time - loss_time), "done with backward")
            )
            # update weights
            opt.step()
            if scheduler:
                # advance learning rate schedule
                lr_scheduler.step()
            opt_time = time.perf_counter()
            logging.info(
                profile_string.format(
                    (opt_time - backward_time), "done with optimizer step"
                )
            )
            # checkpoint/snapshot
            # TODO(dip): call these by MSE/more specific names
            mses.append(mse_loss.detach().cpu().item())
            if lpips_weight > 0:
                lpips_losses.append(lpips_loss.detach().cpu().item())
            logging.info(
                log_string.format(
                    datetime.datetime.now(), it, loss.detach().cpu().item()
                )
            )
            # checkpoint
            if it % 10000 == 0 and it != 0 and validate_args is not None:
                per_sample_val_losses = test_rgb_recon(deconv, **validate_args)
                validate_mses.append(per_sample_val_losses)
            if it % 500 == 0 and it != 0:
                logging.info("checkpointing...")
                checkpoint(
                    deconv,
                    opt,
                    {
                        "mses": mses,
                        "lpips_losses": lpips_losses,
                        "validate_mses": validate_mses,
                    },
                    it,
                    module_name="deconv",
                )
            if (it % snapshot_interval == 0) or (it == num_iterations - 1):
                logging.info("snapshotting...")
                snapshot(
                    deconv,
                    opt,
                    {
                        "mses": mses,
                        "lpips_losses": lpips_losses,
                        "validate_mses": validate_mses,
                    },
                    it,
                    ground_truth_images.detach().cpu(),
                    predicted_images.detach().cpu(),
                    diffused_images.detach().cpu(),
                    module_name="deconv",
                )

            # update iteration count and check for end
            it += 1
            if (len(lpips_step_milestones) > 0) and (it == lpips_step_milestones[0]):
                _ = lpips_step_milestones.pop(0)
                lpips_weight += lpips_step_size
                lpips_weight = min(1.0, lpips_weight)
            end_time = time.perf_counter()
            logging.info(
                profile_string.format((end_time - start_time), "done with loop")
            )
            if it >= num_iterations:
                break


def test_rgb_recon(
    deconv,
    dataloader,
    devices,
    save_dir="./",
):
    """
    Trains a reconstruction network for DLMD.
    """
    # set model to eval mode
    deconv.eval()
    # define logging messages
    log_string = "[{}] sample: {}, loss: {}"
    profile_string = "[{}] {}"
    log_fname = os.path.join(save_dir, "out.log")

    # create mse loss
    mse = nn.MSELoss()

    # initialize losses
    losses = []

    # initialize counter
    counter = 0

    # initialize log file
    f = open(log_fname, "w")

    # initialize end time
    end_time = None

    # train loop
    with torch.no_grad():
        for single in dataloader:
            # WARN(dip): assuming batch size for test dataloader is 1
            start_time = time.perf_counter()
            diffused_image = single["image"].to(devices[0])
            ground_truth_image = single["label"].to(devices[0])
            if end_time is None:
                f.write(profile_string.format(start_time, "done loading new sample"))
            else:
                f.write(
                    profile_string.format(
                        (start_time - end_time), "done loading new sample"
                    )
                )
            # reconstruct images
            predicted_image = deconv(diffused_image)
            recon_time = time.perf_counter()
            f.write(
                profile_string.format(
                    (recon_time - start_time),
                    "done reconstructing sample",
                )
            )
            # calculate loss
            loss = mse(predicted_image, ground_truth_image)
            loss_time = time.perf_counter()
            f.write(
                profile_string.format(
                    (loss_time - recon_time),
                    "done calculating loss",
                )
            )
            # log results
            # TODO(dip): call these by MSE/more specific names
            losses.append(loss.detach().cpu().item())
            # save results
            f.write(log_string.format(datetime.datetime.now(), counter, losses[-1]))
            torch.save(predicted_image.detach().cpu(), f"{save_dir}/recon{counter}.pt")
            torch.save(ground_truth_image.detach().cpu(), f"{save_dir}/sam{counter}.pt")
            torch.save(diffused_image.detach().cpu(), f"{save_dir}/im{counter}.pt")
            # update iteration count
            counter += 1
            end_time = time.perf_counter()
            f.write(
                profile_string.format((end_time - start_time), "done testing sample")
            )
    f.close()
    # return model to train mode
    deconv.train()
    return losses


def create_chunk_indices(defocus_range, num_scopes, overlap=0):
    """
    Returns the necessary chunked defocus ranges and indices for
    selecting planes to be reconstructed. Supports calculating indices
    for chunks with overlap.
    """
    chunk_size = int(len(defocus_range) / num_scopes)
    if len(defocus_range) % num_scopes != 0:
        chunk_size += 1
    sections = [i * chunk_size for i in range(1, num_scopes)]
    defocus_ranges = np.split(defocus_range, sections)
    zidxs = np.arange(len(defocus_range), dtype=np.uint64)
    chunked_zidxs = np.split(zidxs, sections)
    for i in range(num_scopes - 1):
        defocus_ranges[i] = np.concatenate(
            (defocus_ranges[i], defocus_ranges[i + 1][:overlap]), axis=0
        )
        chunked_zidxs[i] = np.concatenate(
            (chunked_zidxs[i], chunked_zidxs[i + 1][:overlap]), axis=0
        )
    chunk_sizes = [len(zidxs) for zidxs in chunked_zidxs]
    return defocus_ranges, chunked_zidxs, chunk_sizes


def select_chunk_indices(num_chunks, chunk_sizes, num_per_chunk=None):
    """
    Returns a sorted list of num_per_chunk random indices from a range of
    length chunk_size for num_chunks chunks.
    """
    if num_per_chunk is None:
        return [np.arange(chunk_size) for chunk_size in chunk_sizes]
    else:
        return [
            np.sort(
                np.random.choice(np.arange(chunk_size), num_per_chunk, replace=False)
            )
            for chunk_size in chunk_sizes
        ]


def chunk_indices_diff(chunk_idxs, chunk_sizes):
    """
    Returns the chunk indices that were not chosen in select_chunk_indices for
    chunks of size chunk_size.
    """
    return [
        np.setdiff1d(idxs, np.arange(chunk_size), assume_unique=True)
        for idxs, chunk_size in zip(chunk_idxs, chunk_sizes)
    ]


def take_chunk_indices(split_array, chunk_idxs):
    """
    Takes a split array-like and selects specified indices from each chunk.
    """
    return [array[idxs] for array, idxs in zip(split_array, chunk_idxs)]


def selective_scatter(array, chunk_idxs, devices):
    """
    Scatters a given array to the specified devices, but only moves the
    specified indices for each chunk of the array.
    """
    chunk_size = int(len(array) / len(devices))
    if len(array) % len(devices) != 0:
        chunk_size += 1
    split_array = torch.split(array, chunk_size)
    return [
        array[idxs].to(d) for array, idxs, d in zip(split_array, chunk_idxs, devices)
    ]


def copy_params_(source, target):
    for s, t in zip(source.parameters(), target.parameters()):
        t.data.copy_(s.data)


def copy_deconv_params_to_placeholder(num_scopes, micdeconv, recon_chunk_zidxs):
    """Copies parameters from CPU deconv modules to GPU placeholders."""
    for didx in range(num_scopes):
        for cidx, zidx in enumerate(recon_chunk_zidxs[didx]):
            copy_params_(
                micdeconv["deconvs"][zidx], micdeconv["placeholder_deconvs"][didx][cidx]
            )


def copy_placeholder_params_to_deconv(num_scopes, micdeconv, recon_chunk_zidxs):
    """Copies parameters from GPU placeholders to CPU deconv modules."""
    for didx in range(num_scopes):
        for cidx, zidx in enumerate(recon_chunk_zidxs[didx]):
            copy_params_(
                micdeconv["placeholder_deconvs"][didx][cidx], micdeconv["deconvs"][zidx]
            )


def high_pass_filter(vol, high_pass_kernel, use_3d=False):
    if use_3d:
        return utils.fftconvn(vol, high_pass_kernel, shape="same")
    else:
        return utils.fftconv2d(vol, high_pass_kernel, shape="same")


def chunk_cylinder_mask(vol, cylinder_mask):
    vol.data *= cylinder_mask
    return vol


def gather_volumes(vols, device="cpu", overlap=0):
    vols = [vol.to(device) for vol in vols]
    if not overlap:
        vol = torch.cat(vols, dim=0)
    else:
        num_planes = sum(
            [
                v.shape[0] - overlap if i < (len(vols) - 1) else v.shape[0]
                for i, v in enumerate(vols)
            ]
        )
        vol = torch.zeros(
            (num_planes, vols[0].shape[1], vols[0].shape[2]), device=device
        )
        counts = torch.zeros_like(vol)
        for i, v in enumerate(vols):
            if i < (len(vols) - 1):
                start = i * (v.shape[0] - overlap)
            else:
                start = i * v.shape[0]
            end = start + v.shape[0]
            vol[start:end] += v
            counts[start:end] += torch.ones_like(v)
        vol /= counts
    return vol


def chunk_recon_and_loss(
    deconvs,
    im,
    sam=None,
    high_pass_sam=None,
    loss_function=F.mse_loss,
    low_pass_weight=0,
    high_pass_kernel=None,
    single_decoder=False,
    input_3d=False,
):
    """
    Reconstructs planes at given defocuses and calculates the given loss
    function between the reconstruction and the ground truth sample. Returns a
    tuple containing the 3D reconstruction and the loss.

    Args: deconvs: An nn.ModuleList of reconstruction networks per plane or a
    single deconvolution module.

        im: The 2D image (N=1, C, H W) which will be used to reconstruct from.

        sam: The ground truth 3D sample (D, H, W) used to calculate the
        loss_function.

        loss_function: The loss function that should be calculated between the
        reconstruction and the given sample.

        high_pass_kernel (optional): If provided, the kernel used for high pass
        filtering of the data.

        single_decoder (optional): Defaults to False, but if True then assume
        deconvs is a single decoder instead of a list of decoders per plane.

        input_3d (optional): Defaults to False, but if True then reconstruction
        input expects 3D image (N=1, C, D, H, W) instead of a 2D image (N, C,
        H, W).
    """
    if single_decoder:
        # reconstruct chunk at once
        # remove batch dimension
        recon = deconvs(im)[0]
        recon = recon.squeeze()
        if len(recon.shape) < 3:
            recon = recon.unsqueeze(0)
    else:
        # reconstruct planes separately
        recon_planes = []
        for deconv in deconvs:
            # remove batch dimension
            plane = deconv(im)[0]
            plane = plane.squeeze()
            if len(plane.shape) < 3:
                plane = plane.unsqueeze(0)
            recon_planes.append(plane)
        # return chunk as tensor
        recon = torch.cat(recon_planes)
    # calculate loss for selected planes,
    # high pass filter if available
    chunk_losses = {}
    if high_pass_kernel is not None:
        high_pass_recon = high_pass_filter(recon, high_pass_kernel)
        # assume high_pass_sam is not None here
        chunk_losses["high_pass_loss"] = loss_function(high_pass_sam, high_pass_recon)
    if low_pass_weight > 0 or high_pass_kernel is None:
        chunk_losses["loss"] = loss_function(sam, recon)
    return (recon, chunk_losses)


def chunk_recon(deconvs, im, single_decoder=False, input_3d=False):
    """
    Reconstructs planes at given defocuses for the given image, where we have
    no ground truth. Returns only the reconstructed 3D volume.

    Args: deconvs: An nn.ModuleList of reconstruction networks per plane or a
    single deconvolution module.

        im: The 2D image (N=1, C, H W) which will be used to reconstruct from.

        single_decoder (optional): Defaults to False, but if True then assume
        deconvs is a single decoder instead of a list of decoders per plane.

        input_3d (optional): Defaults to False, but if True then reconstruction
        input expects 3D image (N=1, C, D, H, W) instead of a 2D image (N, C,
        H, W).
    """
    if single_decoder:
        # reconstruct chunk at once
        # remove batch dimension
        recon = deconvs(im)[0]
        recon = recon.squeeze()
        if len(recon.shape) < 3:
            recon = recon.unsqueeze(0)
    else:
        # reconstruct planes separately
        recon_planes = []
        for deconv in deconvs:
            # remove batch dimension
            plane = deconv(im)[0]
            plane = plane.squeeze()
            if len(plane.shape) < 3:
                plane = plane.unsqueeze(0)
            recon_planes.append(plane)
        # return chunk as tensor
        recon = torch.cat(recon_planes)
    return recon


def chunk_image(sam, psf):
    return utils.fftconv2d(sam, psf, shape="same").sum(0)


def chunk_lost_power(psf, cropped_psf, dk):
    whole_power = m.compute_power(psf, dk)
    cropped_power = m.compute_power(cropped_psf, dk)
    lost_power = (whole_power - cropped_power).sum()
    return lost_power


def checkpoint(
    module_container,
    opt,
    loss_dict,
    it,
    fname="latest.pt",
    module_name="micdeconv",
):
    """
    Saves the current model/optimizer state.
    """
    checkpoint_dict = {
        f"{module_name}_state_dict": module_container.state_dict(),
        "opt_state_dict": opt.state_dict(),
        "it": it,
    }
    checkpoint_dict.update(loss_dict)
    torch.save(checkpoint_dict, fname)


def snapshot(
    module_container,
    opt,
    loss_dict,
    it,
    sam,
    recon,
    im,
    psf=None,
    phase_mask_angle=None,
    mirror_phase=None,
    save_dir="snapshots",
    module_name="micdeconv",
):
    """
    Saves the current model/optimizer state as well as the current sample,
    reconstruction, and image. Optionally saves the current PSF, phase mask,
    and/or mirror phase.
    """
    checkpoint(
        module_container,
        opt,
        loss_dict,
        it,
        fname=f"{save_dir}/state{it}.pt",
        module_name=module_name,
    )
    torch.save(sam.squeeze(), f"{save_dir}/sam{it}.pt")
    torch.save(recon.squeeze(), f"{save_dir}/recon{it}.pt")
    if im is not None:
        torch.save(im.squeeze(), f"{save_dir}/im{it}.pt")
    if psf is not None:
        torch.save(psf.detach().cpu(), f"{save_dir}/psf{it}.pt")
    if phase_mask_angle is not None:
        torch.save(phase_mask_angle.detach().cpu(), f"{save_dir}/phase_mask{it}.pt")
    if mirror_phase is not None:
        torch.save(mirror_phase.detach().cpu(), f"{save_dir}/mirror_phase{it}.pt")
    torch.cuda.empty_cache()
