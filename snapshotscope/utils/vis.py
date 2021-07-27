import datetime

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from .misc import _list


def plot_phase(
    phase,
    wrap_color=False,
    num_pixels=2560,
    pixel_size=0.325,
    NA=0.8,
    wavelength=0.532,
):
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(5, 5), dpi=300)
    ax.axis("off")
    fig.patch.set_facecolor("black")
    pm = phase
    fNA = NA / wavelength
    dk = 1 / (pixel_size * num_pixels)
    k = (
        np.linspace(
            -(num_pixels - 1) / 2, (num_pixels - 1) / 2, num=num_pixels, endpoint=True
        )
        * dk
    )
    kx, ky = np.meshgrid(k, k)
    pupil_mask = np.sqrt(kx ** 2 + ky ** 2) <= fNA

    nz = pm[pm != 0]
    if np.prod(nz.shape) == 0:
        nz = np.zeros_like(pm)
    if wrap_color:
        im = ax.imshow(
            pm % np.pi, interpolation="nearest", cmap="hsv", vmin=-np.pi, vmax=np.pi
        )
    else:
        im = ax.imshow(
            pm,
            interpolation="nearest",
            cmap="bwr",
            vmin=-np.abs(nz).max(),
            vmax=np.abs(nz).max(),
        )
    axins1 = inset_axes(ax, width="30%", height="3%", loc="lower right", borderpad=0)
    cb = plt.colorbar(im, cax=axins1, orientation="horizontal")
    cb.set_label("radians", color="w", labelpad=-35)
    plt.setp(plt.getp(cb.ax.axes, "xticklabels"), color="w")
    plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color="w")
    cb.ax.xaxis.get_offset_text().set_color("w")
    cb.outline.set_visible(False)
    imarray = im.make_image("agg", unsampled=True)[0]
    imarray[:, :, 3] = 255 * pupil_mask
    ax.cla()
    im2 = ax.imshow(imarray)
    ax.set_facecolor("black")
    ax.set_axis_off()
    plt.show()


def plot_psfs(
    psfs,
    gamma=1,
    xy_range=(-416, 416),
    z_range=(-125, 125),
    half_crop=False,
    slices=False,
    num_chunks=16,
    vmin=0,
    vmax_scale=1,
):
    xy_extent = (
        xy_range[0],
        xy_range[1],
        xy_range[1],
        xy_range[0],
    )
    xz_extent = (
        xy_range[0],
        xy_range[1],
        z_range[-1],
        z_range[0],
    )
    yz_extent = (
        z_range[-1],
        z_range[0],
        xy_range[1],
        xy_range[0],
    )

    num_xy_grid_squares = 10
    # calculate number of grid squares for z such that the volume plot is
    # isotropic for the given ranges in xy and z
    num_z_grid_squares = int(
        float(z_range[-1] - z_range[0])
        / float(xy_range[-1] - xy_range[0])
        * num_xy_grid_squares
    )
    num_grid_squares = num_xy_grid_squares + num_z_grid_squares

    with plt.style.context("dark_background"):
        fig = plt.figure(figsize=(5 * (len(psfs)), 5), dpi=300)
        gs = fig.add_gridspec(
            nrows=num_grid_squares, ncols=num_grid_squares * len(psfs)
        )

        for axidx, psf in enumerate(psfs):
            shape = psf.shape
            if half_crop:
                cropped_psf = psf[
                    :,
                    int(0.25 * shape[1]) : int(0.75 * shape[1]),
                    int(0.25 * shape[2]) : int(0.75 * shape[2]),
                ]
            else:
                cropped_psf = psf
            ax = fig.add_subplot(
                gs[
                    0:num_xy_grid_squares,
                    num_grid_squares * axidx : num_grid_squares * axidx
                    + num_xy_grid_squares,
                ]
            )
            vmin = 0
            vmax = (cropped_psf ** gamma).max() * vmax_scale
            im = ax.imshow(
                (cropped_psf ** gamma).max(axis=0),
                interpolation="nearest",
                aspect="auto",
                cmap="afmhot",
                extent=xy_extent,
                vmin=vmin,
                vmax=vmax,
            )
            ax.axis("off")
            ax = fig.add_subplot(
                gs[
                    num_xy_grid_squares:num_grid_squares,
                    num_grid_squares * axidx : num_grid_squares * axidx
                    + num_xy_grid_squares,
                ]
            )
            ax.axis("off")
            yrng = [int(cropped_psf.shape[1] * p) for p in [0.05, 0.95]]
            a = (cropped_psf ** gamma)[:, yrng[0] : yrng[1], :].max(axis=1).squeeze()
            im = ax.imshow(
                a,
                interpolation="nearest",
                cmap="afmhot",
                aspect="auto",
                extent=xz_extent,
                vmin=vmin,
                vmax=vmax,
            )
            ax = fig.add_subplot(
                gs[
                    0:num_xy_grid_squares,
                    num_grid_squares * axidx
                    + num_xy_grid_squares : num_grid_squares * axidx
                    + num_grid_squares,
                ]
            )
            ax.axis("off")
            xrng = [int(cropped_psf.shape[2] * p) for p in [0.05, 0.95]]
            a = np.fliplr(
                np.rot90(
                    (cropped_psf ** gamma)[:, :, xrng[0] : xrng[1]]
                    .max(axis=2)
                    .squeeze(),
                    k=3,
                )
            )
            im = ax.imshow(
                a,
                interpolation="nearest",
                cmap="afmhot",
                aspect="auto",
                extent=yz_extent,
                vmin=vmin,
                vmax=vmax,
            )
            ax = fig.add_subplot(
                gs[
                    num_xy_grid_squares:num_grid_squares,
                    num_grid_squares * axidx
                    + num_xy_grid_squares : num_grid_squares * axidx
                    + num_grid_squares,
                ]
            )
            ax.imshow(
                np.zeros((1, 1)), extent=xy_extent, vmin=0, vmax=100, cmap="afmhot"
            )
            ax.axis("off")
            if axidx == 0:
                scalebar = AnchoredSizeBar(
                    ax.transData,
                    200,
                    "",
                    "lower center",
                    pad=0.1,
                    color="white",
                    frameon=False,
                    size_vertical=1,
                )
                ax.add_artist(scalebar)
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.show()

        if slices:
            nrows, ncols = 1, len(psfs)
            psf_chunk_size = int(psfs[0].shape[0] / num_chunks)
            psf_sections = [i * psf_chunk_size for i in range(1, num_chunks)]
            psf_chunks = np.split(
                np.arange(psf.shape[0]), psf_sections
            )  # make sure we have loaded a psf already!
            for chunkidx in range(num_chunks):
                topslicef, topsliceaxes = plt.subplots(
                    nrows=nrows,
                    ncols=ncols,
                    figsize=(5 * len(psfs), nrows * 5),
                    dpi=300,
                )
                for psfidx, psf in enumerate(psfs):
                    shape = psf.shape
                    chunk = psf_chunks[chunkidx]
                    ax = topsliceaxes[psfidx]
                    if half_crop:
                        cropped_psf = psf[
                            int(chunk[0]) : int(chunk[-1]),
                            int(0.25 * shape[1]) : int(0.75 * shape[1]),
                            int(0.25 * shape[2]) : int(0.75 * shape[2]),
                        ]
                    else:
                        cropped_psf = psf
                    image = np.max(cropped_psf, axis=0)
                    ax.imshow(
                        image,
                        interpolation="nearest",
                        cmap="afmhot",
                        extent=xy_extent,
                        vmin=vmin,
                        vmax=vmax,
                    )
                    ax.axis("off")
                    plt.subplots_adjust(
                        top=1, bottom=0, right=1, left=0, hspace=0, wspace=0
                    )
                    plt.margins(0, 0)
                    scalebar = AnchoredSizeBar(
                        ax.transData,
                        50,
                        "",
                        "lower right",
                        pad=0.1,
                        color="white",
                        frameon=False,
                        size_vertical=1,
                    )
                    ax.add_artist(scalebar)
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0, 0)
            plt.show()


def plot_psf_im_pairs(
    psfs,
    ims,
    gamma=1,
    xy_range=(-416, 416),
    z_range=(-125, 125),
    half_crop=False,
    slices=False,
    num_chunks=16,
    vmin=0,
):
    assert len(ims) == len(psfs)

    xy_extent = (
        xy_range[0],
        xy_range[1],
        xy_range[1],
        xy_range[0],
    )
    xz_extent = (
        xy_range[0],
        xy_range[1],
        z_range[-1],
        z_range[0],
    )
    yz_extent = (
        z_range[-1],
        z_range[0],
        xy_range[1],
        xy_range[0],
    )

    with plt.style.context("dark_background"):
        fig = plt.figure(figsize=(8.5 * (len(psfs)), 5), dpi=300)
        gs = fig.add_gridspec(nrows=15, ncols=25 * len(psfs))

        for axidx, (psf, image) in enumerate(zip(psfs, ims)):
            shape = psf.shape
            if half_crop:
                cropped_psf = psf[
                    :,
                    int(0.25 * shape[1]) : int(0.75 * shape[1]),
                    int(0.25 * shape[2]) : int(0.75 * shape[2]),
                ]
            else:
                cropped_psf = psf
            ax = fig.add_subplot(gs[0:10, 25 * axidx : 25 * axidx + 10])
            vmin = 0
            vmax = (cropped_psf ** gamma).max()
            im = ax.imshow(
                (cropped_psf ** gamma).max(axis=0),
                interpolation="nearest",
                aspect="auto",
                cmap="afmhot",
                extent=xy_extent,
                vmin=vmin,
                vmax=vmax,
            )
            ax.axis("off")
            ax = fig.add_subplot(gs[10:15, 25 * axidx : 25 * axidx + 10])
            ax.axis("off")
            yrng = [int(cropped_psf.shape[1] * p) for p in [0.05, 0.95]]
            a = (cropped_psf ** gamma)[:, yrng[0] : yrng[1], :].max(axis=1).squeeze()
            im = ax.imshow(
                a,
                interpolation="nearest",
                cmap="afmhot",
                aspect="auto",
                extent=xz_extent,
                vmin=vmin,
                vmax=vmax,
            )
            ax = fig.add_subplot(gs[0:10, 25 * axidx + 10 : 25 * axidx + 15])
            ax.axis("off")
            xrng = [int(cropped_psf.shape[2] * p) for p in [0.05, 0.95]]
            a = np.rot90(
                (cropped_psf ** gamma)[:, :, xrng[0] : xrng[1]].max(axis=2).squeeze(),
                k=3,
            )
            im = ax.imshow(
                a,
                interpolation="nearest",
                cmap="afmhot",
                aspect="auto",
                extent=yz_extent,
                vmin=vmin,
                vmax=vmax,
            )
            ax = fig.add_subplot(gs[0:10, 25 * axidx + 15 : 25 * axidx + 25])
            ax.axis("off")
            im = ax.imshow(
                image,
                interpolation="nearest",
                cmap="cividis",
                aspect="auto",
                extent=xy_extent,
            )
            ax = fig.add_subplot(gs[10:15, 25 * axidx + 15 : 25 * axidx + 25])
            ax.imshow(
                np.zeros([1, 1]), cmap="afmhot", vmin=0, vmax=100, extent=xz_extent
            )
            ax.axis("off")
            scalebar = AnchoredSizeBar(
                ax.transData,
                50,
                "",
                "upper left",
                pad=0.1,
                color="white",
                frameon=False,
                size_vertical=1,
            )
            ax.add_artist(scalebar)
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.show()

        if slices:
            nrows, ncols = 1, len(psfs)
            psf_chunk_size = int(psfs[0].shape[0] / num_chunks)
            psf_sections = [i * psf_chunk_size for i in range(1, num_chunks)]
            psf_chunks = np.split(
                np.arange(psf.shape[0]), psf_sections
            )  # make sure we have loaded a psf already!
            for chunkidx in range(num_chunks):
                topslicef, topsliceaxes = plt.subplots(
                    nrows=nrows,
                    ncols=ncols,
                    figsize=(5 * len(psfs), nrows * 5),
                    dpi=300,
                )
                for psfidx, psf in enumerate(psfs):
                    shape = psf.shape
                    chunk = psf_chunks[chunkidx]
                    ax = topsliceaxes[psfidx]
                    if half_crop:
                        cropped_psf = psf[
                            int(chunk[0]) : int(chunk[-1]),
                            int(0.25 * shape[1]) : int(0.75 * shape[1]),
                            int(0.25 * shape[2]) : int(0.75 * shape[2]),
                        ]
                    else:
                        cropped_psf = psf
                    image = np.max(cropped_psf, axis=0)
                    ax.imshow(
                        image,
                        interpolation="nearest",
                        cmap="afmhot",
                        extent=xy_extent,
                        vmin=vmin,
                        vmax=vmax,
                    )
                    ax.axis("off")
                    plt.subplots_adjust(
                        top=1, bottom=0, right=1, left=0, hspace=0, wspace=0
                    )
                    plt.margins(0, 0)
                    scalebar = AnchoredSizeBar(
                        ax.transData,
                        50,
                        "",
                        "lower right",
                        pad=0.1,
                        color="white",
                        frameon=False,
                        size_vertical=1,
                    )
                    ax.add_artist(scalebar)
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0, 0)
            plt.show()


def plot_slices(
    vols,
    labels=None,
    x_range=(-416, 416),
    y_range=(-416, 416),
    z_range=(-125, 125),
    half_crop_horizontal=False,
    half_crop=False,
    num_chunks=16,
    svmin=0,
    svmax=None,
    cmap="afmhot",
    show_axes=False,
    show_scalebar=True,
    style="dark_background",
):
    svmin = _list(svmin)
    svmax = _list(svmax)

    xy_extent = (
        x_range[0],
        x_range[-1],
        y_range[-1],
        y_range[0],
    )
    xz_extent = (
        x_range[0],
        x_range[-1],
        z_range[-1],
        z_range[0],
    )

    num_x_grid_squares = 10
    num_y_grid_squares = int(
        float(y_range[-1] - y_range[0])
        / float(x_range[-1] - x_range[0])
        * num_x_grid_squares
    )
    # calculate number of grid squares for z such that the volume plot is
    # isotropic for the given ranges in xy and z
    num_z_grid_squares = int(
        float(z_range[-1] - z_range[0])
        / float(x_range[-1] - x_range[0])
        * num_x_grid_squares
    )
    width = 5 * len(vols)
    height = int(
        float(y_range[-1] - y_range[0])
        / float(x_range[-1] - x_range[0])
        * 5
        * num_chunks
    )

    with plt.style.context(style):
        vol_chunk_size = int(vols[0].shape[0] / num_chunks)
        vol_sections = [i * vol_chunk_size for i in range(1, num_chunks)]
        vol_chunks = np.split(
            np.arange(vols[0].shape[0]), vol_sections
        )  # make sure we have loaded a vol already!
        topslicef = plt.figure(
            figsize=(width, height),
            dpi=300,
        )
        gs = topslicef.add_gridspec(
            nrows=num_y_grid_squares * num_chunks,
            ncols=num_x_grid_squares * len(vols),
        )
        for chunkidx in range(num_chunks):
            for volidx, vol in enumerate(vols):
                shape = vol.shape
                chunk = vol_chunks[chunkidx]
                ax = topslicef.add_subplot(
                    gs[
                        num_y_grid_squares
                        * chunkidx : num_y_grid_squares
                        * (chunkidx + 1),
                        num_x_grid_squares * volidx : num_x_grid_squares * (volidx + 1),
                    ]
                )
                if half_crop:
                    cropped_vol = vol[
                        int(chunk[0]) : int(chunk[-1]),
                        int(0.25 * shape[1]) : int(0.75 * shape[1]),
                        int(0.25 * shape[2]) : int(0.75 * shape[2]),
                    ]
                elif half_crop_horizontal:
                    cropped_vol = vol[
                        int(chunk[0]) : int(chunk[-1]),
                        :,
                        int(0.25 * shape[2]) : int(0.75 * shape[2]),
                    ]
                else:
                    cropped_vol = vol[int(chunk[0]) : int(chunk[-1])]
                image = np.max(cropped_vol, axis=0)
                ax.imshow(
                    image,
                    interpolation="nearest",
                    aspect="auto",
                    cmap=cmap,
                    extent=xy_extent,
                    vmin=svmin[volidx % len(svmin)],
                    vmax=svmax[volidx % len(svmax)],
                )
                if show_axes:
                    ax.set_xticks([])
                    ax.set_yticks([])
                else:
                    ax.axis("off")
                if chunkidx == 0 and volidx == 0 and show_scalebar:
                    scalebar = AnchoredSizeBar(
                        ax.transData,
                        50,
                        "",
                        "lower right",
                        pad=0.1,
                        color="white",
                        frameon=False,
                        size_vertical=1,
                    )
                    ax.add_artist(scalebar)
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.show()


def plot_vols_without_side(
    vols,
    labels=None,
    x_range=(-416, 416),
    y_range=(-416, 416),
    z_range=(-125, 125),
    radius=386 / 2.0,
    half_crop_horizontal=False,
    half_crop=False,
    slices=False,
    num_chunks=16,
    vmin=0,
    vmax=None,
    svmin=0,
    svmax=None,
    cmap="afmhot",
    show_axes=False,
    style="dark_background",
):
    vmin = _list(vmin)
    vmax = _list(vmax)
    svmin = _list(svmin)
    svmax = _list(svmax)

    xy_extent = (
        x_range[0],
        x_range[-1],
        y_range[-1],
        y_range[0],
    )
    xz_extent = (
        x_range[0],
        x_range[-1],
        z_range[-1],
        z_range[0],
    )

    num_x_grid_squares = 10
    num_y_grid_squares = int(
        float(y_range[-1] - y_range[0])
        / float(x_range[-1] - x_range[0])
        * num_x_grid_squares
    )
    # calculate number of grid squares for z such that the volume plot is
    # isotropic for the given ranges in xy and z
    num_z_grid_squares = int(
        float(z_range[-1] - z_range[0])
        / float(x_range[-1] - x_range[0])
        * num_x_grid_squares
    )
    width = 5 * len(vols)
    height = int(
        float(z_range[-1] - z_range[0]) / float(x_range[-1] - x_range[0]) * 5
    ) + int(float(y_range[-1] - y_range[0]) / float(x_range[-1] - x_range[0]) * 5)

    with plt.style.context(style):
        fig = plt.figure(figsize=(width, height), dpi=300)
        gs = fig.add_gridspec(
            nrows=num_y_grid_squares + num_z_grid_squares,
            ncols=num_x_grid_squares * len(vols),
        )
        for axidx, vol in enumerate(vols):
            shape = vol.shape
            if half_crop:
                cropped_vol = vol[
                    :,
                    int(0.25 * shape[1]) : int(0.75 * shape[1]),
                    int(0.25 * shape[2]) : int(0.75 * shape[2]),
                ]
            elif half_crop_horizontal:
                cropped_vol = vol[
                    :,
                    :,
                    int(0.25 * shape[2]) : int(0.75 * shape[2]),
                ]
            else:
                cropped_vol = vol
            ax = fig.add_subplot(
                gs[
                    0:num_y_grid_squares,
                    num_x_grid_squares * axidx : num_x_grid_squares * (axidx + 1),
                ]
            )
            if show_axes:
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                ax.axis("off")
            im = ax.imshow(
                cropped_vol.max(axis=0),
                interpolation="nearest",
                aspect="auto",
                cmap=cmap,
                extent=xy_extent,
                vmin=vmin[axidx % len(vmin)],
                vmax=vmax[axidx % len(vmax)],
            )
            if labels is not None:
                ax.set_title(labels[axidx % len(labels)])
            ax = fig.add_subplot(
                gs[
                    num_y_grid_squares : num_y_grid_squares + num_z_grid_squares,
                    num_x_grid_squares * axidx : num_x_grid_squares * (axidx + 1),
                ]
            )
            if show_axes:
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                ax.axis("off")
            a = cropped_vol.max(axis=1).squeeze()
            im = ax.imshow(
                a,
                interpolation="nearest",
                cmap=cmap,
                aspect="auto",
                extent=xz_extent,
                vmin=vmin[axidx % len(vmin)],
                vmax=vmax[axidx % len(vmax)],
            )
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.show()

        if slices:
            nrows, ncols = 1, len(vols)
            vol_chunk_size = int(vols[0].shape[0] / num_chunks)
            vol_sections = [i * vol_chunk_size for i in range(1, num_chunks)]
            vol_chunks = np.split(
                np.arange(vol.shape[0]), vol_sections
            )  # make sure we have loaded a vol already!
            for chunkidx in range(num_chunks):
                topslicef, topsliceaxes = plt.subplots(
                    nrows=nrows,
                    ncols=ncols,
                    figsize=(5 * len(vols), nrows * 5),
                    dpi=300,
                )
                for volidx, vol in enumerate(vols):
                    shape = vol.shape
                    chunk = vol_chunks[chunkidx]
                    ax = topsliceaxes[volidx]
                    if half_crop:
                        cropped_vol = vol[
                            int(chunk[0]) : int(chunk[-1]),
                            int(0.25 * shape[1]) : int(0.75 * shape[1]),
                            int(0.25 * shape[2]) : int(0.75 * shape[2]),
                        ]
                    else:
                        cropped_vol = vol[int(chunk[0]) : int(chunk[-1])]
                    image = np.max(cropped_vol, axis=0)
                    ax.imshow(
                        image,
                        interpolation="nearest",
                        cmap=cmap,
                        extent=xy_extent,
                        vmin=svmin[volidx % len(svmin)],
                        vmax=svmax[volidx % len(svmax)],
                    )
                    if show_axes:
                        ax.set_xticks([])
                        ax.set_yticks([])
                    else:
                        ax.axis("off")
                    plt.subplots_adjust(
                        top=1, bottom=0, right=1, left=0, hspace=0, wspace=0
                    )
                    plt.margins(0, 0)
                    if volidx == 0:
                        scalebar = AnchoredSizeBar(
                            ax.transData,
                            50,
                            "",
                            "lower right",
                            pad=0.1,
                            color="white",
                            frameon=False,
                            size_vertical=1,
                        )
                        ax.add_artist(scalebar)
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0, 0)
            plt.show()


def plot_vols(
    vols,
    labels=None,
    xy_range=(-416, 416),
    z_range=(-125, 125),
    radius=386 / 2.0,
    half_crop=False,
    slices=False,
    num_chunks=16,
    vmin=0,
    vmax=None,
    svmin=0,
    svmax=None,
    cmap="afmhot",
    show_axes=False,
    style="dark_background",
):
    vmin = _list(vmin)
    vmax = _list(vmax)
    svmin = _list(svmin)
    svmax = _list(svmax)

    xy_extent = (
        xy_range[0],
        xy_range[1],
        xy_range[1],
        xy_range[0],
    )
    xz_extent = (
        xy_range[0],
        xy_range[1],
        z_range[-1],
        z_range[0],
    )
    yz_extent = (
        z_range[-1],
        z_range[0],
        xy_range[1],
        xy_range[0],
    )

    num_xy_grid_squares = 10
    # calculate number of grid squares for z such that the volume plot is
    # isotropic for the given ranges in xy and z
    num_z_grid_squares = int(
        float(z_range[-1] - z_range[0])
        / float(xy_range[-1] - xy_range[0])
        * num_xy_grid_squares
    )
    num_grid_squares = num_xy_grid_squares + num_z_grid_squares

    with plt.style.context(style):
        fig = plt.figure(figsize=(5 * (len(vols)), 5), dpi=300)
        gs = fig.add_gridspec(
            nrows=num_grid_squares, ncols=num_grid_squares * len(vols)
        )
        for axidx, vol in enumerate(vols):
            shape = vol.shape
            if half_crop:
                cropped_vol = vol[
                    :,
                    int(0.25 * shape[1]) : int(0.75 * shape[1]),
                    int(0.25 * shape[2]) : int(0.75 * shape[2]),
                ]
            else:
                cropped_vol = vol
            ax = fig.add_subplot(
                gs[
                    0:num_xy_grid_squares,
                    num_grid_squares * axidx : num_grid_squares * axidx
                    + num_xy_grid_squares,
                ]
            )
            if show_axes:
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                ax.axis("off")
            im = ax.imshow(
                cropped_vol.max(axis=0),
                interpolation="nearest",
                aspect="auto",
                cmap=cmap,
                extent=xy_extent,
                vmin=vmin[axidx % len(vmin)],
                vmax=vmax[axidx % len(vmax)],
            )
            if labels is not None:
                ax.set_title(labels[axidx % len(labels)])
            ax = fig.add_subplot(
                gs[
                    num_xy_grid_squares:num_grid_squares,
                    num_grid_squares * axidx : num_grid_squares * axidx
                    + num_xy_grid_squares,
                ]
            )
            if show_axes:
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                ax.axis("off")
            yrng = [int(cropped_vol.shape[1] * p) for p in [0.05, 0.95]]
            a = cropped_vol[:, yrng[0] : yrng[1], :].max(axis=1).squeeze()
            im = ax.imshow(
                a,
                interpolation="nearest",
                cmap=cmap,
                aspect="auto",
                extent=xz_extent,
                vmin=vmin[axidx % len(vmin)],
                vmax=vmax[axidx % len(vmax)],
            )
            ax = fig.add_subplot(
                gs[
                    0:num_xy_grid_squares,
                    num_grid_squares * axidx
                    + num_xy_grid_squares : num_grid_squares * axidx
                    + num_grid_squares,
                ]
            )
            if show_axes:
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                ax.axis("off")
            xrng = [int(cropped_vol.shape[2] * p) for p in [0.05, 0.95]]
            a = np.fliplr(
                np.rot90(
                    cropped_vol[:, :, xrng[0] : xrng[1]].max(axis=2).squeeze(),
                    k=3,
                )
            )
            im = ax.imshow(
                a,
                interpolation="nearest",
                cmap=cmap,
                aspect="auto",
                extent=yz_extent,
                vmin=vmin[axidx % len(vmin)],
                vmax=vmax[axidx % len(vmax)],
            )
            ax = fig.add_subplot(
                gs[
                    num_xy_grid_squares:num_grid_squares,
                    num_grid_squares * axidx
                    + num_xy_grid_squares : num_grid_squares * axidx
                    + num_grid_squares,
                ]
            )
            ax.imshow(np.zeros((1, 1)), extent=xy_extent, vmin=0, vmax=100, cmap=cmap)
            ax.axis("off")
            if axidx == 0:
                scalebar = AnchoredSizeBar(
                    ax.transData,
                    50,
                    "",
                    "lower left",
                    pad=0.1,
                    color="white",
                    frameon=False,
                    size_vertical=1,
                )
                ax.add_artist(scalebar)
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.show()

        if slices:
            nrows, ncols = 1, len(vols)
            vol_chunk_size = int(vols[0].shape[0] / num_chunks)
            vol_sections = [i * vol_chunk_size for i in range(1, num_chunks)]
            vol_chunks = np.split(
                np.arange(vol.shape[0]), vol_sections
            )  # make sure we have loaded a vol already!
            for chunkidx in range(num_chunks):
                topslicef, topsliceaxes = plt.subplots(
                    nrows=nrows,
                    ncols=ncols,
                    figsize=(5 * len(vols), nrows * 5),
                    dpi=300,
                )
                for volidx, vol in enumerate(vols):
                    shape = vol.shape
                    chunk = vol_chunks[chunkidx]
                    ax = topsliceaxes[volidx]
                    if half_crop:
                        cropped_vol = vol[
                            int(chunk[0]) : int(chunk[-1]),
                            int(0.25 * shape[1]) : int(0.75 * shape[1]),
                            int(0.25 * shape[2]) : int(0.75 * shape[2]),
                        ]
                    else:
                        cropped_vol = vol[int(chunk[0]) : int(chunk[-1])]
                    image = np.max(cropped_vol, axis=0)
                    ax.imshow(
                        image,
                        interpolation="nearest",
                        cmap=cmap,
                        extent=xy_extent,
                        vmin=svmin[volidx % len(svmin)],
                        vmax=svmax[volidx % len(svmax)],
                    )
                    if show_axes:
                        ax.set_xticks([])
                        ax.set_yticks([])
                    else:
                        ax.axis("off")
                    plt.subplots_adjust(
                        top=1, bottom=0, right=1, left=0, hspace=0, wspace=0
                    )
                    plt.margins(0, 0)
                    if volidx == 0:
                        scalebar = AnchoredSizeBar(
                            ax.transData,
                            50,
                            "",
                            "lower right",
                            pad=0.1,
                            color="white",
                            frameon=False,
                            size_vertical=1,
                        )
                        ax.add_artist(scalebar)
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0, 0)
            plt.show()


def plot_ims(ims, xy_range=(-416, 416), draw_scalebar=True, cmap="cividis"):
    # create spatial extents
    xy_extent = (
        xy_range[0],
        xy_range[1],
        xy_range[1],
        xy_range[0],
    )
    with plt.style.context("dark_background"):
        fig = plt.figure(figsize=(5 * len(ims), 5), constrained_layout=False, dpi=300)
        for imidx, im in enumerate(ims):
            gs = fig.add_gridspec(nrows=5, ncols=5 * len(ims))
            ax = fig.add_subplot(gs[:, 5 * (imidx) : 5 * (imidx + 1)])
            implt = ax.imshow(im, interpolation="nearest", cmap=cmap, extent=xy_extent)
            ax.axis("off")
            if imidx == 0 and draw_scalebar:
                scalebar = AnchoredSizeBar(
                    ax.transData,
                    200,
                    "",
                    "lower right",
                    pad=0.1,
                    color="white",
                    frameon=False,
                    size_vertical=1,
                )
                ax.add_artist(scalebar)
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.show()
