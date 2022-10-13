import os, sys, faulthandler

faulthandler.enable()

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import snapshotscope.microscope as m
import snapshotscope.networks.deconv as d
import snapshotscope.data.augment as augment
import snapshotscope.data.dataloaders as data
import snapshotscope.optical_elements.phase_masks as pm
import snapshotscope.networks.control as control

import logging

logging.basicConfig(filename="out.log", level=logging.DEBUG, format="%(message)s")

learning_rate = 1e-4
pm_learning_rate = 1e2 * learning_rate
regularize_lost_power = 0
num_iterations = 1000000

# setup the simulation parameters, make a microscope
num_pixels = 1280
pixel_size = 0.325
num_scopes = 8
num_planes = 192
downsample = 5
plane_subsample = 2
psf_pad = 640
taper_width = 5
regularize_power = False
wavelength = 0.532
pupil_NA = 0.8
devices = [torch.device(f"cuda:{i}") for i in range(num_scopes)]


# calculate downsampled sizes
subsampled_num_planes = int(num_planes / plane_subsample)
downsampled_num_pixels = int(num_pixels / downsample)
downsampled_radius = int(0.5 * 193 / pixel_size / downsample)

# set grad sizes
num_grad_im_planes = int(subsampled_num_planes / num_scopes)
num_grad_recon_planes = int(subsampled_num_planes / num_scopes)

# create chunked defocus ranges
defocus_range = np.round(np.linspace(-100, 100, num=subsampled_num_planes))
chunk_size = int(len(defocus_range) / num_scopes)
if len(defocus_range) % num_scopes != 0:
    chunk_size += 1
sections = [i * chunk_size for i in range(1, num_scopes)]
defocus_ranges = np.split(defocus_range, sections)

# create microscope param dicts
param_dict = dict(
    wavelength=[0.532],
    ratios=[1.0],
    NA=0.8,
    n_immersion=1.33,
    pixel_size=pixel_size,
    num_pixels=num_pixels,
    pad=psf_pad,
    taper_width=taper_width,
    downsample=downsample,
)
param_dicts = [
    dict(list(param_dict.items()) + [("device", device)]) for device in devices
]

# create augmentations
augmentations = augment.create_augmentations(
    [
        {
            "name": "adjust_brightness",
            "args": {
                "scale": 52.8,
                "background": (0, 60),
                "brightness": (0.33, 3.0),
                "log": False,
            },
        },
        {"name": "rotate_pitch", "args": {"angle": (-5, 5)}},
        {"name": "rotate_roll", "args": {"angle": (-10, 10)}},
        {"name": "rotate_yaw", "args": {"angle": (-20, 20)}},
        {
            "name": "horizontal_jitter",
            "args": {"jitter_amount": (0, 60), "direction": ("left", "right")},
        },
        {
            "name": "flip_planes",
            "args": {"allowed_flips": [0, 2], "num_flips": "random"},
        },
        {"name": "flip_volume", "args": {"probability": 0.5}},
        {"name": "reflect_planes", "args": {"probability": 0.5}},
        {
            "name": "cylinder_crop",
            "args": {
                "radius": downsampled_radius,
                "center": "center",
                "max_depth": subsampled_num_planes,
            },
        },
        {
            "name": "pad_volume",
            "args": {
                "target_shape": (
                    subsampled_num_planes,
                    downsampled_num_pixels,
                    downsampled_num_pixels,
                )
            },
        },
    ]
)
augmentations = augment.compose_augmentations(augmentations)

test_augmentations = augment.create_augmentations(
    [
        {
            "name": "adjust_brightness",
            "args": {"scale": 52.8, "background": 30, "brightness": 1, "log": False},
        },
        {
            "name": "pad_volume",
            "args": {
                "target_shape": (
                    subsampled_num_planes,
                    downsampled_num_pixels * 4,
                    downsampled_num_pixels * 2,
                )
            },
        },
    ]
)
test_augmentations = augment.compose_augmentations(test_augmentations)

test_train_augmentations = augment.create_augmentations(
    [
        {
            "name": "adjust_brightness",
            "args": {"scale": 52.8, "background": 30, "brightness": 1, "log": False},
        },
        {
            "name": "cylinder_crop",
            "args": {
                "radius": downsampled_radius,
                "center": "center",
                "max_depth": subsampled_num_planes,
            },
        },
        {
            "name": "pad_volume",
            "args": {
                "target_shape": (
                    subsampled_num_planes,
                    downsampled_num_pixels,
                    downsampled_num_pixels,
                )
            },
        },
    ]
)
test_train_augmentations = augment.compose_augmentations(test_train_augmentations)


def create_microscopes():
    # create single gpu microscope
    mics = [m.PupilFunction4FMicroscope(**param_dict) for param_dict in param_dicts]
    mics = nn.ModuleList(mics)
    return mics


def create_phase_mask(kx, ky, phase_mask_init=None):
    if phase_mask_init is None:
        # create defocused ramps
        defocused_ramps = pm.DefocusedRamps(
            kx, ky, pupil_NA / wavelength, 1.33, wavelength, delta=1187.0
        )
        phase_mask_init = defocused_ramps()
    # create unconstrained pixels from ramps
    pixels = pm.Pixels(kx, ky, pixels=phase_mask_init)
    return pixels


def create_reconstruction_networks():
    # create multi gpu reconstruction network list for 1 gpu
    deconvs = [
        d.UNet3D(
            1,
            1,
            int(subsampled_num_planes / num_scopes),
            conv_kernel_size=(11, 7, 7),
            conv_padding=(5, 3, 3),
            pool_kernel_size=(1, 2, 2),
            f_maps=tuple(5 for level in range(4)),
            num_levels=4,
            input_scaling_mode="scaling",
            quantile_scale=1e-4,
            device=device,
        )
        for device in devices
    ]
    deconvs = nn.ModuleList(deconvs)
    return deconvs


def initialize_microscope_reconstruction(latest=None, phase_mask_init=None):
    mics = create_microscopes()
    pixels = create_phase_mask(
        mics[0].kx.cpu(), mics[0].ky.cpu(), phase_mask_init=phase_mask_init
    )
    deconvs = create_reconstruction_networks()
    micdeconv = nn.ModuleDict({"mics": mics, "deconvs": deconvs, "phase_mask": pixels})
    if latest is not None:
        print("[info] loading from checkpoint")
        micdeconv.load_state_dict(latest["micdeconv_state_dict"], strict=True)
    return micdeconv


def initialize_optimizer(micdeconv, latest=None):
    # optimize microscope parameters and reconstruction network
    opt = optim.Adam(
        [{"params": micdeconv["deconvs"].parameters(), "lr": learning_rate}],
        lr=learning_rate,
    )
    if latest is not None:
        opt.load_state_dict(latest["opt_state_dict"])
    return opt


def create_dataloader(test=False):
    base_path = "../../../../data/interpolated_1.625um/"
    if not test:
        dataset = data.ConfocalVolumesDataset(
            [
                os.path.join(base_path, "zjabc_train/2019-12-10-6dpf-Huc-H2B-jRGECO"),
                os.path.join(base_path, "zjabc_train/2019-12-16-5dpf-Huc-H2B-G7FF"),
                os.path.join(base_path, "zjabc_train/2019-12-17-6dpf-Huc-H2B-G7FF"),
                os.path.join(
                    base_path,
                    "zjabc_train/2020-02-25_abc-Elavl3-H2B-GCaMP+Cytosolic-GCaMP",
                ),
                os.path.join(base_path, "zjabc_train/2020-02-27_abc-Elavl3-H2B-GCaMP"),
                os.path.join(
                    base_path, "zjabc_train/2020-02-28_abc-Elval3-H2B-GCaMP+jRGECO"
                ),
                os.path.join(
                    base_path, "zjabc_train/2020-03-01-5dpf-elavl3-H2B-jRGECO"
                ),
                os.path.join(
                    base_path, "zjabc_train/2020-03-02-6dpf-elavl3-H2B-jRGECO"
                ),
                os.path.join(
                    base_path, "zjabc_train/2020-03-03_abc-Elavl3-H2B-GC7f_gfapjRGECO1b"
                ),
                os.path.join(base_path, "zjabc_train/2020-03-03-Elavl3-H2B-GCaMP"),
                os.path.join(
                    base_path, "zjabc_train/2020-03-15-_Elavl3-H2B-GC7ff_gfapjRGECO1b"
                ),
                os.path.join(
                    base_path, "zjabc_train/2020-03-17-Elavl3-H2B-GC7f_gfapjRGECO1b"
                ),
                os.path.join(
                    base_path, "zjabc_train/2020-03-18-Elavl3-H2B-GC7f_gfapjRGECO1b"
                ),
            ],
            shape=(num_planes, downsampled_num_pixels, downsampled_num_pixels),
            location="random",
            strategy="center",
            bounds_roi="data",
            valid_ratio=(0.5, 1.0, 0.1),
            steps=(plane_subsample, 1, 1),
            augmentations=augmentations,
            balance=True,
        )
        dataloader = torch.utils.data.DataLoader(dataset, num_workers=5, shuffle=True)
    else:
        dataset = data.ConfocalVolumesDataset(
            [
                os.path.join(
                    base_path, "zjabc_test/2019-02-24_abc-Elavl3-H2B-GC7f_gfapjRGECO1b"
                ),
                os.path.join(base_path, "zjabc_test/2020-03-04-5dpf-elva3-H2B-jRGECO"),
            ],
            shape=(num_planes, downsampled_num_pixels * 4, downsampled_num_pixels * 2),
            location="center",
            strategy="center",
            bounds_roi="data",
            steps=(plane_subsample, 1, 1),
            augmentations=test_augmentations,
            balance=False,
        )
        dataloader = torch.utils.data.DataLoader(dataset, num_workers=5, shuffle=False)
    return dataloader


def create_test_train_dataloader():
    base_path = "../../../../data/interpolated_1.625um/"
    dataset = data.ConfocalVolumesDataset(
        [
            os.path.join(base_path, "zjabc_train/2019-12-10-6dpf-Huc-H2B-jRGECO"),
            os.path.join(base_path, "zjabc_train/2019-12-16-5dpf-Huc-H2B-G7FF"),
            os.path.join(base_path, "zjabc_train/2019-12-17-6dpf-Huc-H2B-G7FF"),
            os.path.join(
                base_path, "zjabc_train/2020-02-25_abc-Elavl3-H2B-GCaMP+Cytosolic-GCaMP"
            ),
            os.path.join(base_path, "zjabc_train/2020-02-27_abc-Elavl3-H2B-GCaMP"),
            os.path.join(
                base_path, "zjabc_train/2020-02-28_abc-Elval3-H2B-GCaMP+jRGECO"
            ),
            os.path.join(base_path, "zjabc_train/2020-03-01-5dpf-elavl3-H2B-jRGECO"),
            os.path.join(base_path, "zjabc_train/2020-03-02-6dpf-elavl3-H2B-jRGECO"),
            os.path.join(
                base_path, "zjabc_train/2020-03-03_abc-Elavl3-H2B-GC7f_gfapjRGECO1b"
            ),
            os.path.join(base_path, "zjabc_train/2020-03-03-Elavl3-H2B-GCaMP"),
            os.path.join(
                base_path, "zjabc_train/2020-03-15-_Elavl3-H2B-GC7ff_gfapjRGECO1b"
            ),
            os.path.join(
                base_path, "zjabc_train/2020-03-17-Elavl3-H2B-GC7f_gfapjRGECO1b"
            ),
            os.path.join(
                base_path, "zjabc_train/2020-03-18-Elavl3-H2B-GC7f_gfapjRGECO1b"
            ),
        ],
        shape=(num_planes, downsampled_num_pixels, downsampled_num_pixels),
        location="random",
        strategy="center",
        bounds_roi="data",
        valid_ratio=(0.5, 1.0, 0.1),
        steps=(plane_subsample, 1, 1),
        augmentations=test_train_augmentations,
        balance=False,
    )
    dataloader = torch.utils.data.DataLoader(dataset, num_workers=5, shuffle=True)
    return dataloader


def create_test_extents():
    # define extents as list of lists of tuples of tuples
    # len(extents) = number of samples in dataloader
    # each item in extents is a list of locations in the sample
    # each location is a tuple of dimensions
    # each dimension is a tuple containing a start and end index
    length = downsampled_radius * 2
    extents = [
        [
            (
                (0, subsampled_num_planes),
                (280, 280 + length),
                (120 + 64, 120 + 64 + length),
            ),
            (
                (0, subsampled_num_planes),
                (600, 600 + length),
                (138 + 64, 138 + 64 + length),
            ),
        ],
        [
            (
                (0, subsampled_num_planes),
                (280, 280 + length),
                (138 + 64, 138 + 64 + length),
            ),
            (
                (0, subsampled_num_planes),
                (550, 550 + length),
                (138 + 64, 138 + 64 + length),
            ),
        ],
        [
            (
                (0, subsampled_num_planes),
                (190, 190 + length),
                (138 + 64, 138 + 64 + length),
            ),
            (
                (0, subsampled_num_planes),
                (400, 400 + length),
                (138 + 64, 138 + 64 + length),
            ),
        ],
        [
            (
                (0, subsampled_num_planes),
                (200, 200 + length),
                (138 + 64, 138 + 64 + length),
            ),
            (
                (0, subsampled_num_planes),
                (500, 500 + length),
                (150 + 64, 150 + 64 + length),
            ),
        ],
        [
            (
                (0, subsampled_num_planes),
                (155, 155 + length),
                (120 + 64, 120 + 64 + length),
            ),
            (
                (0, subsampled_num_planes),
                (400, 400 + length),
                (150 + 64, 150 + 64 + length),
            ),
        ],
        [
            (
                (0, subsampled_num_planes),
                (190, 190 + length),
                (138 + 64, 138 + 64 + length),
            ),
            (
                (0, subsampled_num_planes),
                (460, 460 + length),
                (138 + 64, 138 + 64 + length),
            ),
        ],
        [
            (
                (0, subsampled_num_planes),
                (230, 230 + length),
                (138 + 64, 138 + 64 + length),
            ),
            (
                (0, subsampled_num_planes),
                (520, 520 + length),
                (138 + 64, 138 + 64 + length),
            ),
        ],
        [
            (
                (0, subsampled_num_planes),
                (230, 230 + length),
                (138 + 64, 138 + 64 + length),
            ),
            (
                (0, subsampled_num_planes),
                (520, 520 + length),
                (138 + 64, 138 + 64 + length),
            ),
        ],
        [
            (
                (0, subsampled_num_planes),
                (190, 190 + length),
                (138 + 64, 138 + 64 + length),
            ),
            (
                (0, subsampled_num_planes),
                (460, 460 + length),
                (138 + 64, 138 + 64 + length),
            ),
        ],
        [
            (
                (0, subsampled_num_planes),
                (240, 240 + length),
                (120 + 64, 120 + 64 + length),
            ),
            (
                (0, subsampled_num_planes),
                (460, 460 + length),
                (138 + 64, 138 + 64 + length),
            ),
        ],
    ]
    return extents


def create_train_extents(num=1):
    # define extents as list of lists of tuples of tuples
    # len(extents) = number of samples in dataloader
    # each item in extents is a list of locations in the sample
    # each location is a tuple of dimensions
    # each dimension is a tuple containing a start and end index
    extents = [
        [
            (
                (0, subsampled_num_planes),
                (0, downsampled_num_pixels),
                (0, downsampled_num_pixels),
            )
        ]
    ]
    extents = extents * num
    return extents


def train():
    # initialize model for training
    if os.path.exists("latest.pt"):
        latest = torch.load("latest.pt")
    else:
        latest = None
    # initialize phase mask from pretrained
    phase_mask_init = torch.load(
        "../psf_optimization/snapshots/phase_mask999999.pt", map_location="cpu"
    )
    micdeconv = initialize_microscope_reconstruction(
        latest=latest, phase_mask_init=phase_mask_init
    )
    print(micdeconv)

    # initialize optimizer
    opt = initialize_optimizer(micdeconv, latest=latest)

    # initialize data
    dataloader = create_dataloader()
    val_dataloader = create_dataloader(test=True)

    # initialize test extents
    extents = create_test_extents()

    # define logging message
    log_string = "[{}] iter: {}, loss: {}, norm(phase mask): {}"
    profile_string = "[{}] {}"

    # initialize iteration count
    if latest is not None:
        latest_iter = latest["it"]
        losses = latest["mses"]
        high_pass_losses = latest["high_pass_mses"]
        regularized_losses = latest["regularized_losses"]
        validate_losses = latest["validate_high_pass_mses"]
    else:
        latest_iter = 0
        losses = []
        high_pass_losses = []
        regularized_losses = []
        validate_losses = []

    # remove loaded checkpoint
    if latest is not None:
        del latest
        torch.cuda.empty_cache()

    # create mse loss
    mse = nn.MSELoss()

    # initialize iteration count
    it = int(latest_iter)

    # create folder for validation data
    if not os.path.exists("snapshots/validate/"):
        os.mkdir("snapshots/validate/")
    val_dir = "snapshots/validate/"

    # run psf training
    control.train_recon(
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
        single_decoder=True,
        input_3d=True,
        high_pass_kernel_size=11,
        low_pass_weight=0.1,
        validate_losses=validate_losses,
        validate_args={
            "dataloader": val_dataloader,
            "defocus_range": defocus_range,
            "target_shape": (
                subsampled_num_planes,
                downsampled_num_pixels,
                downsampled_num_pixels,
            ),
            "devices": devices,
            "extents": extents,
            "single_decoder": True,
            "input_3d": True,
            "high_pass_kernel_size": 11,
            "aperture_radius": downsampled_radius,
            "save_dir": val_dir,
        },
        it=it,
    )


def test():
    # initialize model for training
    if os.path.exists("latest.pt"):
        latest = torch.load("latest.pt")
    else:
        latest = None
    micdeconv = initialize_microscope_reconstruction(latest=latest)
    print(micdeconv)
    num_params = sum([p.view(-1).shape[0] for p in micdeconv["deconvs"].parameters()])
    print(num_params)

    # initialize data
    dataloader = create_dataloader(test=True)

    # remove loaded checkpoint
    if latest is not None:
        del latest
        torch.cuda.empty_cache()

    # initialize results storage folder
    if not os.path.exists(f"./test"):
        os.mkdir(f"./test")
    save_dir = f"./test"

    extents = create_test_extents()

    control.test_recon(
        micdeconv,
        dataloader,
        defocus_range,
        (subsampled_num_planes, downsampled_num_pixels, downsampled_num_pixels),
        devices,
        extents,
        single_decoder=True,
        input_3d=True,
        high_pass_kernel_size=11,
        aperture_radius=downsampled_radius,
        save_dir=save_dir,
    )


if __name__ == "__main__":
    if sys.argv[1:]:
        arg1 = sys.argv[1]
    else:
        arg1 = "train"
    if arg1 == "test":
        test()
    else:
        train()
