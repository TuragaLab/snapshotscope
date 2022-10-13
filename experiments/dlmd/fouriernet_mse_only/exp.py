import os, sys, math, datetime, glob, faulthandler

faulthandler.enable()

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.cuda._utils import _get_device_index

import snapshotscope
import snapshotscope.networks.deconv as d
import snapshotscope.data.dataloaders as data
import snapshotscope.networks.control as control

import logging

logging.basicConfig(filename="out.log", level=logging.DEBUG, format="%(message)s")

learning_rate = 1e-4
num_iterations = 1000000

# setup the simulation parameters, make a microscope
image_shape = (1080, 1920)
num_chunks = 1
devices = [torch.device(f"cuda:{i}") for i in range(num_chunks)]

# calculate downsampled sizes
downsample = 4
downsampled_image_shape = (int(s / downsample) for s in image_shape)


def create_reconstruction_network():
    # create multi gpu reconstruction network list for 1 gpu
    deconv = d.FourierNetRGB(
        20,
        downsampled_image_shape,
        fourier_conv_args={"stride": 2},
        conv_kernel_sizes=[(11, 11), (11, 11), (11, 11)],
        conv_fmap_nums=[64, 64, 3],
        input_scaling_mode=None,
        device=devices[0],
    )
    return deconv


def initialize_reconstruction(latest=None):
    deconv = create_reconstruction_network()
    if latest is not None:
        print("[info] loading from checkpoint")
        deconv.load_state_dict(latest["deconv_state_dict"], strict=True)
    return deconv


def initialize_optimizer(deconv, latest=None):
    # optimize microscope parameters and reconstruction network
    opt = optim.Adam(
        [{"params": deconv.parameters(), "lr": learning_rate}], lr=learning_rate
    )
    if latest is not None:
        opt.load_state_dict(latest["opt_state_dict"])
    return opt


def create_dataset(test=False):
    base_path = "../../../data/dlmd/dataset/"
    data_dir = os.path.join(base_path, "diffuser_images")
    label_dir = os.path.join(base_path, "ground_truth_lensed")
    if not test:
        csv_path = os.path.join(base_path, "dataset_train.csv")
        dataset = data.DiffuserMirflickrDataset(csv_path, data_dir, label_dir)
    else:
        csv_path = os.path.join(base_path, "dataset_test.csv")
        dataset = data.DiffuserMirflickrDataset(csv_path, data_dir, label_dir)
    return dataset


def create_dataloader(dataset, test=False):
    if not test:
        dataloader = torch.utils.data.DataLoader(
            dataset, num_workers=5, batch_size=4, shuffle=True
        )
    else:
        dataloader = torch.utils.data.DataLoader(
            dataset, num_workers=5, batch_size=1, shuffle=False
        )
    return dataloader


def train():
    # initialize model for training
    if os.path.exists("latest.pt"):
        latest = torch.load("latest.pt")
    else:
        latest = None
    deconv = initialize_reconstruction(latest=latest)
    print(deconv)

    # initialize optimizer
    opt = initialize_optimizer(deconv, latest=latest)

    # initialize data
    dataset = create_dataset()
    dataset, val_dataset = torch.utils.data.random_split(
        dataset, [23000, 1000], generator=torch.Generator().manual_seed(42)
    )
    dataloader = create_dataloader(dataset)
    val_dataloader = create_dataloader(val_dataset, test=True)

    # initialize iteration count
    if latest is not None:
        latest_iter = latest["it"]
        losses = latest["mses"]
        validate_losses = latest["validate_mses"]
    else:
        latest_iter = 0
        losses = []
        validate_losses = []

    # remove loaded checkpoint
    if latest is not None:
        del latest
        torch.cuda.empty_cache()

    # initialize iteration count
    it = int(latest_iter)

    # create folder for validation data
    if not os.path.exists("snapshots/validate/"):
        os.mkdir("snapshots/validate/")
    val_dir = "snapshots/validate/"

    # run psf training
    control.train_rgb_recon(
        deconv,
        opt,
        dataloader,
        devices,
        losses,
        num_iterations,
        validate_mses=validate_losses,
        validate_args={
            "dataloader": val_dataloader,
            "devices": devices,
            "save_dir": val_dir,
        },
        it=it,
    )


def test():
    # initialize model for training
    if os.path.exists("latest.pt"):
        latest = torch.load("snapshots/state220000.pt")
    else:
        latest = None
    deconv = initialize_reconstruction(latest=latest)
    print(deconv)
    num_params = sum([p.view(-1).shape[0] for p in deconv.parameters()])
    print(num_params)

    # initialize data
    dataset = create_dataset(test=True)
    dataloader = create_dataloader(dataset, test=True)

    # remove loaded checkpoint
    if latest is not None:
        del latest
        torch.cuda.empty_cache()

    # initialize results storage folder
    if not os.path.exists(f'../test/{os.getcwd().split("/")[-1]}'):
        os.mkdir(f'../test/{os.getcwd().split("/")[-1]}')
    save_dir = f'../test/{os.getcwd().split("/")[-1]}'

    control.test_rgb_recon(deconv, dataloader, devices, save_dir=save_dir)


def test_train():
    raise NotImplementedError


if __name__ == "__main__":
    if sys.argv[1:]:
        arg1 = sys.argv[1]
    else:
        arg1 = "train"
    if arg1 == "test":
        test()
    elif arg1 == "test_train":
        test_train()
    else:
        train()
