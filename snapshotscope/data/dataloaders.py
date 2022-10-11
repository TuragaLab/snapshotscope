import os
import time
from glob import glob
from itertools import cycle, islice

import numpy as np
import pandas as pd
import skimage
import torch
import zarr
from torch.utils.data import Dataset

from snapshotscope.data.roi import ROI
from snapshotscope.utils import _list, _triple


class ConfocalVolumesDataset(Dataset):
    """
    Dataset for loading 3D volume chunks or whole volumes from large confocal
    image volumes formatted on disk as .zarr arrays. Supports padding of
    volumes to desired shape if selected location goes out of bounds of the
    sample. Supports multiple loading locations/strategies, as well as
    augmentations. Also supports caching of volumes in memory on first load to
    prevent slow disk or network file system reads during training. This can be
    disabled if not enough memory is available to cache all volumes.

    Args:
        paths: A list of paths that contain .zarr files to be loaded.

        shape (optional): A tuple describing the shape of the chunked volume to
        be taken out of the larger volume. Defaults to None, in which case the
        whole volume is returned.

        steps (optional): A tuple describing the step size to load the chunked
        volume in 3 dimensions. Defaults to (1, 1, 1).

        location (optional): A string that describes the location in the large
        volume to load from. Can be either 'random' to pull from a random
        location in (x, y, z), or 'center' to pull from the center of the large
        volume. If 'random' is selected, the valid_ratio describes the percent
        distance (0 - 1) from the center of each dimension that is allowed as a
        valid location. Defaults to 'random'.

        strategy (optional): A string that describes the strategy for loading
        from the large volume. If set to 'center' this will load the desired
        shape from the selected location as a midpoint. If set to 'top' this
        loads from the very top of the volume down to the desired shape using
        the x, y coordinates of the selected location but ignoring the z
        coordinate. Defaults to 'center'.

        bounds_roi (optional): Either a string or an ROI describing the safe
        region of interest. Defines the volume considered by valid_ratio. If
        set to the string 'data' then the actual entire large volume is
        considered as the ROI to possibly load from. Can be set to a smaller
        ROI if desired.

        augmentations (optional): A list of callables for each example in the
        dataset. Can actually be a list of just one callable, in which case the
        same callable is used for all the examples in the dataset. This
        callable can still contain randomness within itself. In fact, it can be
        any callable, and therefore change the shape or contents of the
        returned chunked volume. Defaults to None in which case no
        augmentations are performed.

        exclude (optional): A list of filenames including the full path (strings) that
        should be ignored. Useful to ignore data without deleting or moving it
        from a path. Defaults to None in which case no filenames are ignored in
        the given paths.

        balance (optional): A boolean representing whether to balance loading
        from each of the given paths. If True, then the files from each path
        will be duplicated to match the path with the largest number of .zarr
        files. This ensures that no path is overrepresented during training.
        Defaults to True.

        cache (optional): A boolean representing whether to cache large volumes
        in memory as they are requested. Defaults to True.

        num_tries (optional): An integer that defines how many times to
        reattempt to load a volume, in case the file system is unresponsive or
        the file is locked. Defaults to 2.

        return_dict (deprecated): An old formatting option that wraps the
        output into a dictionary. Not used anymore, and will be removed in
        future.

    """

    def __init__(
        self,
        paths,
        shape=None,
        steps=(1, 1, 1),
        location="random",
        strategy="center",
        bounds_roi="data",
        valid_ratio=1.0,
        augmentations=None,
        exclude=None,
        balance=True,
        repeat=0,
        cache=True,
        num_tries=2,
        return_dict=False,
    ):
        super().__init__()
        # collect .zarr filenames from specified folders
        self.paths = [os.path.realpath(path) for path in paths]
        self.fnames = [
            sorted(list(glob(os.path.join(path, "*.zarr")))) for path in self.paths
        ]
        if balance:
            # duplicate filenames to balance dataset sizes
            max_volumes = max([len(flist) for flist in self.fnames])
            self.fnames = [
                list(islice(cycle(flist), max_volumes)) for flist in self.fnames
            ]
        # create flat list of filenames
        self.fnames = [fname for flist in self.fnames for fname in flist]
        if exclude is not None:
            self.fnames = [f for f in self.fnames if f not in exclude]
        # repeat fnames if needed
        self.repeat = repeat
        if self.repeat > 0:
            self.fnames = self.fnames * repeat
        self.shape = shape
        self.steps = steps
        assert location in [
            "random",
            "center",
        ], "location must be either 'random' or 'center'"
        self.location = location
        assert bounds_roi in ["data"] or isinstance(
            bounds_roi, ROI
        ), "bounds_roi must be either 'data' or an ROI instance"
        self.strategy = strategy
        assert strategy in [
            "center",
            "top",
        ], "strategy must be either 'center' or 'top'"
        self.read_function = eval(f"self.from_{self.strategy}")
        self.bounds_roi = bounds_roi
        self.valid_ratio = _triple(valid_ratio)
        if augmentations is not None:
            augmentations = _list(augmentations)
        self.augmentations = augmentations
        self.cache = cache
        self.cached_data = [None for i in range(len(self.fnames))]
        assert num_tries > 1, "Number of tries must be >= 1"
        self.num_tries = num_tries
        self.return_dict = return_dict

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        for _ in range(self.num_tries):
            try:
                if self.cache and self.cached_data[index] is not None:
                    vol_group, vol_roi = self.cached_data[index]
                else:
                    vol_group, vol_roi = self.read_volume(self.fnames[index])
                if self.cache:
                    vol_group = vol_group[...]
                    self.cached_data[index] = (vol_group, vol_roi)
                bounds_roi = self.bounds_roi
                if bounds_roi == "data":
                    bounds_roi = self.safe_bounds(vol_roi)
                vol = self.read_function(vol_group, vol_roi, bounds_roi, self.location)
                # apply augmentations
                if self.augmentations is not None:
                    augment = self.augmentations[index % len(self.augmentations)]
                    out = augment(vol)
                    if isinstance(out, tuple):
                        vol = out[0]
                        logs = out[1]
                        logs.append(self.fnames[index])
                    if self.return_dict:
                        return {
                            "data": torch.from_numpy(
                                out, dtype=torch.float32
                            ).unsqueeze(0)
                        }
                    else:
                        return out
                else:
                    if self.return_dict:
                        return {
                            "data": torch.from_numpy(
                                vol, dtype=torch.float32
                            ).unsqueeze(0)
                        }
                    else:
                        return vol
            except:
                time.sleep(0.5)
                pass

    def read_volume(self, fname):
        zroot = zarr.open(fname, mode="r")
        vol_group = zroot["volume/data"]
        if "volume/attrs" in zroot:
            attrs_group = zroot["volume/attrs"]
            attrs = attrs_group.attrs
            vol_roi = ROI(attrs["extents"], attrs["pixel_size"])
        else:
            vol_roi = ROI([(0, d) for d in vol_group.shape])
        return vol_group, vol_roi

    def safe_bounds(self, vol_roi):
        sizes = (0,) * 3
        if self.shape is not None:
            sizes = tuple(sh * vol_roi.pixel_size[i] for i, sh in enumerate(self.shape))
        bounds_roi = vol_roi.grow([sz / 2 for sz in sizes], [-sz / 2 for sz in sizes])
        bounds_roi = bounds_roi.grow(
            [(1 - r) * 0.5 * sz for r, sz in zip(self.valid_ratio, bounds_roi.size)],
            [-(1 - r) * 0.5 * sz for r, sz in zip(self.valid_ratio, bounds_roi.size)],
        )
        bounds_roi = bounds_roi.intersect(vol_roi)
        return bounds_roi

    def from_center(self, vol_group, vol_roi, bounds_roi, location_func="random"):
        if not bounds_roi.empty() and self.shape is not None:
            shape = self.shape
            steps = tuple(int(s) for s in self.steps)
            location_command = f"(bounds_roi.{location_func}_location())"
            center = bounds_roi.to_pixels(eval(location_command))
            slices = tuple(
                slice(
                    center[d] - int(shape[d] / 2),
                    center[d] + int(shape[d] / 2),
                    steps[d],
                )
                for d in range(3)
            )
            clipped_slices = []
            for d in range(3):
                start = max(0, center[d] - int(shape[d] / 2))
                end = min(center[d] + int(shape[d] / 2), vol_roi.shape[d])
                clipped_slices.append(slice(start, end, steps[d]))
            clipped_slices = tuple(clipped_slices)
            vol = vol_group[clipped_slices]
            pads = []
            pad = False
            for d in range(3):
                to_pad = [
                    round((clipped_slices[d].start - slices[d].start) / steps[d]),
                    round((slices[d].stop - clipped_slices[d].stop) / steps[d]),
                ]
                if any([p > 0 for p in to_pad]):
                    pad = True
                    if sum(to_pad) != ((self.shape[d] / steps[d]) - vol.shape[d]):
                        to_pad[-1] += 1
                pads.append(tuple(to_pad))
            if pad:
                background = np.mean(vol_group[0:10, 0:10, 0:10])
                vol = skimage.util.pad(vol, pads, constant_values=background)
            vol = vol[0 : int(shape[0] / steps[0])]
            return vol
        elif bounds_roi.empty():
            shape = vol_roi.shape
            steps = tuple(int(s) for s in self.steps)
            slices = tuple(slice(0, shape[d], steps[d]) for d in range(3))
            vol = vol_group[slices]
            return vol
        else:
            return vol_group[:]

    def from_top(self, vol_group, vol_roi, bounds_roi, location_func="random"):
        if not bounds_roi.empty() and self.shape is not None:
            shape = self.shape
            steps = tuple(int(s) for s in self.steps)
            location_command = f"(bounds_roi.{location_func}_location())"
            center = bounds_roi.to_pixels(eval(location_command))
            slices = tuple(
                slice(
                    center[d] - int(shape[d] / 2),
                    center[d] + int(shape[d] / 2),
                    steps[d],
                )
                for d in [1, 2]
            )
            slices = (slice(0, self.shape[0], steps[0]), *slices)
            clipped_slices = [slice(0, min(self.shape[0], vol_roi.shape[0]), steps[0])]
            for d in [1, 2]:
                start = max(0, center[d] - int(shape[d] / 2))
                end = min(center[d] + int(shape[d] / 2), vol_roi.shape[d])
                clipped_slices.append(slice(start, end, steps[d]))
            clipped_slices = tuple(clipped_slices)
            vol = vol_group[clipped_slices]
            pads = []
            pad = False
            for d in range(3):
                to_pad = [
                    round((clipped_slices[d].start - slices[d].start) / steps[d]),
                    round((slices[d].stop - clipped_slices[d].stop) / steps[d]),
                ]
                if any([p > 0 for p in to_pad]):
                    pad = True
                    if sum(to_pad) != ((self.shape[d] / steps[d]) - vol.shape[d]):
                        to_pad[-1] += 1
                pads.append(tuple(to_pad))
            if pad:
                background = np.mean(vol_group[0:10, 0:10, 0:10])
                vol = skimage.util.pad(vol, pads, constant_values=background)
            return vol
        elif bounds_roi.empty():
            shape = vol_roi.shape
            steps = tuple(int(s) for s in self.steps)
            slices = tuple(slice(0, shape[d], steps[d]) for d in range(3))
            vol = vol_group[slices]
            return vol
        else:
            return vol_group[:]


class DiffuserMirflickrDataset(Dataset):
    """
    Dataset for loading pairs of diffused images collected through DiffuserCam
    and ground truth, unblurred images collected through a DSLR camera. For use
    with DLMD (DiffuserCam Lensless Mirflickr Dataset). Optionally supports any
    callable transform.

    Args:
        csv_path: Path to .csv file containing filenames of images (both
        diffused and ground truth images share the same filename).

        data_dir: Path to directory containing diffused image data.

        label_dir: Path to directory containing ground truth image data.

        transform (optional): An optional callable that will be applied to
        every image pair. Defaults to None, in which case nothing happens.
    """

    def __init__(self, csv_path, data_dir, label_dir, transform=None):
        super().__init__()
        self.csv_contents = pd.read_csv(csv_path)
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.transform = transform

    def __len__(self):
        return len(self.csv_contents)

    def __getitem__(self, idx):

        img_name = self.csv_contents.iloc[idx, 0]

        path_diffused = os.path.join(self.data_dir, img_name)
        path_gt = os.path.join(self.label_dir, img_name)

        image = np.load(path_diffused[0:-9] + ".npy")
        label = np.load(path_gt[0:-9] + ".npy")
        image = image.transpose((2, 0, 1))
        label = label.transpose((2, 0, 1))
        image = torch.from_numpy(image)
        label = torch.from_numpy(label)

        sample = {"image": image, "label": label}

        if self.transform:
            sample = self.transform(sample)

        return sample

