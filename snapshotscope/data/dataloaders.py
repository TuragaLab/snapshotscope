import os
import random
import time
from glob import glob
from itertools import cycle, islice

import numpy as np
import pandas as pd
import skimage
import tifffile
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


class TIFFStacksDataset(Dataset):
    def __init__(
        self,
        paths,
        augmentations=None,
        exclude=None,
        balance=False,
        num_tries=2,
    ):
        super().__init__()
        # collect .tif(f) filenames from specified folders
        self.paths = [os.path.realpath(path) for path in paths]
        self.fnames = [
            sorted(list(glob(os.path.join(path, "*.tif*")))) for path in self.paths
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
        if augmentations is not None:
            augmentations = _list(augmentations)
        self.augmentations = augmentations
        assert num_tries > 1, "Number of tries must be >= 1"
        self.num_tries = num_tries

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        for _ in range(self.num_tries):
            try:
                im = tifffile.imread(self.fnames[index])
                break
            except:
                time.sleep(0.5)
                pass
        im = np.float32(im)
        if self.augmentations is not None:
            augment = self.augmentations[index % len(self.augmentations)]
            im = augment(im)
        # add channel dimension
        im = np.expand_dims(im, 0)
        return im


class NumpyStacksDataset(Dataset):
    def __init__(self, path, augmentations=None, exclude=None, num_tries=2):
        super().__init__()
        self.path = os.path.realpath(path)
        self.fnames = sorted(list(glob(os.path.join(self.path, "*.npy"))))
        if exclude is not None:
            self.fnames = [f for f in self.fnames if f not in exclude]
        self.augmentations = _list(augmentations)
        self.num_tries = num_tries

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        for _ in range(self.num_tries):
            try:
                im = np.load(self.fnames[index % len(self.fnames)])
                break
            except:
                time.sleep(0.5)
                pass
        im = im.astype(np.float32)
        if self.augmentations is not None:
            augment = self.augmentations[index % len(self.augmentations)]
            im = augment(im)
        return im


class BalancedZarrStacksDataset(Dataset):
    def __init__(
        self,
        paths,
        augmentations=None,
        exclude=None,
        num_tries=2,
        random_location=True,
        center=None,
        valid_ratio=0.5,
        max_depth=None,
        max_size=None,
        step=1,
    ):
        super().__init__()
        # collect .zarr filenames from specified folders
        self.paths = [os.path.realpath(path) for path in paths]
        self.fnames = [
            sorted(list(glob(os.path.join(path, "*.zarr")))) for path in self.paths
        ]
        # duplicate filenames to balance dataset sizes
        max_volumes = max([len(flist) for flist in self.fnames])
        self.fnames = [list(islice(cycle(flist), max_volumes)) for flist in self.fnames]
        # create flat list of filenames
        self.fnames = [fname for flist in self.fnames for fname in flist]
        if exclude is not None:
            self.fnames = [f for f in self.fnames if f not in exclude]
        if augmentations is not None:
            self.augmentations = _list(augmentations)
        else:
            self.augmentations = None
        self.num_tries = num_tries
        self.random_location = random_location
        self.center = center
        self.max_size = max_size
        self.max_depth = max_depth
        self.valid_ratio = valid_ratio
        self.step = step

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        fname = None
        for _ in range(self.num_tries):
            try:
                fname = self.fnames[index % len(self.fnames)]
                f = zarr.open(fname, mode="r")
                if self.random_location and self.max_size != None:
                    ds = f["volume/data"]
                    center = [
                        random.randint(
                            int((1 - self.valid_ratio) / 2 * ds.shape[d]),
                            int((1 + self.valid_ratio) / 2 * ds.shape[d]),
                        )
                        for d in [1, 2]
                    ]
                    depth = ds.shape[0] if self.max_depth == None else self.max_depth
                    midplane = int(ds.shape[0] / 2)
                    start = max(0, midplane - int(depth / 2))
                    end = min(ds.shape[0], start + depth)
                    im = ds[
                        start : end : self.step,
                        max(center[-2] - self.max_size, 0) : min(
                            center[-2] + self.max_size + 1, ds.shape[1]
                        ),
                        max(center[-1] - self.max_size, 0) : min(
                            center[-1] + self.max_size + 1, ds.shape[2]
                        ),
                    ]
                elif self.max_size != None:
                    ds = f["volume/data"]
                    if self.center is None:
                        center = [int(ds.shape[d] / 2) for d in [1, 2]]
                    else:
                        center = self.center
                    depth = ds.shape[0] if self.max_depth == None else self.max_depth
                    midplane = int(ds.shape[0] / 2)
                    start = max(0, midplane - int(depth / 2))
                    end = min(ds.shape[0], start + depth)
                    im = ds[
                        start : end : self.step,
                        max(center[-2] - self.max_size, 0) : min(
                            center[-2] + self.max_size + 1, ds.shape[1]
                        ),
                        max(center[-1] - self.max_size, 0) : min(
                            center[-1] + self.max_size + 1, ds.shape[2]
                        ),
                    ]
                else:
                    im = f["volume/data"][:]
                break
            except:
                index += 1
                pass
        im = im.astype(np.float32)
        if self.augmentations is not None:
            augment = self.augmentations[index % len(self.augmentations)]
            out = augment(im)
            if isinstance(out, tuple):
                im = out[0]
                logs = out[1]
                logs.append(fname)
                return out
            else:
                im = out
        return im


class LSMStacksDataset(Dataset):
    def __init__(
        self,
        path,
        augmentations=None,
        exclude=None,
        order="DCHW",
        num_channels=2,
        num_tries=2,
    ):
        super().__init__()
        self.path = os.path.realpath(path)
        self.fnames = sorted(list(glob(os.path.join(self.path, "*.lsm"))))
        if exclude is not None:
            self.fnames = [f for f in self.fnames if f not in exclude]
        self.order = order.upper()
        assert self.order in ["DCHW", "CDHW"], "LSM order can be DCHW or CDHW"
        self.num_channels = num_channels
        self.augmentations = _list(augmentations)
        self.num_tries = num_tries

    def __len__(self):
        return len(self.fnames) * self.num_channels

    def __getitem__(self, index):
        for _ in range(self.num_tries):
            try:
                im = tifffile.imread(
                    self.fnames[int(index / self.num_channels) % len(self.fnames)]
                )[0]
                break
            except:
                time.sleep(0.5)
                pass
        im = im.astype(np.float32)
        if self.order == "DCHW":
            im = im[:, index % self.num_channels, :, :]
        elif self.order == "CDHW":
            im = im[index % self.num_channels]
        else:
            raise ValueError("LSM order should have been DCHW or CDHW")
        if self.augmentations is not None:
            augment = self.augmentations[index % len(self.augmentations)]
            im = augment(im)
        return im


class LambdaDataset(Dataset):
    def __init__(self, function, arg_dict, augmentations=None, num_tries=2):
        super().__init__()
        self.function = function
        self.augmentations = _list(augmentations)
        self.num_tries = num_tries

    def __len__(self):
        return 1

    def __getitem__(self, index):
        for _ in range(self.num_tries):
            try:
                vol = self.function(**arg_dict)
                break
            except:
                time.sleep(0.5)
                pass
        im = im.astype(np.float32)
        if self.augmentations is not None:
            augment = self.augmentations[index % len(self.augmentations)]
            im = augment(im)
        return im


class BalancedConcatDataset(Dataset):
    def minsize(self):
        lengths = [len(d) for d in self.datasets]
        return min(lengths)

    def __init__(self, datasets):
        super().__init__()
        assert len(datasets) > 0, "datasets should not be an empty iterable"
        self.datasets = list(datasets)
        for d in self.datasets:
            assert not isinstance(
                d, torch.utils.data.IterableDataset
            ), "BalancedConcatDataset does not support IterableDataset"
        self.min_size = self.minsize()

    def __len__(self):
        return self.min_size * len(self.datasets)

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError(
                    "absolute value of index should not exceed dataset length"
                )
            idx = len(self) + idx
        dataset_idx = int(idx / self.min_size)
        if len(self.datasets[dataset_idx]) == self.min_size:
            sample_idx = idx % self.min_size
        else:
            sample_idx = random.randint(0, len(self.datasets[dataset_idx]) - 1)
        return self.datasets[dataset_idx][sample_idx]


class CacheDataset(Dataset):
    """
    Wrapper for any map-based Dataset that caches the desired number of
    examples in memory once they are loaded for the first time. This can be
    helpful to reduce spikes in loading time from a Dataset where the data
    source is slow to read, e.g. large files on a networked file system or a
    hard drive.

    Can be used to implement a simple version of the data caching outlined in
    [Faster Neural Network Training with Data
    Echoing](https://arxiv.org/abs/1907.05550).

    Args:

        dataset (Dataset): A map-based torch.data.Dataset that provides the
        real desired examples.

        cache_size (int, optional): An integer determining the number of
        examples to be cached.

        flush_every (int, optional): An integer determining the number of
        retrievals before the cache is flushed and new examples are selected
        for caching.

        augmentations (callable, optional): A callable that will be used to
        augment the data before returning it.
    """

    def __init__(self, dataset, cache_size=20, flush_every=100, augmentations=None):
        super(CacheDataset).__init__()
        self.dataset = dataset
        self.augmentations = augmentations
        self.cache_size = cache_size
        self.flush_every = flush_every
        self.flush_cache()

    def __len__(self):
        return self.cache_size

    def __getitem__(self, index):
        if self.cache[index] is not None:
            # if we already cached this sample, return it from cache
            data = self.cache[index]
        else:
            # otherwise, load the sample from real dataset and cache it
            self.cache[index] = self.dataset[self.chosen_indices[index]]
            data = self.cache[index]
        self.num_retrievals += 1
        if self.num_retrievals >= self.flush_every:
            # flush cache if we have retrieved the same examples enough times
            self.flush_cache()
        # apply augmentations
        if self.augmentations is not None:
            data = self.augmentations(data)
        return data

    def flush_cache(self):
        """Clears current cache and randomly chooses new examples to cache."""
        self.num_retrievals = 0
        self.chosen_indices = random.choices(
            list(range(len(self.dataset))), k=self.cache_size
        )
        self.cache = [None for i in range(self.cache_size)]
