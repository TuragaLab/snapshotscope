import random

import numpy as np

from snapshotscope.utils import _list


class ROI:
    """
    ROI (region of interest) implementation based on ``daisy.Roi`` from
    https://github.com/funkelab/daisy.
    """

    def __init__(self, extents, pixel_size=None, pixel_offset=None):
        self.extents = extents
        if pixel_size is None:
            pixel_size = (1.0,) * len(self.extents)
        if pixel_offset is None:
            pixel_offset = (0,) * len(self.extents)
        self.pixel_size = pixel_size
        self.pixel_offset = pixel_offset

    def __repr__(self):
        return "\n".join(
            [
                f"ROI ({self.size} @ {self.pixel_size})",
                f"extents: {self.extents}",
                f"shape: {self.shape}",
                f"pixel_offset: {self.pixel_offset}",
                "=====================================",
            ]
        )

    @property
    def size(self):
        return tuple(end - start for (start, end) in self.extents)

    @property
    def shape(self):
        return tuple(int((sz) / self.pixel_size[i]) for i, sz in enumerate(self.size))

    def dims(self):
        return len(self.shape)

    def start(self):
        return tuple(s if s is not None else -np.inf for (s, e) in self.extents)

    def end(self):
        return tuple(e if e is not None else np.inf for (s, e) in self.extents)

    def volume(self):
        return np.prod(self.size)

    def empty(self):
        return all(tuple((end - start) == 0 for (start, end) in self.extents))

    def contains(self, other):
        if not isinstance(other, ROI):
            return all(
                [other[d] >= self.start()[d] for d in range(self.dims())]
            ) and all([other[d] < self.end()[d] for d in range(self.dims())])
        if other.empty():
            return self.contains(other.start())
        else:
            return self.contains(other.start()) and self.contains(
                other.end() - (1,) * other.dims()
            )

    def intersects(self, other):
        assert self.dims() == other.dims(), "ROIs must have same dims"
        if self.empty() or other.empty():
            return False
        separated = any(
            [
                ((None not in [s1, s2, e1, e2]) and ((s1 >= e2) or (s2 >= e2)))
                for (s1, s2, e1, e2) in zip(
                    self.start(), other.start(), self.end(), other.end()
                )
            ]
        )
        return not separated

    def intersect(self, other):
        if not self.intersects(other):
            return ROI([(0, 0) for d in self.dims()], self.pixel_size)
        start = tuple(max(s1, s2) for s1, s2 in zip(self.start(), other.start()))
        end = tuple(min(e1, e2) for e1, e2 in zip(self.end(), other.end()))
        start = tuple(s if s > -np.inf else None for s in start)
        end = tuple(e if e < np.inf else None for e in end)
        extents = [(s, e) for (s, e) in zip(start, end)]
        pixel_offset = tuple(
            (self.pixel_offset[i] + int(max(0, s2 - s1) / self.pixel_size[i]))
            for (i, (s1, s2)) in enumerate(zip(self.start(), start))
        )
        return ROI(extents, pixel_size=self.pixel_size, pixel_offset=pixel_offset)

    def union(self, other):
        start = tuple(min(s1, s2) for s1, s2 in zip(self.start(), other.start()))
        end = tuple(max(e1, e2) for e1, e2 in zip(self.end(), other.end()))
        start = tuple(s if s > -np.inf else None for s in start)
        end = tuple(e if e < np.inf else None for e in end)
        extents = [(s, e) for (s, e) in zip(start, end)]
        pixel_offset = tuple(
            min(p1, p2) for p1, p2 in zip(self.pixel_offset, other.pixel_offset)
        )
        return ROI(extents, pixel_size=self.pixel_size, pixel_offset=pixel_offset)

    def shift(self, by):
        extents = [(s + b, e + b) for (b, (s, e)) in zip(by, self.extents)]
        return ROI(extents, self.pixel_size)

    def grow(self, amount_neg, amount_pos):
        if amount_neg is None:
            amount_neg = (0,) * self.dims()
        if amount_pos is None:
            amount_pos = (0,) * self.dims()
        amount_neg = _list(amount_neg, repetitions=self.dims())
        amount_pos = _list(amount_pos, repetitions=self.dims())
        pixel_offset = tuple(
            self.pixel_offset[i] + int(n / self.pixel_size[i])
            for i, n in enumerate(amount_neg)
        )
        start = tuple(s + amount_neg[i] for i, s in enumerate(self.start()))
        end = tuple(e + amount_pos[i] for i, e in enumerate(self.end()))
        start = tuple(s if s > -np.inf else None for s in start)
        end = tuple(e if e < np.inf else None for e in end)
        extents = [(s, e) for (s, e) in zip(start, end)]
        return ROI(extents, pixel_size=self.pixel_size, pixel_offset=pixel_offset)

    def interpolate(self, pixel_size):
        """
        Creates new ROI with same extent in real units but different shape
        resulting from the specified ``pixel_size``.
        """
        return ROI(self.extents, pixel_size=pixel_size)

    def random_location(self):
        return tuple(random.uniform(s, e) for (s, e) in self.extents)

    def center_location(self):
        return tuple(0.5 * (s + e) for (s, e) in self.extents)

    def to_pixels(self, location):
        pixel_offset = self.pixel_offset
        pixel_size = self.pixel_size
        return tuple(
            (pixel_offset[i] + int((loc - s - pixel_size[i] * 1e-6) / pixel_size[i]))
            for i, (loc, (s, e)) in enumerate(zip(location, self.extents))
        )
