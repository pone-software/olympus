from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from .detector import Detector
from .constants import defaults


class AbstractInjector(ABC):
    def __init__(
        self,
        detector: Detector,
        radius_extension: Optional[float] = 50,
        height_extension: Optional[float] = 100,
        rng: Optional[np.random.RandomState] = defaults['rng'],
        **kwargs,
    ) -> None:
        self.detector = detector
        self.radius_extension = radius_extension
        self.height_extension = height_extension

        self.cylinder_height = self.detector.outer_cylinder[1] + height_extension
        self.cylinder_radius = self.detector.outer_cylinder[0] + radius_extension
        self.rng = rng

    @abstractmethod
    def get_position(
        self, n: Optional[int] = 1
    ) -> np.ndarray:
        pass


class VolumeInjector(AbstractInjector):
    def get_position(
        self, n: Optional[int] = 1
    ) -> np.ndarray:
        theta = self.rng.uniform(0, 2 * np.pi, size=n)
        r = self.cylinder_radius * np.sqrt(self.rng.uniform(0, 1, size=n))
        samples = np.empty((n, 3))
        samples[:, 0] = r * np.sin(theta)
        samples[:, 1] = r * np.cos(theta)
        samples[:, 2] = self.rng.uniform(
            -self.cylinder_height / 2, self.cylinder_height / 2, size=n
        )

        return samples


class SurfaceInjector(AbstractInjector):
    def get_position(
        self, n: Optional[int] = 1
    ) -> np.ndarray:
        """Sample points on a cylinder surface.

        :param n: number of positions to generate, defaults to 1
        :type n: Optional[int], optional
        :return: numpy array containing all the positions as numpy arrays
        :rtype: np.ndarray
        """
        radius = self.cylinder_radius
        height = self.cylinder_height
        side_area = 2 * np.pi * radius * height
        top_area = 2 * np.pi * radius**2

        ratio = top_area / (top_area + side_area)

        is_top = self.rng.uniform(0, 1, size=n) < ratio
        n_is_top = is_top.sum()
        samples = np.empty((n, 3))
        theta = self.rng.uniform(0, 2 * np.pi, size=n)

        # top / bottom points

        r = radius * np.sqrt(self.rng.uniform(0, 1, size=n_is_top))

        samples[is_top, 0] = r * np.sin(theta[is_top])
        samples[is_top, 1] = r * np.cos(theta[is_top])
        samples[is_top, 2] = self.rng.choice(
            [-height / 2, height / 2], replace=True, size=n_is_top
        )

        # side points

        r = radius
        samples[~is_top, 0] = r * np.sin(theta[~is_top])
        samples[~is_top, 1] = r * np.cos(theta[~is_top])
        samples[~is_top, 2] = self.rng.uniform(-height / 2, height / 2, size=n - n_is_top)

        return samples
