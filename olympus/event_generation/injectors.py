from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from .detector import Detector
from .constants import Defaults

class AbstractInjector(ABC):
    def __init__(self, detector: Detector, radius_extension: Optional[float] =50, height_extension: Optional[float] = 100) -> None:
        super().__init__()
        self.detector = detector
        self.radius_extension = radius_extension
        self.height_extension = height_extension
        
        self.cylinder_height = self.detector.outer_cylinder[1] + height_extension
        self.cylinder_radius = self.detector.outer_cylinder[0] + radius_extension

    @abstractmethod
    def get_position(self, rng: Optional[np.random.RandomState] = Defaults.rng, n: Optional[int] = 1) -> np.ndarray:
        pass

class VolumeInjector(AbstractInjector):
    def get_position(self, rng: Optional[np.random.RandomState] = Defaults.rng, n: Optional[int] = 1) -> np.ndarray:
        theta = rng.uniform(0, 2 * np.pi, size=n)
        r = self.cylinder_radius * np.sqrt(rng.uniform(0, 1, size=n))
        samples = np.empty((n, 3))
        samples[:, 0] = r * np.sin(theta)
        samples[:, 1] = r * np.cos(theta)
        samples[:, 2] = rng.uniform(-self.cylinder_height / 2, self.cylinder_height / 2, size=n)

        return samples


class SurfaceInjector(AbstractInjector):
    def get_position(self, rng: Optional[np.random.RandomState] = Defaults.rng, n: Optional[int] = 1) -> np.ndarray:
        
        