from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from .constants import defaults


class AbstractSpectrum(ABC):
    def __init__(self, rng: Optional[np.random.RandomState] = defaults['rng']) -> None:
        super().__init__()
        self.rng = rng

    @abstractmethod
    def get_energy(
        self, n: Optional[int] = 1
    ) -> np.ndarray:
        pass


class UniformSpectrum(AbstractSpectrum):
    def __init__(self, log_minimal_energy: float, log_maximal_energy: float) -> None:
        super().__init__()
        self.log_minimal_energy = log_minimal_energy
        self.log_maximal_energy = log_maximal_energy

    def get_energy(self, n: Optional[int] = 1) -> np.ndarray:
        return np.power(10, self.rng.uniform(self.log_minimal_energy, self.log_maximal_energy, size=n))