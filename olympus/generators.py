from abc import ABC, abstractmethod
from typing import Optional, Type

import numpy as np

from ananke.models.detector import Detector

from ananke.models.event import Collection, EventRecords, Sources
from olympus.constants import defaults
from olympus.event_generation.photon_propagation.interface import AbstractPhotonPropagator


class AbstractGenerator(ABC):
    def __init__(
            self,
            seed: Optional[int] = defaults["seed"],
            *args,
            **kwargs
    ):
        super().__init__()
        self.rng = np.random.default_rng(seed)

    @abstractmethod
    def generate(
            self,
            number_of_samples: int,
    ) -> Collection:
        pass
