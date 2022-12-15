"""Module containing the abstract photon propagator interface."""
from abc import ABC, abstractmethod

import numpy as np
from numpy.random import Generator

from ananke.models.detector import Detector
from ananke.models.event import SourceRecords, Hits


class AbstractPhotonPropagator(ABC):
    """Parent class to ensure common interface for photon propagation."""

    def __init__(
            self,
            detector: Detector,
            c_medium: float,
            seed: int = 1337,
            **kwargs
    ) -> None:
        """Constructor already saving the detector.

        Args:
            detector: Detector to be set
        """
        super().__init__(**kwargs)
        self.detector = detector
        self.c_medium = c_medium
        self.rng = np.random.default_rng(seed)

    @abstractmethod
    def propagate(
            self, sources: SourceRecords
    ) -> Hits:
        """Propagates photon source towards the detector.

        Args:
            sources: photon source to propagate
            seed: seed by which to propagate

        Returns:
            List of the detector hits based on photon source
        """
        raise NotImplementedError('Propagate Function not implemented')
