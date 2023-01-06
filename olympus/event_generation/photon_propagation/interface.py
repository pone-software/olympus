"""Module containing the abstract photon propagator interface."""
from abc import ABC, abstractmethod

import numpy as np

from ananke.models.detector import Detector
from ananke.models.event import Sources, Hits, Records
from olympus.constants import defaults
from olympus.event_generation.medium import Medium


class AbstractPhotonPropagator(ABC):
    """Parent class to ensure common interface for photon propagation."""

    def __init__(
            self,
            detector: Detector,
            medium: Medium,
            seed: int = defaults['seed'],
            default_wavelength: float = 450,
            **kwargs
    ) -> None:
        """Constructor already saving the detector.

        Args:
            detector: Detector to be set
            medium: Medium for which to propagate
            seed: Seed of random number Generator
        """
        super().__init__(**kwargs)
        self.detector = detector
        self.medium = medium
        self.default_wavelengths = default_wavelength
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    @abstractmethod
    def propagate(
            self,
            records: Records,
            sources: Sources,
            **kwargs
    ) -> Hits:
        """Propagates photon source towards the detector.

        Args:
            records: events of sources to propagate
            sources: photon source to propagate

        Returns:
            List of the detector hits based on photon source
        """
        raise NotImplementedError('Propagate Function not implemented')
