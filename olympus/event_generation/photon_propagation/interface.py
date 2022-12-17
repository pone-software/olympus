"""Module containing the abstract photon propagator interface."""
from abc import ABC, abstractmethod

import numpy as np
from numpy.random import Generator

from ananke.models.detector import Detector
from ananke.models.event import SourceRecords, Hits, EventRecords
from olympus.event_generation.medium import Medium


class AbstractPhotonPropagator(ABC):
    """Parent class to ensure common interface for photon propagation."""

    def __init__(
            self,
            detector: Detector,
            medium: Medium,
            seed: int = 1337,
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
        self.rng = np.random.default_rng(seed)

    @abstractmethod
    def propagate(
            self,
            events: EventRecords,
            sources: SourceRecords,
            **kwargs
    ) -> Hits:
        """Propagates photon source towards the detector.

        Args:
            events: events of sources to propagate
            sources: photon source to propagate

        Returns:
            List of the detector hits based on photon source
        """
        raise NotImplementedError('Propagate Function not implemented')
