"""Module containing the abstract photon propagator interface."""
from abc import ABC, abstractmethod
from typing import TypeVar, Generic

import numpy as np

from ananke.models.detector import Detector
from ananke.models.event import Sources, Hits, Records
from olympus.configuration.photon_propagation import PhotonPropagatorConfiguration
from olympus.event_generation.medium import Medium

_PhotonPropagatorConfiguration = TypeVar(
    '_PhotonPropagatorConfiguration',
    bound=PhotonPropagatorConfiguration
)


class AbstractPhotonPropagator(ABC, Generic[_PhotonPropagatorConfiguration]):
    """Parent class to ensure common interface for photon propagation."""

    def __init__(
            self,
            detector: Detector,
            configuration: _PhotonPropagatorConfiguration
    ) -> None:
        """Constructor already saving the detector.

        Args:
            detector: Detector to be set
            configuration: Configuration for photon propagator
        """
        self.configuration = configuration
        self.detector = detector
        self.medium = Medium(configuration.medium)
        self.default_wavelengths = configuration.default_wavelength
        self.seed = configuration.seed
        self.rng = np.random.default_rng(configuration.seed)

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
