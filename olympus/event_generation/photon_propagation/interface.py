"""Module containing the abstract photon propagator interface."""
from abc import ABC, abstractmethod
from typing import TypeVar, Generic, List, Optional, Union

import numpy as np

from ananke.models.collection import Collection
from ananke.models.detector import Detector
from ananke.models.event import Hits
from ananke.schemas.event import Types
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
            collection: Collection,
            record_type: Optional[
                Union[
                    List[Types],
                    Types
                ]
            ] = None,
            **kwargs
    ) -> Hits:
        """Propagates photon source towards the detector.

        Args:
            collection: Collection of Records and Sources to Propagate
            record_type: Record type to propagate

        Returns:
            List of the detector hits based on photon source
        """
        raise NotImplementedError('Propagate Function not implemented')
