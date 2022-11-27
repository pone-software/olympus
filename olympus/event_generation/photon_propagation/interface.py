"""Module containing the abstract photon propagator interface."""
from abc import ABC, abstractmethod

from ananke.models.detector import Detector
from ananke.models.event import SourceRecordCollection, HitCollection


class AbstractPhotonPropagator(ABC):
    """Parent class to ensure common interface for photon propagation."""

    def __init__(
            self,
            detector: Detector,
            c_medium: float,
            **kwargs
    ) -> None:
        """Constructor already saving the detector.

        Args:
            detector: Detector to be set
        """
        super().__init__(**kwargs)
        self.detector = detector
        self.c_medium = c_medium
        self.detector_df = detector.to_pandas()

    @abstractmethod
    def propagate(
            self, sources: SourceRecordCollection,
            seed: int = 1337
    ) -> HitCollection:
        """Propagates photon source towards the detector.

        Args:
            sources: photon source to propagate
            seed: seed by which to propagate

        Returns:
            List of the detector hits based on photon source
        """
        raise NotImplementedError('Propagate Function not implemented')
