"""Module containing the factory for photon propagators."""
from typing import Type, Optional

from ananke.models.detector import Detector
from olympus.configuration.photon_propagation import (
    PhotonPropagatorConfiguration,
    PhotonPropagators,
)
from olympus.event_generation.photon_propagation.interface import \
    AbstractPhotonPropagator
from olympus.event_generation.photon_propagation.mock_photons import \
    MockPhotonPropagator
from olympus.event_generation.photon_propagation.norm_flow_photons import \
    NormalFlowPhotonPropagator


def get_photon_propagator(
        detector: Detector,
        configuration: PhotonPropagatorConfiguration
):
    """Get photon propagator.

    Args:
        configuration: for photon propagator
        detector: Detector of the propagator

    Returns:
        Photon propagator based on configuration
    """
    cls: Optional[Type[AbstractPhotonPropagator]] = None
    if configuration.type == PhotonPropagators.Mock:
        cls = MockPhotonPropagator

    if configuration.type == PhotonPropagators.NormalFlow:
        cls = NormalFlowPhotonPropagator

    if cls is None:
        ValueError(
            'Photon propagator type {} does not exist'.format(configuration.type)
        )

    return cls(
        configuration=configuration,
        detector=detector
    )
