"""Module containing all configuration for photon propagation."""
from enum import Enum
from typing import Literal

from pydantic import BaseModel, NonNegativeFloat, NonNegativeInt

from ..event_generation.medium import MediumEstimationVariant
from ..constants import defaults


class PhotonPropagators(str, Enum):
    """Possible photon propagators.

    Note that the enum is built in a way that the keys mimick the class name of the
    generator without the 'PhotonPropagator' term at the end for lean code.
    """

    Mock = 'mock'

    NormalFlow = 'normal_flow'


class PhotonPropagatorConfiguration(BaseModel):
    """Configuration for the Event Propagators."""

    #: Type of the Photon Propagation
    type: str

    medium: MediumEstimationVariant = MediumEstimationVariant.PONE_OPTIMISTIC

    #: default wavelength to use
    default_wavelength: NonNegativeFloat = 450

    #: seed for the photon propagator
    seed: int = defaults['seed']


class MockPhotonPropagatorConfiguration(PhotonPropagatorConfiguration):
    """Configuration for the Mock Photon Propagator."""

    #: Type of the Photon Propagation
    type: Literal[
        PhotonPropagators.Mock,
    ] = PhotonPropagators.Mock

    resolution: NonNegativeInt = 18000


class NormalFlowPhotonPropagatorConfiguration(PhotonPropagatorConfiguration):
    """Configuration for the Normal Flow Photon Propagator."""

    #: Type of the Photon Propagation
    type: Literal[
        PhotonPropagators.NormalFlow,
    ] = PhotonPropagators.NormalFlow

    #: path of the shape model
    shape_model_path: str

    #: path of the counts model
    counts_model_path: str

    #: Logarithmic base to calculate bucket size when compiling the sampling function.
    padding_base: NonNegativeInt = 4
