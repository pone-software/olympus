"""Module containing all the configurations for generating events."""
from enum import Enum
from typing import Literal, Union, List, Optional

from pydantic import (
    BaseModel,
    Field,
    NonNegativeFloat,
    validator, NonNegativeInt,
)

from ananke.configurations.detector import DetectorConfiguration
from ananke.schemas.event import EventType, NoiseType
from .photon_propagation import (
    MockPhotonPropagatorConfiguration,
    NormalFlowPhotonPropagatorConfiguration,
)
from ..constants import defaults


class GeneratorConfiguration(BaseModel):
    """Configuration for a generator."""

    #: seed for the generator
    seed: int = defaults['seed']

    #: type of the configuration
    type: str

    #: For debugging, it might be useful to have similar UUIDs all the time
    fix_uuids: bool = False


class EventPropagatorConfiguration(BaseModel):
    """Configuration for event propagators."""

    #: resolution for the event propagators
    resolution: float = 0.2


class UniformSpectrumConfiguration(BaseModel):
    """Configuration for spectrums of the propagators."""

    #: minimal energy of the event
    log_minimal_energy: NonNegativeFloat

    #: maximal energy of the event
    log_maximal_energy: NonNegativeFloat

    @validator('log_minimal_energy')
    def minimal_energy_validation(cls, log_minimal_energy, values, **kwargs):
        """Checks whether minimal energy is smaller than maximal."""
        if 'log_maximal_energy' in values and \
                log_minimal_energy > values['log_maximal_energy']:
            raise ValueError('minimal energy is greater than maximal.')
        return log_minimal_energy

    @validator('log_maximal_energy')
    def maximal_energy_validation(cls, log_maximal_energy, values, **kwargs):
        """Checks whether maximal energy is higher than minimal."""
        if 'log_minimal_energy' in values and \
                log_maximal_energy < values['log_minimal_energy']:
            raise ValueError('maximal energy is smaller than minimal.')
        return log_maximal_energy


class EventGeneratorConfiguration(GeneratorConfiguration):
    """Configuration for event generators."""

    #: type of the configuration
    type: Literal[
        EventType.REALISTIC_TRACK,
        EventType.STARTING_TRACK,
        EventType.CASCADE,
    ]

    # TODO: Write good documentation
    #: particle ID of the event
    particle_id: int = 11

    #: starting time of the events
    start_time: float = 0.0

    # TODO: Check whether this gets more beautiful
    #: Configuration for the event propagators
    event_propagator: EventPropagatorConfiguration = \
        Field(default_factory=lambda: EventPropagatorConfiguration())

    source_propagator: Union[
        MockPhotonPropagatorConfiguration,
        NormalFlowPhotonPropagatorConfiguration,
    ] = Field(
        default_factory=lambda: MockPhotonPropagatorConfiguration(),
        discriminator='type',
    )

    # HINT: Adapt to others once more spectrums are coming
    #: Spectrum of the events to generate
    spectrum: UniformSpectrumConfiguration


class NoiseGeneratorConfiguration(GeneratorConfiguration):
    """Class for noise generators"""

    #: type of noise generator
    type: Literal[
        NoiseType.ELECTRICAL,
    ]

    #: Time when the noise should start
    start_time: float

    #: Duration of the noise interval
    duration: NonNegativeFloat


class GenerationConfiguration(BaseModel):
    """Configuration for a single generation."""
    generator: Union[
        EventGeneratorConfiguration,
        NoiseGeneratorConfiguration,
    ] = Field(..., discriminator='type')

    #: Number of samples for this Generator
    number_of_samples: NonNegativeInt


class DatasetStatus(str, Enum):
    """Enum containing different collection generation value."""
    NOT_STARTED = 'not_started'
    STARTED = 'started'
    ERROR = 'error'
    COMPLETE = 'complete'


# TODO: Move to generation maybe?
class DatasetStatusConfiguration(BaseModel):
    """Configuration for the collection value."""
    value: DatasetStatus = DatasetStatus.NOT_STARTED
    error_message: Optional[str] = None
    current_index: NonNegativeInt = 0


class DatasetConfiguration(BaseModel):
    """Class to configure how to generate what."""

    detector: DetectorConfiguration

    #: Generator to generate from
    generators: List[GenerationConfiguration]

    #: Path for the generated data to be
    data_path: str

    status: DatasetStatusConfiguration = Field(
        default_factory=lambda: DatasetStatusConfiguration()
    )
