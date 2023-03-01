# import os

# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"  # add this
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "\"platform\""
#
# import jax
#
# # Global flag to set a specific platform, must be used at startup.
# jax.config.update('jax_platform_name', 'cpu')


from olympus.configuration.generators import EventGeneratorConfiguration
from olympus.configuration.generators import GenerationConfiguration
from olympus.event_generation.medium import MediumEstimationVariant
from olympus.configuration.generators import UniformSpectrumConfiguration
from ananke.schemas.event import EventType
from olympus.configuration.photon_propagation import MockPhotonPropagatorConfiguration
from olympus.configuration.generators import DatasetConfiguration

from ananke.configurations.presets.detector import single_line_configuration
from olympus.event_generation.generators import generate

import logging

logging.getLogger().setLevel(logging.INFO)

photon_propagator_configuration = MockPhotonPropagatorConfiguration(
    resolution=18000,
    medium=MediumEstimationVariant.PONE_OPTIMISTIC,
    max_memory_usage=int(2147483648 / 4)
)

configuration = DatasetConfiguration(
    detector=single_line_configuration,
    generators=[
        GenerationConfiguration(
            generator=EventGeneratorConfiguration(
                type=EventType.REALISTIC_TRACK,
                spectrum=UniformSpectrumConfiguration(
                    log_minimal_energy=2.0,
                    log_maximal_energy=5.5
                ),
                source_propagator=photon_propagator_configuration
            ),
            number_of_samples=10000
        )
    ],
    data_path="../../data/realistic_track_10000"
)

collection = generate(configuration)
