from ananke.configurations.presets.detector import single_line_configuration
from ananke.schemas.event import NoiseType
from olympus.configuration.generators import (
    DatasetConfiguration,
    GenerationConfiguration,
    UniformSpectrumConfiguration,
    BioluminescenceGeneratorConfiguration,
)
from olympus.configuration.photon_propagation import MockPhotonPropagatorConfiguration
from olympus.event_generation.generators import generate
import logging

from olympus.event_generation.medium import MediumEstimationVariant

logging.getLogger().setLevel(logging.INFO)

configuration = DatasetConfiguration(
    detector=single_line_configuration,
    generators=[
        GenerationConfiguration(
            generator=BioluminescenceGeneratorConfiguration(
                type=NoiseType.BIOLUMINESCENCE,
                start_time=0,
                duration=1000,
                julia_data_path='../../data/biolumi_sims',
                batch_size= 48
            ),
            number_of_samples=100000
        )
    ],
    data_path="data/bioluminescence_100000"
)

collection = generate(configuration)