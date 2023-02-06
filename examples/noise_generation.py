from ananke.configurations.presets.detector import single_line_configuration
from ananke.schemas.event import NoiseType
from olympus.configuration.generators import (
    NoiseGeneratorConfiguration,
    DatasetConfiguration, GenerationConfiguration,
)
from olympus.event_generation.generators import (
    generate
)

noise_generator_config = NoiseGeneratorConfiguration(
    type=NoiseType.ELECTRICAL,
    start_time=0,
    duration=1000,
)

dataset_configuration = DatasetConfiguration(
    detector=single_line_configuration,
    generators=[
        GenerationConfiguration(
            generator=noise_generator_config,
            number_of_samples=2000
        ),
    ],
    data_path='../../data/new/electrical_noise_2000'
)

mock_collection = generate(dataset_configuration)
