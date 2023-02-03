from ananke.configurations.presets.detector import single_line_configuration
from ananke.schemas.event import NoiseType
from ananke.services.detector import DetectorBuilderService
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
    fix_uuids=True,
    seed=62225
)

noise_generator_config2 = NoiseGeneratorConfiguration(
    type=NoiseType.ELECTRICAL,
    start_time=0,
    duration=1000,
    fix_uuids=True,
    seed=652122
)

dataset_configuration = DatasetConfiguration(
    detector=single_line_configuration,
    generators=[
        GenerationConfiguration(
            generator=noise_generator_config,
            number_of_samples=5
        ),
        GenerationConfiguration(
            generator=noise_generator_config2,
            number_of_samples=7,
        ),
    ],
    data_path='data'
)

mock_collection = generate(dataset_configuration)


records = mock_collection.get_records()
hits = mock_collection.get_hits(-3889598616203030035)
