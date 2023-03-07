from ananke.configurations.collection import HDF5StorageConfiguration
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
            number_of_samples=10
        ),
    ],
    storage=HDF5StorageConfiguration(
        data_path='data/electrical_noise_10.h5',
        read_only=False
    )
)

mock_collection = generate(dataset_configuration)

mock_collection.open()
mock_collection.storage.get_detector()