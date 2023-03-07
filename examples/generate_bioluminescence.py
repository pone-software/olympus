import logging

from ananke.configurations.collection import HDF5StorageConfiguration
from ananke.configurations.presets.detector import single_line_configuration
from ananke.schemas.event import NoiseType
from olympus.configuration.generators import (
    DatasetConfiguration,
    GenerationConfiguration,
    BioluminescenceGeneratorConfiguration,
)
from olympus.event_generation.generators import generate

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
            number_of_samples=10
        )
    ],
    storage=HDF5StorageConfiguration(
        data_path='data/bioluminescence_10.h5',
        read_only=False
    )
)

collection = generate(configuration)

collection.open()
collection.storage.get_records()
collection.close()