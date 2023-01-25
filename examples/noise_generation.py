from ananke.configurations.presets.detector import single_line_configuration
from ananke.schemas.event import NoiseType
from ananke.services.detector import DetectorBuilderService
from olympus.configuration.generators import NoiseGeneratorConfiguration
from olympus.event_generation.generators import (
    get_generator
)

detector_service = DetectorBuilderService()
det = detector_service.get(configuration=single_line_configuration)

noise_generator_config = NoiseGeneratorConfiguration(
    type=NoiseType.ELECTRICAL,
    start_time=0,
    duration=1000
)

electronic_noise_generator = get_generator(
    detector=det,
    configuration=noise_generator_config
)

mock_collection = electronic_noise_generator.generate(5)

mock_collection.hits.df.head()