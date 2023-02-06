from ananke.configurations.presets.detector import single_line_configuration
from ananke.models.collection import Collection, CollectionExporters
from ananke.schemas.event import NoiseType
from olympus.configuration.generators import (
    NoiseGeneratorConfiguration,
    DatasetConfiguration, GenerationConfiguration,
)
from olympus.event_generation.generators import (
    generate
)

collection = Collection(data_path='../../data/new/electrical_noise_2000/data.h5')

collection.export(export_path='../../data/new/electrical_noise_2000_graph_net', exporter=CollectionExporters.GRAPH_NET)
