from ananke.configurations.detector import DetectorConfiguration
from ananke.services.detector import DetectorBuilderService
from olympus.event_generation.generators import (
    ElectronicNoiseGenerator,
)

oms_per_line = 3
dist_z = 50  # m
dark_noise_rate = 16 * 1e-5  # 1/ns
side_len = 100  # m
pmts_per_module = 16
pmt_cath_area_r = 75e-3 / 2  # m
module_radius = 0.21  # m
efficiency = 0.42 # Christian S. Number

detector_configuration = DetectorConfiguration.parse_obj(
    {
        "string": {
            "module_number": oms_per_line,
            "module_distance": dist_z
        },
        "pmt": {
            "efficiency": efficiency,
            "noise_rate": dark_noise_rate,
            "area": pmt_cath_area_r
        },
        "module": {
            "radius": module_radius
        },
        "geometry": {
            "type": "single",
        },
        "seed": 31339
    }
)

detector_service = DetectorBuilderService()
det = detector_service.get(configuration=detector_configuration)

electronic_noise_generator = ElectronicNoiseGenerator(
    detector=det,
    start_time=0,
    duration=1000
)

mock_collection = electronic_noise_generator.generate(5)

mock_collection.hits.df.head()