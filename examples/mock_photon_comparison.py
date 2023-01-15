from ananke.configurations.detector import DetectorConfiguration
from ananke.services.detector import DetectorBuilderService
from olympus.event_generation.generators import GeneratorCollection, GeneratorFactory
from olympus.event_generation.medium import MediumEstimationVariant, Medium
from olympus.event_generation.photon_propagation.mock_photons import MockPhotonPropagator
from olympus.event_generation.photon_propagation.norm_flow_photons import NormalFlowPhotonPropagator

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
        "seed": 31338
    }
)

detector_service = DetectorBuilderService()
det = detector_service.get(configuration=detector_configuration)

medium = Medium(MediumEstimationVariant.PONE_OPTIMISTIC)

generator_factory = GeneratorFactory(det)

cascades_generator = generator_factory.create(
    "cascade", particle_id=11, log_minimal_energy=2, log_maximal_energy=5.5, rate=0.05
)

records = cascades_generator.generate_records(
    number_of_samples=4
)

sources = cascades_generator.propagate(records)

mock_photon_propagator = MockPhotonPropagator(
    detector=det,
    medium=medium,
    angle_resolution=180,
)

normal_photon_propagator = NormalFlowPhotonPropagator(
    detector=det,
    shape_model_path="../../hyperion/data/photon_arrival_time_nflow_params.pickle",
    counts_model_path="../../hyperion/data/photon_arrival_time_counts_params.pickle",
    medium=medium
)

mock_hits = mock_photon_propagator.propagate(records, sources)

mock_hits.df.head()

normal_hits = normal_photon_propagator.propagate(records, sources)
normal_hits.df.head()