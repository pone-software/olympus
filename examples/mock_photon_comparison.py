import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"  # add this
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "\"platform\""

import jax

# Global flag to set a specific platform, must be used at startup.
jax.config.update('jax_platform_name', 'cpu')

from ananke.configurations.detector import DetectorConfiguration
from ananke.services.detector import DetectorBuilderService
from olympus.configuration.generators import GenerationConfiguration
from olympus.configuration.photon_propagation import (
    MockPhotonPropagatorConfiguration,
    NormalFlowPhotonPropagatorConfiguration,
)
from olympus.event_generation.generators import get_generator
from olympus.event_generation.photon_propagation.mock_photons import MockPhotonPropagator
from olympus.event_generation.photon_propagation.norm_flow_photons import NormalFlowPhotonPropagator

oms_per_line = 20
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

configuration = GenerationConfiguration.parse_obj({
    'generator': {
        'type': 'cascade',
        'spectrum': {
            'log_minimal_energy': 2.0,
            'log_maximal_energy': 5.5,
        }
    },
    'number_of_samples': 2
})

cascades_generator = get_generator(
    detector=det,
    configuration=configuration.generator
)

records = cascades_generator.generate_records(
    number_of_samples=4
)

sources = cascades_generator.propagate(records)

mock_photon_propagator_configuration = MockPhotonPropagatorConfiguration(resolution=18000)

mock_photon_propagator = MockPhotonPropagator(
    detector=det,
    configuration=mock_photon_propagator_configuration
)

normal_photon_propagator_configuration = NormalFlowPhotonPropagatorConfiguration(
    shape_model_path="../../hyperion/data/photon_arrival_time_nflow_params.pickle",
    counts_model_path="../../hyperion/data/photon_arrival_time_counts_params.pickle"
)

normal_photon_propagator = NormalFlowPhotonPropagator(
    detector=det,
    configuration=normal_photon_propagator_configuration
)

mock_hits = mock_photon_propagator.propagate(records, sources)

mock_hits.df.head()

normal_hits = normal_photon_propagator.propagate(records, sources)
normal_hits.df.head()