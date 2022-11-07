import json
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from jax import numpy as jnp

from hyperion.constants import Constants
from hyperion.medium import medium_collections
from olympus.event_generation.detector import (
    make_triang,
)
from olympus.event_generation.generators import GeneratorCollection, GeneratorFactory
from olympus.event_generation.photon_propagation.norm_flow_photons import (
    make_generate_norm_flow_photons,
)

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"


path_to_config = "../../hyperion/data/pone_config_optimistic.json"
config = json.load(open(path_to_config))["photon_propagation"]
ref_ix_f, sca_a_f, sca_l_f, something_else = medium_collections[config["medium"]]


def c_medium_f(wl):
    """Speed of light in medium for wl (nm)."""
    return Constants.BaseConstants.c_vac / ref_ix_f(wl)


rng = np.random.RandomState(31338)
oms_per_line = 20
dist_z = 50  # m
dark_noise_rate = 16 * 1e-5  # 1/ns
side_len = 100  # m
pmts_per_module = 16
pmt_cath_area_r = 75e-3 / 2  # m
module_radius = 0.21  # m

# Calculate the relative area covered by PMTs
efficiency = (
        pmts_per_module * pmt_cath_area_r ** 2 * np.pi / (4 * np.pi * module_radius ** 2)
)
det = make_triang(
    side_len, oms_per_line, dist_z, dark_noise_rate, rng, efficiency=efficiency
)
module_positions = jnp.asarray(det.module_coords)

plt.scatter(module_positions[:, 0], module_positions[:, 1])
plt.xlabel("x [m]")
plt.ylabel("y [m]")

gen_ph = make_generate_norm_flow_photons(
    "../../hyperion/data/photon_arrival_time_nflow_params.pickle",
    "../../hyperion/data/photon_arrival_time_counts_params.pickle",
    c_medium=c_medium_f(700) / 1e9,
)

generator_factory = GeneratorFactory(det, gen_ph)

cascades_generator = generator_factory.create(
    "cascade", particle_id=11, log_minimal_energy=2, log_maximal_energy=5.5, rate=0.05
)

cascades_generator2 = generator_factory.create(
    "cascade", particle_id=11, log_minimal_energy=4, log_maximal_energy=5.5, rate=0.01
)

noise_generator = generator_factory.create("noise")

track_generator = generator_factory.create(
    'track',
    log_minimal_energy=2,
    log_maximal_energy=5.5,
    rate=0.02
)

generator_collection = GeneratorCollection(detector=det)

# generator_collection.add_generator(track_generator)
generator_collection.add_generator(cascades_generator)
generator_collection.add_generator(cascades_generator2)
generator_collection.add_generator(noise_generator)

event_collection = generator_collection.generate(
    start_time=0,
    end_time=100,
)


pickle.dump(event_collection, open('./dataset/test', "wb"))
