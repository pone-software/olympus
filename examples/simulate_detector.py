from asyncio import events
from datetime import datetime
import os
import json

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"

import matplotlib.pyplot as plt
import numpy as np

import seaborn as sns


from itertools import product

import awkward as ak
import pandas as pd

from olympus.event_generation.photon_propagation.norm_flow_photons import (
    make_generate_norm_flow_photons,
    make_nflow_photon_likelihood,
)
from olympus.event_generation.data import EventData
from olympus.event_generation.photon_propagation.utils import sources_to_model_input
from olympus.event_generation.detector import (
    make_hex_grid,
    Detector,
    make_line,
    make_triang,
    make_rhombus,
)
from olympus.event_generation.event_generation import (
    generate_cascade,
    generate_cascades,
    simulate_noise,
    generate_realistic_track,
    generate_realistic_tracks,
    generate_realistic_starting_tracks,
)
from olympus.event_generation.lightyield import (
    make_pointlike_cascade_source,
    make_realistic_cascade_source,
)
from olympus.event_generation.utils import sph_to_cart_jnp, proposal_setup
from olympus.plotting import plot_timeline
from olympus.event_generation.generators import GeneratorCollection, GeneratorFactory
from olympus.event_generation.propagator import CascadePropagator

from hyperion.medium import medium_collections
from hyperion.constants import Constants


from jax import random
from jax import numpy as jnp



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
    pmts_per_module * (pmt_cath_area_r) ** 2 * np.pi / (4 * np.pi * module_radius**2)
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

generator_collection = GeneratorCollection()

# generator_collection.add_generator(track_generator)
generator_collection.add_generator(cascades_generator)
generator_collection.add_generator(cascades_generator2)
generator_collection.add_generator(noise_generator)

events, records = generator_collection.generate(
    start_time=0,
    end_time=1000,
)

histogram = cascades_generator.generate_histogram(events, records)

np.save("histograms" + datetime.now().strftime("%d%m%Y-%H%M%S"), histogram[0])

# 1 redistribute events
# 2 stupidest network possible


# Vortrag

# Martin Dinkel

# 1 P-One What is it, module structure, challenges (bioluminecense) 2-3 min
# 2 Simulation Framework 3 min
# 3 Detector Optimization 3 min
# 4 Trigger Algorithm (FPGA Trigger, Bell Labs)
# 5 Single detector explanation 3 min
# 6 Multi Detector Explanation 3 min
# 7 Future perspective FPGA Trigger, Multi Parameter etc.