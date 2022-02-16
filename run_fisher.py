import os
import argparse
import pickle

from olympus.event_generation.utils import sph_to_cart_jnp

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"
import functools

import numpy as np
from jax import random
from olympus.event_generation.detector import make_rhombus, make_triang
from olympus.event_generation.lightyield import (
    make_pointlike_cascade_source,
    make_realistic_cascade_source,
)
from olympus.event_generation.photon_propagation.norm_flow_photons import (
    make_generate_norm_flow_photons,
    make_nflow_photon_likelihood_per_module,
)
from olympus.optimization.fisher_information import calc_fisher_info_cascades

from hyperion.constants import Constants
from hyperion.medium import cascadia_ref_index_func, sca_len_func_antares
from hyperion.utils import make_cascadia_abs_len_func

from olympus.event_generation.detector import (
    generate_noise,
    sample_cylinder_surface,
    sample_cylinder_volume,
    sample_direction,
)

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--spacing", type=float, required=True)
parser.add_argument("-e", "--energy", type=float, required=True)
parser.add_argument("--pmts", type=int, required=True)
parser.add_argument("-o", "--outfile", type=str, required=True)
parser.add_argument("--shape_model", type=str, required=True)
parser.add_argument("--counts_model", type=str, required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--mode", choices=["full", "counts", "tfirst"], required=True)
parser.add_argument("--pad_base", default=4, type=int, required=False)

args = parser.parse_args()

ref_index_func = cascadia_ref_index_func
abs_len = make_cascadia_abs_len_func(sca_len_func_antares)


def c_medium_f(wl):
    """Speed of light in medium for wl (nm)."""
    return Constants.BaseConstants.c_vac / cascadia_ref_index_func(wl)


dark_noise_rate = 16 * 1e4 * 1e-9  # 1/ns

gen_ph = make_generate_norm_flow_photons(
    args.shape_model, args.counts_model, c_medium=c_medium_f(700) / 1e9
)

lh_per_mod = make_nflow_photon_likelihood_per_module(
    args.shape_model, args.counts_model, mode=args.mode
)
pmts_per_module = args.pmts
pmt_cath_area_r = 75e-3 / 2  # m
module_radius = 0.21  # m

# Calculate the relative area covered by PMTs
efficiency = (
    pmts_per_module * (pmt_cath_area_r) ** 2 * np.pi / (4 * np.pi * module_radius ** 2)
)

rng = np.random.RandomState(args.seed)
det = make_triang(args.spacing, 20, 50, dark_noise_rate, rng, efficiency=efficiency)
radius, height = det.outer_cylinder

event_pos = sample_cylinder_volume(height, radius, 1, rng).squeeze()
event_dir = sample_direction(1, rng).squeeze()

theta = np.arccos(event_dir[2])
phi = np.arccos(event_dir[0] / np.sin(theta))

event_data = {
    "time": 0.0,
    "theta": theta,
    "phi": phi,
    "pos": event_pos,
    "energy": args.energy,
    "particle_id": 11,
}

event_data["dir"] = sph_to_cart_jnp(event_data["theta"], event_data["phi"])


converter = functools.partial(
    make_realistic_cascade_source, resolution=0.3, moliere_rand=True
)
ph_prop = functools.partial(gen_ph)

fisher = calc_fisher_info_cascades(
    det,
    event_data,
    random.PRNGKey(args.seed),
    converter,
    gen_ph,
    lh_per_mod,
    c_medium=c_medium_f(700) / 1e9,
    n_ev=50,
    pad_base=args.pad_base,
)

pickle.dump(
    {
        "spacing": args.spacing,
        "energy": args.energy,
        "position": event_pos,
        "theta": theta,
        "phi": phi,
        "fisher": fisher,
        "pmts": args.pmts,
    },
    open(args.outfile, "wb"),
)
