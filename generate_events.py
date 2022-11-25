from argparse import ArgumentParser
import os
import pickle
import numpy as np

from ananke.schemas.detector import DetectorConfiguration
from olympus.event_generation.detector import DetectorBuilder

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"

from functools import partial
import json

from hyperion.constants import Constants
from hyperion.medium import medium_collections
from olympus.event_generation.event_generation import (
    generate_cascades,
    generate_realistic_starting_tracks,
    generate_realistic_tracks,
)
from olympus.event_generation.lightyield import (
    make_realistic_cascade_source,
)
from olympus.event_generation.photon_propagation.norm_flow_photons import (
    make_generate_norm_flow_photons,
)

from olympus.event_generation.utils import proposal_setup

parser = ArgumentParser()
parser.add_argument(
    "--type", choices=["cascade", "track", "starting_track"], required=True
)
parser.add_argument("-s", "--seed", required=True, type=int)
parser.add_argument("-n", "--n_events", required=True, type=int)
parser.add_argument("-o", "--outfile", required=True, type=str)
parser.add_argument("--shape-model", required=True, dest="shape_model")
parser.add_argument("--counts-model", required=True, dest="counts_model")
parser.add_argument("--config", required=True, dest="config")

args = parser.parse_args()

config = json.load(open(args.config))["photon_propagation"]
ref_ix_f, sca_a_f, sca_l_f, _ = medium_collections[config["medium"]]


def c_medium_f(wl):
    """Speed of light in medium for wl (nm)."""
    return Constants.BaseConstants.c_vac / ref_ix_f(wl)


dark_noise_rate = 16 * 1e4 * 1e-9  # 1/ns

pmts_per_module = 16
pmt_cath_area_r = 75e-3 / 2  # m
module_radius = 0.21  # m

# Calculate the relative area covered by PMTs
efficiency = pmts_per_module * (pmt_cath_area_r) ** 2 * np.pi / (
            4 * np.pi * module_radius ** 2)

detector_configuration = DetectorConfiguration.parse_obj({
    "string": {
        "module_number": 20,
        "module_distance": 50
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
        "type": "triangular",
        "side_length": 100,
    },
    "seed": 31338
})

detector_service = DetectorBuilder()
det = detector_service.get(configuration=config)
# det = Detector(make_line(0, 0, 20, 50, rng, dark_noise_rate, 0, efficiency=efficiency))

gen_ph = make_generate_norm_flow_photons(
    args.shape_model,
    args.counts_model,
    c_medium=c_medium_f(700) / 1e9,
)

if args.type == "cascade":
    events, records = generate_cascades(
        det,
        cylinder_height=det.outer_cylinder[1] + 100,
        cylinder_radius=200,
        log_emin=2,
        log_emax=6,
        nsamples=args.n_events,
        seed=args.seed,
        particle_id=11,
        pprop_func=gen_ph,
        noise_function=None,
        converter_func=partial(
            make_realistic_cascade_source, moliere_rand=True, resolution=0.2
        ),
    )
elif args.type == "track":
    events, records = generate_realistic_tracks(
        det,
        cylinder_height=det.outer_cylinder[1] + 100,
        cylinder_radius=200,
        log_emin=2,
        log_emax=6,
        nsamples=args.n_events,
        seed=args.seed,
        pprop_func=gen_ph,
        noise_function=None,
        proposal_prop=proposal_setup(),
    )

elif args.type == "starting_track":
    events, records = generate_realistic_starting_tracks(
        det,
        cylinder_height=det._outer_cylinder[1] + 100,
        cylinder_radius=200,
        log_emin=2,
        log_emax=6,
        nsamples=args.n_events,
        seed=args.seed,
        pprop_func=gen_ph,
        noise_function=None,
        proposal_prop=proposal_setup(),
    )


pickle.dump((events, records), open(args.outfile, "wb"))
