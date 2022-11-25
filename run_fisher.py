import os
import argparse
import pickle

from ananke.schemas.detector import DetectorConfiguration
from olympus.event_generation.utils import sph_to_cart_jnp

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"
import functools

import numpy as np
import jax.numpy as jnp
from olympus.event_generation.lightyield import (
    make_realistic_cascade_source,
)
from olympus.event_generation.photon_propagation.norm_flow_photons import (
    make_generate_norm_flow_photons,
    NormFlowPhotonLHPerModule,
)
from olympus.optimization.fisher_information import (
    calc_fisher_info_cascades,
    calc_fisher_info_double_cascades,
    calc_fisher_info_tracks,
)

from hyperion.constants import Constants
from hyperion.medium import cascadia_ref_index_func

from olympus.event_generation.detector import (
    sample_cylinder_surface,
    sample_cylinder_volume,
    sample_direction, DetectorBuilder,
)

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--spacing", type=float, required=True)
parser.add_argument("--pmts", type=int, required=True)
parser.add_argument("-o", "--outfile", type=str, required=True)
parser.add_argument("--shape_model", type=str, required=True)
parser.add_argument("--counts_model", type=str, required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--mode", choices=["full", "counts", "tfirst"], required=True)
parser.add_argument("--pad_base", default=8, type=int, required=False)
parser.add_argument("--nev", default=100, type=int, required=False)
parser.add_argument("--nsamples", default=100, type=int, required=False)
parser.add_argument("--det", choices=["triangle", "cluster", "single"], required=True)

subparsers = parser.add_subparsers(help="event type", dest="ev_type")
parser_dbl_casc = subparsers.add_parser("double_casc")
parser_dbl_casc.add_argument("-e", "--energy", type=float, required=True)
# parser_dbl_casc.add_argument("--energy2", type=float, required=True)
# parser_dbl_casc.add_argument("--separation", type=float, required=True)

parser_casc = subparsers.add_parser("casc")
parser_casc.add_argument("-e", "--energy", type=float, required=True)

parser_casc = subparsers.add_parser("track")
parser_casc.add_argument("-e", "--energy", type=float, required=True)

args = parser.parse_args()

ref_index_func = cascadia_ref_index_func


def c_medium_f(wl):
    """Speed of light in medium for wl (nm)."""
    return Constants.BaseConstants.c_vac / cascadia_ref_index_func(wl)


dark_noise_rate = args.pmts * 1e4 * 1e-9  # 1/ns

gen_ph = make_generate_norm_flow_photons(
    args.shape_model, args.counts_model, c_medium=c_medium_f(700) / 1e9
)

noise_window_len = 2000  # ns

llhobj = NormFlowPhotonLHPerModule(
    args.shape_model,
    args.counts_model,
    noise_window_len=noise_window_len,
    c_medium=c_medium_f(700) * 1e-9,
)

pmts_per_module = args.pmts
pmt_cath_area_r = 75e-3 / 2  # m
module_radius = 0.21  # m

# Calculate the relative area covered by PMTs
efficiency = pmts_per_module * (pmt_cath_area_r) ** 2 * np.pi / (
        4 * np.pi * module_radius ** 2)

detector_builder = DetectorBuilder()
detector_configuration_dict = {
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
}
rng = np.random.RandomState(args.seed)
if args.det == "triangle":
    detector_configuration_dict["geometry"] = {
        "type": "triangular",
        "side_length": args.spacing,
    }
    detector_configuration = \
        DetectorConfiguration.parse_obj(detector_configuration_dict)
    det = detector_builder.get(configuration=detector_configuration)
elif args.det == "cluster":
    detector_configuration_dict["geometry"] = {
        "type": "hexagonal",
        "number_of_strings_per_side": 3,
        "distance_between_strings": args.spacing,
    }
    detector_configuration = \
        DetectorConfiguration.parse_obj(detector_configuration_dict)
    det = detector_builder.get(configuration=detector_configuration)
else:
    raise NotImplementedError()
radius, height = det.outer_cylinder

res_dicts = []
for _ in range(args.nev):

    if args.ev_type == "casc":

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

        call_func = calc_fisher_info_cascades

    elif args.ev_type == "track":

        event_pos = sample_cylinder_surface(height, radius, 1, rng).squeeze()
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

        call_func = calc_fisher_info_tracks

    elif args.ev_type == "double_casc":

        event_pos = sample_cylinder_volume(height, radius, 1, rng).squeeze()
        event_dir = sample_direction(1, rng).squeeze()

        theta = np.arccos(event_dir[2])
        phi = np.arccos(event_dir[0] / np.sin(theta))

        inelas = 10 ** rng.uniform(-2, 0)
        energy = args.energy * inelas
        energy2 = args.energy * (1 - inelas)

        tau_mass = 1.77686
        tau_tau = 290.3e-15
        gamma = energy2 / tau_mass

        separation = tau_tau * gamma * Constants.BaseConstants.c_vac

        event_data = {
            "time": 0.0,
            "theta": theta,
            "phi": phi,
            "pos": event_pos,
            "energy": energy,
            "energy2": energy2,
            "separation": separation,
            "particle_id": 11,
        }
        event_data["dir"] = sph_to_cart_jnp(event_data["theta"], event_data["phi"])
        # print(event_record)
        call_func = calc_fisher_info_double_cascades

    converter = functools.partial(
        make_realistic_cascade_source, resolution=0.3, moliere_rand=True
    )
    ph_prop = gen_ph

    fisher = call_func(
        det,
        event_data,
        args.seed,
        converter,
        gen_ph,
        llhobj,
        noise_window_len=noise_window_len,
        n_ev=args.nsamples,
        pad_base=args.pad_base,
        mode=args.mode,
    )

    result_dict = {
        key: np.asarray(val) if isinstance(val, jnp.DeviceArray) else val
        for key, val in event_data.items()
    }
    result_dict["fisher"] = np.asarray(fisher)
    result_dict["pmts"] = args.pmts
    result_dict["spacing"] = args.spacing
    result_dict["det"] = args.det
    result_dict["mode"] = args.mode
    result_dict["ev_type"] = args.ev_type

    res_dicts.append(result_dict)

pickle.dump(
    res_dicts,
    open(args.outfile, "wb"),
)
