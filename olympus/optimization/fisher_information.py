import awkward as ak
import jax
import numpy as np
from jax import numpy as jnp
from jax import random

from ..event_generation.event_generation import (
    generate_cascade,
    generate_muon_energy_losses,
    generate_realistic_track,
    simulate_noise,
    make_double_casc_source,
    make_single_casc_source,
)
from ..event_generation.utils import proposal_setup, sph_to_cart_jnp
from ..utils import rotate_to_new_direc_v
from ..event_generation.constants import Constants
from functools import partial


def pad_event(event):

    pad_len = np.int32(np.ceil(ak.max(ak.count(event, axis=1)) / 256) * 256)

    if ak.max(ak.count(event, axis=1)) > pad_len:
        raise RuntimeError()
    padded = ak.pad_none(event, target=pad_len, clip=True, axis=1)
    # mask = ak.is_none(padded, axis=1)
    ev_np = np.asarray((ak.fill_none(padded, np.inf)))
    return ev_np


def pad_array_log_bucket(array, base):
    if ak.count(array) == 0:
        return np.array([np.inf], dtype=np.float)

    log_cnt = np.log(ak.count(array)) / np.log(base)
    pad_len = int(np.power(base, np.ceil(log_cnt)))
    if ak.count(array) > pad_len:
        raise RuntimeError()

    padded = ak.pad_none(array, target=pad_len, clip=True, axis=0)
    ev_np = np.asarray((ak.fill_none(padded, np.inf)))
    return ev_np


def calc_fisher_info_cascades(
    det,
    event_data,
    seed,
    converter,
    ph_prop,
    llhobj,
    noise_window_len,
    n_ev=20,
    pad_base=4,
    mode="full",
):
    key = random.PRNGKey(seed)
    rng = np.random.RandomState(seed)

    def eval_for_mod(
        x,
        y,
        z,
        theta,
        phi,
        t,
        log10e,
        times,
        counts,
        mod_coords,
        mod_eff,
        noise_rate,
        key,
    ):

        pos = jnp.asarray([x, y, z])
        dir = sph_to_cart_jnp(theta, phi)

        sources = converter(
            pos, t, dir, 10**log10e, particle_id=event_data["particle_id"], key=key
        )

        # Call signature is the same for tfirst and full
        shape_lh, counts_lh = llhobj.per_module_full_llh(
            times,
            counts,
            sources[0],
            sources[1],
            sources[2],
            sources[3],
            mod_coords,
            noise_rate,
            mod_eff,
        )

        finite_times = jnp.isfinite(times)
        summed = (shape_lh * finite_times).sum() + counts_lh
        return summed

    def eval_for_mod_tfirst(
        x,
        y,
        z,
        theta,
        phi,
        t,
        log10e,
        times,
        counts,
        mod_coords,
        mod_eff,
        noise_rate,
        key,
    ):

        pos = jnp.asarray([x, y, z])
        dir = sph_to_cart_jnp(theta, phi)

        sources = converter(
            pos, t, dir, 10**log10e, particle_id=event_data["particle_id"], key=key
        )

        shape_lh, counts_lh = llhobj.per_module_tfirst_llh(
            times,
            counts,
            sources[0],
            sources[1],
            sources[2],
            sources[3],
            mod_coords,
            noise_rate,
            mod_eff,
        )

        finite_times = jnp.isfinite(times)
        summed = (shape_lh.squeeze() * finite_times).sum() + counts_lh
        return summed

    def eval_for_mod_counts(
        x,
        y,
        z,
        theta,
        phi,
        t,
        log10e,
        _,
        counts,
        mod_coords,
        mod_eff,
        noise_rate,
        key,
    ):

        pos = jnp.asarray([x, y, z])
        dir = sph_to_cart_jnp(theta, phi)

        sources = converter(
            pos, t, dir, 10**log10e, particle_id=event_data["particle_id"], key=key
        )

        # Call signature is the same for tfirst and full
        counts_lh = llhobj.per_module_poisson_llh_for_sources(
            counts,
            mod_coords,
            noise_rate,
            mod_eff,
            sources[0],
            sources[1],
            sources[2],
            sources[3],
        )
        return counts_lh

    eval_jacobian = jax.jit(jax.jacobian(eval_for_mod, [0, 1, 2, 3, 4, 5, 6]))
    eval_jacobian_tfirst = jax.jit(
        jax.jacobian(eval_for_mod_tfirst, [0, 1, 2, 3, 4, 5, 6])
    )
    eval_jacobian_counts = jax.jit(
        jax.jacobian(eval_for_mod_counts, [0, 1, 2, 3, 4, 5, 6])
    )

    matrices = []
    for _ in range(n_ev):
        key, k1, k2 = random.split(key, 3)
        event, _ = generate_cascade(
            det,
            event_data,
            pprop_func=ph_prop,
            seed=k1,
            converter_func=converter,
        )

        event, _ = simulate_noise(det, event, noise_window_len, rng)

        jacsum = 0
        counts = np.asarray(ak.count(event, axis=1))
        for j in range(len(event)):
            if (mode == "counts") or (len(event[j]) == 0):
                eval_func = eval_jacobian_counts
            elif mode == "tfirst":
                eval_func = eval_jacobian_tfirst
            else:
                eval_func = eval_jacobian

            if mode == "full":
                padded = pad_array_log_bucket(event[j], pad_base)
            elif mode == "tfirst":
                if len(event[j]) == 0:
                    padded = jnp.asarray([])
                else:
                    padded = float(ak.min(event[j]))
            else:
                padded = jnp.asarray([])
            res = jnp.stack(
                eval_func(
                    event_data["pos"][0],
                    event_data["pos"][1],
                    event_data["pos"][2],
                    event_data["theta"],
                    event_data["phi"],
                    event_data["time"],
                    np.log10(event_data["energy"]),
                    padded,
                    counts[j],
                    det.module_coords[j],
                    det.module_efficiencies[j],
                    det.module_noise_rates[j],
                    k2,
                )
            )
            jacsum += res
        matrices.append(np.asarray(jacsum[:, np.newaxis] * jacsum[np.newaxis, :]))

    fisher = np.average(np.stack(matrices), axis=0)
    return fisher


def calc_fisher_info_double_cascades(
    det,
    event_data,
    seed,
    converter,
    ph_prop,
    llhobj,
    noise_window_len,
    n_ev=20,
    pad_base=4,
    mode="full",
):
    def make_sources(x, y, z, theta, phi, t, log10e1, log10e2, log10separation, key):

        ev_data = {
            "pos": jnp.asarray([x, y, z]),
            "theta": theta,
            "phi": phi,
            "energy": 10**log10e1,
            "energy2": 10**log10e2,
            "separation": 10**log10separation,
            "time": t,
            "particle_id": 11,
        }
        ev_data["dir"] = sph_to_cart_jnp(theta, phi)

        sources = make_double_casc_source(ev_data, converter, key)

        return sources

    def eval_for_mod(
        x,
        y,
        z,
        theta,
        phi,
        t,
        log10e1,
        log10e2,
        logseparation,
        times,
        counts,
        mod_coords,
        mod_eff,
        noise_rate,
        key,
    ):

        sources = make_sources(
            x, y, z, theta, phi, t, log10e1, log10e2, logseparation, key
        )

        # Call signature is the same for tfirst and full
        shape_lh, counts_lh = llhobj.per_module_full_llh(
            times,
            counts,
            sources[0],
            sources[1],
            sources[2],
            sources[3],
            mod_coords,
            noise_rate,
            mod_eff,
        )

        finite_times = jnp.isfinite(times)
        summed = (shape_lh * finite_times).sum() + counts_lh
        return summed

    def eval_for_mod_tfirst(
        x,
        y,
        z,
        theta,
        phi,
        t,
        log10e1,
        log10e2,
        logseparation,
        times,
        counts,
        mod_coords,
        mod_eff,
        noise_rate,
        key,
    ):

        sources = make_sources(
            x, y, z, theta, phi, t, log10e1, log10e2, logseparation, key
        )

        shape_lh, counts_lh = llhobj.per_module_tfirst_llh(
            times,
            counts,
            sources[0],
            sources[1],
            sources[2],
            sources[3],
            mod_coords,
            noise_rate,
            mod_eff,
        )

        finite_times = jnp.isfinite(times)
        summed = (shape_lh.squeeze() * finite_times).sum() + counts_lh
        return summed

    def eval_for_mod_counts(
        x,
        y,
        z,
        theta,
        phi,
        t,
        log10e1,
        log10e2,
        logseparation,
        _,
        counts,
        mod_coords,
        mod_eff,
        noise_rate,
        key,
    ):

        sources = make_sources(
            x, y, z, theta, phi, t, log10e1, log10e2, logseparation, key
        )

        # Call signature is the same for tfirst and full
        counts_lh = llhobj.per_module_poisson_llh_for_sources(
            counts,
            mod_coords,
            noise_rate,
            mod_eff,
            sources[0],
            sources[1],
            sources[2],
            sources[3],
        )
        return counts_lh

    key = random.PRNGKey(seed)
    rng = np.random.RandomState(seed)

    eval_jacobian = jax.jit(jax.jacobian(eval_for_mod, list(range(9))))
    eval_jacobian_tfirst = jax.jit(jax.jacobian(eval_for_mod_tfirst, list(range(9))))
    eval_jacobian_counts = jax.jit(jax.jacobian(eval_for_mod_counts, list(range(9))))

    matrices = []
    for _ in range(n_ev):
        key, k1, k2 = random.split(key, 3)

        sources = make_sources(
            event_data["pos"][0],
            event_data["pos"][1],
            event_data["pos"][2],
            event_data["theta"],
            event_data["phi"],
            event_data["time"],
            np.log10(event_data["energy"]),
            np.log10(event_data["energy2"]),
            np.log10(event_data["separation"]),
            k1,
        )

        event = ph_prop(
            det.module_coords,
            det.module_efficiencies,
            sources[0],
            sources[1],
            sources[2],
            sources[3],
            seed=k2,
        )
        event, _ = simulate_noise(det, event, noise_window_len, rng)

        jacsum = 0
        counts = np.asarray(ak.count(event, axis=1))
        for j in range(len(event)):
            if (mode == "counts") or (len(event[j]) == 0):
                eval_func = eval_jacobian_counts
            elif mode == "tfirst":
                eval_func = eval_jacobian_tfirst
            else:
                eval_func = eval_jacobian

            if mode == "full":
                padded = pad_array_log_bucket(event[j], pad_base)
            elif mode == "tfirst":
                if len(event[j]) == 0:
                    padded = jnp.asarray([])
                else:
                    padded = float(ak.min(event[j]))
            else:
                padded = jnp.asarray([])
            res = jnp.stack(
                eval_func(
                    event_data["pos"][0],
                    event_data["pos"][1],
                    event_data["pos"][2],
                    event_data["theta"],
                    event_data["phi"],
                    event_data["time"],
                    np.log10(event_data["energy"]),
                    np.log10(event_data["energy2"]),
                    np.log10(event_data["separation"]),
                    padded,
                    counts[j],
                    det.module_coords[j],
                    det.module_efficiencies[j],
                    det.module_noise_rates[j],
                    k2,
                )
            )
            jacsum += res
        if jnp.any(jacsum == 0):
            raise RuntimeError("Got zero grad")
        matrices.append(np.asarray(jacsum[:, np.newaxis] * jacsum[np.newaxis, :]))

    fisher = np.average(np.stack(matrices), axis=0)
    return fisher


def calc_fisher_info_tracks(det, event_data, key, ph_prop, lh_func, c_medium):
    def make_wrap_lh_call(event, source_pos, source_dir, source_time, source_nphotons):

        ref_track_dir = sph_to_cart_jnp(event_data["theta"], event_data["phi"])

        def wrap_lh_call(x, y, z, theta, phi, time):

            new_track_dir = sph_to_cart_jnp(theta, phi)
            pos = jnp.asarray([x, y, z])

            old_pos_rel = source_pos - pos
            dist_along = old_pos_rel[:, 0] / ref_track_dir[0]

            new_source_pos = (
                pos[np.newaxis, :]
                + new_track_dir[np.newaxis, :] * dist_along[:, np.newaxis]
            )

            new_source_dir = rotate_to_new_direc_v(
                ref_track_dir, new_track_dir, source_dir
            )

            new_source_time = source_time - event_data["t0"] + time

            return lh_func(
                event,
                det.module_coords,
                new_source_pos,
                new_source_dir,
                new_source_time,
                source_nphotons,
                c_medium,
            )

        return wrap_lh_call

    event_dir = sph_to_cart_jnp(event_data["theta"], event_data["phi"])
    matrices = []

    prop = proposal_setup()

    for i in range(20):
        key, subkey = random.split(key)

        sources, prop_dist = generate_muon_energy_losses(
            prop,
            event_data["energy"],
            300,
            event_data["position"],
            event_dir,
            event_data["time"],
        )

        event = ph_prop(
            det.module_coords,
            det.module_efficiencies,
            sources,
            seed=subkey,
            c_medium=c_medium,
        )

        wrap_lh_call = make_wrap_lh_call(event, sources)

        jac = jax.jacobian(wrap_lh_call, argnums=list(range(6)))(
            event_data["pos"][0],
            event_data["pos"][1],
            event_data["pos"][2],
            event_data["theta"],
            event_data["phi"],
            event_data["t0"],
        )

        jac = jnp.stack(jac)[:, np.newaxis]
        matrices.append(jac * jac.T)
    matrix = jnp.average(jnp.stack(matrices), axis=0)
    return matrix
