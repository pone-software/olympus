import awkward as ak
import jax
import numpy as np
from jax import numpy as jnp
from jax import random

from ..event_generation.event_generation import (
    generate_cascade,
    generate_muon_energy_losses,
    simulate_noise,
    make_double_casc_source,
)
from ..event_generation.utils import proposal_setup, sph_to_cart_jnp
from ..utils import rotate_to_new_direc_v
from olympus.constants import Constants


def pad_event(array):
    if ak.count(array) == 0:
        return np.array([np.inf], dtype=np.float)
    pad_len = np.int32(np.ceil(ak.count(array) / 256) * 256)

    if ak.count(array) > pad_len:
        raise RuntimeError()
    padded = ak.pad_none(array, target=pad_len, clip=True, axis=0)
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
    def make_sources(x, y, z, theta, phi, t, log10e1, log10e2, x2, y2, z2, key):

        ev_data = {
            "pos": jnp.asarray([x, y, z]),
            "theta": theta,
            "phi": phi,
            "energy": 10**log10e1,
            "energy2": 10**log10e2,
            "pos2": jnp.asarray([x2, y2, z2]),
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
        x2,
        y2,
        z2,
        times,
        counts,
        mod_coords,
        mod_eff,
        noise_rate,
        key,
    ):

        sources = make_sources(
            x, y, z, theta, phi, t, log10e1, log10e2, x2, y2, z2, key
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
        x2,
        y2,
        z2,
        times,
        counts,
        mod_coords,
        mod_eff,
        noise_rate,
        key,
    ):

        sources = make_sources(
            x, y, z, theta, phi, t, log10e1, log10e2, x2, y2, z2, key
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
        x2,
        y2,
        z2,
        _,
        counts,
        mod_coords,
        mod_eff,
        noise_rate,
        key,
    ):

        sources = make_sources(
            x, y, z, theta, phi, t, log10e1, log10e2, x2, y2, z2, key
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

    eval_jacobian = jax.jit(jax.jacobian(eval_for_mod, list(range(11))))
    eval_jacobian_tfirst = jax.jit(jax.jacobian(eval_for_mod_tfirst, list(range(11))))
    eval_jacobian_counts = jax.jit(jax.jacobian(eval_for_mod_counts, list(range(11))))

    matrices = []
    for _ in range(n_ev):
        key, k1, k2 = random.split(key, 3)

        pos2 = event_data["pos"] + event_data["separation"] * event_data["dir"]

        sources = make_sources(
            event_data["pos"][0],
            event_data["pos"][1],
            event_data["pos"][2],
            event_data["theta"],
            event_data["phi"],
            event_data["time"],
            np.log10(event_data["energy"]),
            np.log10(event_data["energy2"]),
            pos2[0],
            pos2[1],
            pos2[2],
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
                    pos2[0],
                    pos2[1],
                    pos2[2],
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


def calc_fisher_info_tracks(
    det,
    event_data,
    seed,
    _,
    ph_prop,
    llhobj,
    noise_window_len,
    n_ev=20,
    pad_base=4,
    mode="full",
):

    key = random.PRNGKey(seed)
    rng = np.random.RandomState(seed)
    prop = proposal_setup()

    def make_sources(event_data, key):
        backtrack_len = 200
        start_pos = event_data["pos"] - backtrack_len * event_data["dir"]
        start_time = event_data["time"] - backtrack_len / Constants.c_vac

        (
            base_source_pos,
            base_source_dir,
            base_source_time,
            base_source_photons,
            _,
        ) = generate_muon_energy_losses(
            prop,
            event_data["energy"],
            1500,
            start_pos,
            event_data["dir"],
            start_time,
            key,
        )

        # rotate source_directions to rel e_z
        base_source_dir = rotate_to_new_direc_v(
            event_data["dir"], np.asarray([0, 0, 1.0]), base_source_dir
        )

        return (base_source_pos, base_source_dir, base_source_time, base_source_photons)

    def update_sources(x, y, z, theta, phi, t, log10e1, base_sources):

        new_pos = jnp.asarray([x, y, z])
        new_dir = sph_to_cart_jnp(theta, phi)

        new_source_dirs = rotate_to_new_direc_v(
            np.asarray([0, 0, 1.0]), new_dir, base_sources[1]
        )
        new_source_pos = base_sources[0] + (new_pos - event_data["pos"])
        new_source_time = base_sources[2] + event_data["time"] - t
        new_source_photons = base_sources[3] / event_data["energy"] * 10**log10e1

        return new_source_pos, new_source_dirs, new_source_time, new_source_photons

    def eval_for_mod(
        x,
        y,
        z,
        theta,
        phi,
        t,
        log10e1,
        times,
        counts,
        mod_coords,
        mod_eff,
        noise_rate,
        base_sources,
    ):

        sources = update_sources(x, y, z, theta, phi, t, log10e1, base_sources)

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
        times,
        counts,
        mod_coords,
        mod_eff,
        noise_rate,
        base_sources,
    ):

        sources = update_sources(x, y, z, theta, phi, t, log10e1, base_sources)

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
        _,
        counts,
        mod_coords,
        mod_eff,
        noise_rate,
        base_sources,
    ):

        sources = update_sources(x, y, z, theta, phi, t, log10e1, base_sources)

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

    eval_jacobian = jax.jit(jax.jacobian(eval_for_mod, list(range(7))))
    eval_jacobian_tfirst = jax.jit(jax.jacobian(eval_for_mod_tfirst, list(range(7))))
    eval_jacobian_counts = jax.jit(jax.jacobian(eval_for_mod_counts, list(range(7))))

    matrices = []
    for _ in range(n_ev):
        key, k1, k2 = random.split(key, 3)

        sources = make_sources(
            event_data,
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
                padded = pad_event(event[j])
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
                    sources,
                )
            )
            jacsum += res
        if jnp.any(jacsum == 0):
            raise RuntimeError(f"Got zero grad: {jacsum}")
        matrices.append(np.asarray(jacsum[:, np.newaxis] * jacsum[np.newaxis, :]))

    fisher = np.average(np.stack(matrices), axis=0)
    return fisher
