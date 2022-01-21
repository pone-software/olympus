import awkward as ak
import numpy as np
import jax
from jax import numpy as jnp
from jax import random
from jax.lax import Precision
from ..event_generation.event_generation import (
    generate_cascade,
    generate_realistic_track,
    generate_muon_energy_losses,
)
from olympus.event_generation.utils import proposal_setup
from olympus.event_generation.photon_propagation.utils import sources_to_array


def pad_event(event):

    pad_len = np.int32(np.ceil(ak.max(ak.count(event, axis=1)) / 256) * 256)
    print(f"Pad len: {pad_len}")

    if ak.max(ak.count(event, axis=1)) > pad_len:
        raise RuntimeError()
    padded = ak.pad_none(event, target=pad_len, clip=True, axis=1)
    # mask = ak.is_none(padded, axis=1)
    ev_np = np.asarray((ak.fill_none(padded, -np.inf)))
    return ev_np


def sph_to_cart_jnp(theta, phi=0):
    """Transform spherical to cartesian coordinates."""
    x = jnp.sin(theta) * jnp.cos(phi)
    y = jnp.sin(theta) * jnp.sin(phi)
    z = jnp.cos(theta)

    return jnp.asarray([x, y, z])


def calc_fisher_info_cascades(
    det, event_data, key, converter, ph_prop, lh_func, c_medium, n_ev=20
):
    def make_wrap_lh_call(event):
        def wrap_lh_call(x, y, z, theta, phi, time, log10_energy):
            event_dir = sph_to_cart_jnp(theta, phi)
            pos = jnp.asarray([x, y, z])

            source_pos, source_dir, source_time, source_nphotons = converter(
                pos, time, event_dir, 10 ** log10_energy, 11
            )

            return lh_func(
                event,
                det.module_coords,
                source_pos,
                source_dir,
                source_time,
                source_nphotons,
                c_medium,
            )

        return wrap_lh_call

    event_dir = sph_to_cart_jnp(event_data["theta"], event_data["phi"])

    matrices = []
    for _ in range(n_ev):
        key, subkey = random.split(key)
        event, record = generate_cascade(
            det,
            event_data["pos"],
            event_data["t0"],
            event_dir,
            energy=event_data["energy"],
            particle_id=event_data["pid"],
            pprop_func=ph_prop,
            seed=subkey,
            converter_func=converter,
            pprop_extras={"c_medium": c_medium},
        )

        padded = pad_event(event)

        wrap_lh_call = make_wrap_lh_call(padded)

        jac = jax.jit(
            jax.jacobian(wrap_lh_call, argnums=list(range(7)))(
                event_data["pos"][0],
                event_data["pos"][1],
                event_data["pos"][2],
                event_data["theta"],
                event_data["phi"],
                event_data["t0"],
                np.log10(event_data["energy"]),
            )
        )

        jac = np.stack(jac)[:, np.newaxis]
        matrices.append(jac * jac.T)
    matrix = np.average(np.stack(matrices), axis=0)
    return matrix


def rotate_to_new_direc(old_dir, new_dir, operand):

    axis = jnp.cross(old_dir, new_dir)
    axis /= jnp.linalg.norm(axis)

    theta = jnp.arccos(jnp.dot(old_dir, new_dir, precision=Precision.HIGHEST))

    # Rodrigues' rotation formula

    v_rot = (
        operand * jnp.cos(theta)
        + jnp.cross(axis, operand) * jnp.sin(theta)
        + axis
        * jnp.dot(axis, operand, precision=Precision.HIGHEST)
        * (1 - jnp.cos(theta))
    )
    return v_rot


rotate_to_new_direc_v = jax.jit(jax.vmap(rotate_to_new_direc, in_axes=[None, None, 0]))


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
