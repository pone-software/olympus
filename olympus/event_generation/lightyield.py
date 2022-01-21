"""Light yield calculation."""
import os

import numpy as np
import scipy.integrate
from fennel import Fennel, config

from .constants import Constants
from .photon_source import PhotonSource

import jax.numpy as jnp
import jax

config["general"]["jax"] = True
fennel_instance = Fennel()


def simple_cascade_light_yield(energy, *args):
    """
    Approximation for cascade light yield.

    Parameters:
        energy: float
            Particle energy in GeV
    """
    photons_per_GeV = 5.3 * 250 * 1e2

    return energy * photons_per_GeV


def fennel_total_light_yield(energy, particle_id):
    """
    Calculate total light yield using fennel.

    Parameters:
        energy: float
            Particle energy in GeV
        particle_id: int
    """

    funcs = fennel_instance.auto_yields(energy, particle_id, function=True)
    counts_func = funcs[0]

    wavelengths = jnp.linspace(350, 500, 50)
    light_yield = jnp.trapz(counts_func(energy, wavelengths).ravel(), wavelengths)

    return light_yield


def fennel_frac_long_light_yield(energy, particle_id, resolution=0.2):
    """
    Calculate the longitudinal light yield contribution.

    Integrate the longitudinal distribution in steps of `resolution` and
    return the relative contributions.

    Parameters:
        energy: float
            Particle energy in GeV
        particle_id: int
        resolution: float
            Step length in m for evaluating the longitudinal distribution
    """
    funcs = fennel_instance.auto_yields(energy, particle_id, function=True)
    long_func = funcs[4]
    int_grid = jnp.arange(1e-3, 30, resolution)

    def integrate(low, high, resolution=1000):
        trapz_x_eval = jnp.linspace(low, high, resolution) * 100  # to cm
        trapz_y_eval = long_func(energy, trapz_x_eval)
        return jnp.trapz(trapz_y_eval, trapz_x_eval)

    integrate_v = jax.vmap(integrate, in_axes=[0, 0])

    norm = integrate(1e-3, 100)
    frac_yields = integrate_v(int_grid[:-1], int_grid[1:]) / norm

    return frac_yields, int_grid


def make_pointlike_cascade_source(pos, t0, dir, energy, particle_id):
    """
    Create a pointlike lightsource.

    Parameters:
        pos: float[3]
            Cascade position
        t0: float
            Cascade time
        dir: float[3]
            Cascade direction
        energy: float
            Cascade energy
        particle_id: int
            Particle type (PDG ID)

    Returns:
        List[PhotonSource]

    """
    source_nphotons = jnp.asarray([fennel_total_light_yield(energy, particle_id)])[
        np.newaxis, :
    ]

    source_pos = pos[np.newaxis, :]
    source_dir = dir[np.newaxis, :]
    source_time = jnp.asarray([t0])[np.newaxis, :]
    # source = PhotonSource(pos, n_photons, t0, dir)
    return source_pos, source_dir, source_time, source_nphotons


def make_realistic_cascade_source(pos, t0, dir, energy, particle_id, resolution=0.2):
    """
    Create a realistic (elongated) particle cascade.

    The longitudinal profile is approximated by placing point-like light sources
    every `resolution` steps.

    Parameters:
        pos: float[3]
            Cascade position
        t0: float
            Cascade time
        dir: float[3]
            Cascade direction
        energy: float
            Cascade energy
        particle_id: int
            Particle type (PDG ID)
        resolution: float
            Step size for point-like light sources
    """
    n_photons_total = fennel_total_light_yield(energy, particle_id)
    frac_yields, grid = fennel_frac_long_light_yield(energy, particle_id, resolution)

    dist_along = 0.5 * (grid[:-1] + grid[1:])
    source_pos = dist_along[:, np.newaxis] * dir[np.newaxis, :] + pos[np.newaxis, :]
    source_dir = jnp.tile(dir, (dist_along.shape[0], 1))
    source_nphotons = frac_yields * n_photons_total
    source_time = t0 + dist_along / (Constants.c_vac)

    return (
        source_pos,
        source_dir,
        source_time[:, np.newaxis],
        source_nphotons[:, np.newaxis],
    )

    """
    sources = []
    for i, frac_yield in enumerate(frac_yields):

        dist_along = 0.5 * (grid[i] + grid[i + 1])
        src_pos = dist_along * dir + pos
        sources.append(
            PhotonSource(
                src_pos,
                frac_yield * n_photons_total,
                t0 + dist_along / (Constants.c_vac),
                dir,
            )
        )
    return sources
    """
