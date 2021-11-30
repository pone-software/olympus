"""Light yield calculation."""
import os

import numpy as np
import scipy.integrate
from fennel import Fennel

from .constants import Constants
from .photon_propagation import PhotonSource

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

    wavelengths = np.linspace(350, 500, 50)
    light_yield = np.trapz(counts_func(energy, wavelengths).ravel(), wavelengths)

    return light_yield


def fennel_frac_long_light_yield(energy, particle_id, resolution):
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
    int_grid = np.arange(0, 30, resolution)

    def integrand(z):
        return long_func(energy, z)

    norm = scipy.integrate.quad(integrand, 0, np.infty)[0]

    frac_yields = np.empty(len(int_grid) - 1)

    for i in range(len(int_grid) - 1):
        inte = scipy.integrate.quad(integrand, int_grid[i] * 100, int_grid[i + 1] * 100)
        frac_yields[i] = inte[0] / norm

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
    n_photons = fennel_total_light_yield(energy, particle_id)
    source = PhotonSource(pos, n_photons, t0, dir)
    return [source]


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
