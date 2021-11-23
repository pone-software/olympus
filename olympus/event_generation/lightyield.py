import numpy as np
from fennel import Fennel
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
    if np.abs(particle_id) == 11:
        counts_func, _, _ = fennel_instance.em_yields(
            energy=test_energy, particle=particle_id, mean=True, function=True
        )
    wavelengths = np.linspace(350, 500, 50)
    light_yield = np.trapz(
        counts_func(energy, wavelengths, particle=particle_id).ravel(), wavelengths
    )

    return light_yield


def fennel_differential_long_light_yield(energy, particle_id, resolution):
    if np.abs(particle_id) == 11:
        _, long_func, _ = fennel_instance.em_yields(
            energy=energy, particle=particle_id, mean=True, function=True
        )

    int_grid = np.arange(0, 30, resolution)
    integrand = lambda z: long_func(energy, z, particle=particle_id)

    norm = scipy.integrate.quad(integrand, 0, np.infty)[0]

    frac_yields = np.empty(len(int_grid) - 1)

    for i in range(len(int_grid) - 1):
        inte = scipy.integrate.quad(integrand, int_grid[i] * 100, int_grid[i + 1] * 100)
        frac_yields[i] = inte[0] / norm

    return frac_yields, int_grid


def make_pointlike_cascade_source(pos, t0, dir, energy, particle_id):
    n_photons = fennel_total_light_yield(energy, particle_id)
    source = PhotonSource(pos, n_photons, t0, dir)
    return [source]


def make_realistic_cascade_source(pos, t0, dir, energy, particle_id, resolution=0.5):

    n_photons_total = fennel_total_light_yield(energy, particle_id)
    yields, grid = fennel_differential_long_light_yield(energy, particle_id, resolution)

    for i, ly in enumerate(yields):

        inte = scipy.integrate.quad(integrand, int_grid[i] * 100, int_grid[i + 1] * 100)

        dist_along = 0.5 * (grid[i] + grid[i + 1])
        src_pos = dist_along * dir + pos
        ph_sources.append(
            PhotonSource(
                src_pos,
                inte[0] / norm * n_photons_total,
                t0 + dist_along / Constants.c_vac,
                dir,
            )
        )
    return ph_sources
