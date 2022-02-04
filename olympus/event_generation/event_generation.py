"""Event Generators."""
import logging

import awkward as ak
import jax.numpy as jnp
import numpy as np
from jax import random
from tqdm.auto import trange

from .constants import Constants
from .detector import (
    generate_noise,
    sample_cylinder_surface,
    sample_cylinder_volume,
    sample_direction,
)
from .lightyield import make_pointlike_cascade_source, make_realistic_cascade_source
from .mc_record import MCRecord
from .photon_source import PhotonSource
from .photon_propagation.utils import source_array_to_sources

logger = logging.getLogger(__name__)


def simulate_noise(det, event):

    if ak.count(event) == 0:
        time_range = [-1000, 5000]
        noise = generate_noise(det, time_range)
        event = ak.sort(noise, axis=1)

    else:
        time_range = [
            ak.min(ak.flatten(event)) - 1000,
            ak.max(ak.flatten(event)) + 5000,
        ]
        noise = generate_noise(det, time_range)
        event = ak.sort(ak.concatenate([event, noise], axis=1))

    return event, noise


def generate_cascade(
    det,
    event_data,
    seed,
    pprop_func,
    converter_func,
):
    """
    Generate a single cascade with given amplitude and position and return time of detected photons.

    Parameters:
        det: Detector
            Instance of Detector class
        event_data: dict
            Container of the event data
        seed: int
        pprop_func: function
            Function to calculate the photon signal
        converter_func: function
            Function to calculate number of photons as function of energy

    """

    k1, k2 = random.split(seed)

    source_pos, source_dir, source_time, source_nphotons = converter_func(
        event_data["pos"],
        event_data["time"],
        event_data["dir"],
        event_data["energy"],
        event_data["particle_id"],
        key=k1,
    )

    record = MCRecord(
        "cascade",
        source_array_to_sources(source_pos, source_dir, source_time, source_nphotons),
        event_data,
    )

    propagation_result = pprop_func(
        det.module_coords,
        det.module_efficiencies,
        source_pos,
        source_dir,
        source_time,
        source_nphotons,
        seed=k2,
    )

    return propagation_result, record


def generate_cascades(
    det,
    height,
    radius,
    nsamples,
    seed,
    log_emin,
    log_emax,
    pprop_func,
    pprop_extras=None,
    noise_function=simulate_noise,
):
    """Generate a sample of cascades, randomly sampling the positions in a cylinder of given radius and length."""
    rng = np.random.RandomState(seed)

    events = []
    records = []

    for i in trange(nsamples):
        pos = sample_cylinder_volume(height, radius, 1, rng).squeeze()
        energy = np.power(10, rng.uniform(log_emin, log_emax))
        dir = sample_direction(1, rng).squeeze()

        event, record = generate_cascade(
            det,
            pos,
            0,
            dir,
            energy=energy,
            seed=seed + i,
            pprop_func=pprop_func,
            pprop_extras=pprop_extras,
        )
        if noise_function is not None:
            event = noise_function(det, event)

        events.append(event)
        records.append(record)

    return events, records


def generate_muon_energy_losses(
    propagator, energy, track_len, position, direction, time
):
    try:
        import proposal as pp
    except ImportError as e:
        logger.critical("Could not import proposal!")
        raise e

    init_state = pp.particle.ParticleState()
    init_state.energy = energy * 1e3  # initial energy in MeV
    init_state.position = pp.Cartesian3D(
        position[0] * 100, position[1] * 100, position[2] * 100
    )
    init_state.direction = pp.Cartesian3D(direction[0], direction[1], direction[2])
    track = propagator.propagate(init_state, track_len * 100)  # cm

    aspos = []
    asdir = []
    astime = []
    asph = []

    # harvest losses
    for loss in track.stochastic_losses():
        # dist = loss.position.z / 100
        e_loss = loss.energy / 1e3

        """
        dir = np.asarray([loss.direction.x, loss.direction.y, loss.direction.z])
        
        p = position + dist * direction
        t = dist / Constants.c_vac + time
        """

        p = np.asarray([loss.position.x, loss.position.y, loss.position.z]) / 100
        dir = np.asarray([loss.direction.x, loss.direction.y, loss.direction.z])
        t = np.linalg.norm(p - position) / Constants.c_vac + time

        # TODO: Switch on loss type
        if e_loss < 1e3:
            spos, sdir, stime, sph = make_pointlike_cascade_source(
                p, t, dir, e_loss, 11
            )
        else:
            spos, sdir, stime, sph = make_realistic_cascade_source(
                p, t, dir, e_loss, 11
            )

        aspos.append(spos)
        asdir.append(sdir)
        astime.append(stime)
        asph.append(sph)

    return (
        jnp.concatenate(aspos),
        jnp.concatenate(asdir),
        jnp.concatenate(astime),
        jnp.concatenate(asph),
        track.track_propagated_distances()[-1] / 100,
    )


def generate_realistic_track(
    det,
    pos,
    direc,
    track_len,
    energy,
    t0,
    seed,
    pprop_func,
    pprop_extras=None,
    propagator=None,
):
    """
    Generate a realistic track using energy losses from PROPOSAL.

    Parameters:
      det: Detector
        Instance of Detector class
      pos: np.ndarray
        Position (x, y, z) of the track at t0
      direc: np.ndarray
        Direction (dx, dy, dz) of the track
      track_len: float
        Length of the track
      energy: float
        Initial energy of the track
      t0: float
        Time at position `pos`
      seed: int
      rng: RandomState
      propagator: Proposal propagator
      kwargs: dict
         kwargs passed to `generate_photons`
    """

    if propagator is None:
        raise RuntimeError()

    if pprop_extras is None:
        pprop_extras = {}

    (
        source_pos,
        source_dir,
        source_time,
        source_photons,
        prop_dist,
    ) = generate_muon_energy_losses(propagator, energy, track_len, pos, direc, t0)

    record = MCRecord(
        "realistic_track",
        None,
        {
            "position": pos,
            "energy": energy,
            "track_len": prop_dist,
            "direction": direc,
        },
    )

    hit_times = ak.sort(
        ak.Array(
            pprop_func(
                det.module_coords,
                det.module_efficiencies,
                source_pos,
                source_dir,
                source_time,
                source_photons,
                seed=seed,
                **pprop_extras
            )
        )
    )
    return hit_times, record


def generate_realistic_tracks(
    det,
    height,
    radius,
    nsamples,
    seed,
    log_emin,
    log_emax,
    pprop_func,
    propagator=None,
    pprop_extras=None,
):
    """Generate realistic muon tracks."""
    rng = np.random.RandomState(seed)
    # Safe length to that tracks will appear infinite
    # TODO: Calculate intersection with generation cylinder
    track_length = 3000

    events = []
    records = []
    noises = []

    for i in trange(nsamples):
        pos = sample_cylinder_surface(height, radius, 1, rng).squeeze()
        energy = np.power(10, rng.uniform(log_emin, log_emax, size=1))

        # determine the surface normal vectors given the samples position
        # surface normal always points out

        if pos[2] == height / 2:
            # upper cap
            area_norm = np.array([0, 0, 1])
        elif pos[2] == -height / 2:
            # lower cap
            area_norm = np.array([0, 0, -1])
        else:
            area_norm = np.array(pos, copy=True)
            area_norm[2] = 0
            area_norm /= np.linalg.norm(area_norm)

        orientation = 1
        # Rejection sampling to generate only inward facing tracks
        while orientation > 0:
            direc = sample_direction(1, rng).squeeze()
            orientation = np.dot(area_norm, direc)

        # shift pos back by half the length:
        # pos = pos - track_length / 2 * direc

        result = generate_realistic_track(
            det,
            pos,
            direc,
            track_length,
            energy=energy,
            t0=0,
            seed=seed + i,
            rng=rng,
            propagator=propagator,
            pprop_func=generate_photons,
            pprop_extras=pprop_extras,
        )

        event, record = result
        event, noise = simulate_noise(det, event)

        noises.append(noise)
        events.append(event)
        records.append(record)

    return events, records, noises


def generate_realistic_starting_tracks(
    det,
    height,
    radius,
    nsamples,
    seed,
    log_emin,
    log_emax,
    propagator=None,
    pprop_func=None,
    pprop_extras=None,
):
    """Generate realistic starting tracks (cascade + track)."""
    rng = np.random.RandomState(seed)
    # Safe length to that tracks will appear infinite
    # TODO: Calculate intersection with generation cylinder
    track_length = 3000

    events = []
    records = []

    for i in trange(nsamples):
        pos = sample_cylinder_volume(height, radius, 1, rng).squeeze()
        energy = np.power(10, rng.uniform(log_emin, log_emax))
        direc = sample_direction(1, rng).squeeze()
        inelas = rng.uniform(1e-6, 1 - 1e-6)

        track, track_record = generate_realistic_track(
            det,
            pos,
            direc,
            track_length,
            energy=energy * inelas,
            t0=0,
            seed=seed + i,
            rng=rng,
            propagator=propagator,
            pprop_func=generate_photons,
            pprop_extras=pprop_extras,
        )
        cascade, cascade_record = generate_cascade(
            det,
            pos,
            t0=0,
            seed=seed + 1,
            energy=energy * (1 - inelas),
            pprop_func=generate_photons,
            pprop_extras=pprop_extras,
        )

        event = ak.sort(ak.concatenate([track, cascade], axis=1))
        record = track_record + cascade_record

        event = simulate_noise(det, event)
        events.append(event)
        records.append(record)

    return events, records


def generate_uniform_track(
    det, pos, direc, track_len, energy, t0=0, res=10, seed=31337
):
    """
    Generate a track approximated by cascades at fixed intervals.

    Parameters:
      det: Detector
        Instance of Detector class
      pos: np.ndarray
        Position (x, y, z) of the track at t0
      direc: np.ndarray
        Direction (dx, dy, dz) of the track
      track_len: float
        Length of the track
      energy: float
        Energy of each individual cascade
      t0: float
        Time at position `pos`
      res: float
        Distance of cascades along the track [m]
      seed: float

    """
    sources = []

    for i in np.arange(0, track_len, res):
        p = pos + i * direc

        # Check if this position is way outside of the detector.
        # In that case: ignore

        if np.linalg.norm(p) > det.outer_radius + 3 * Constants.lambda_abs:
            continue

        t = i / Constants.c_vac + t0
        sources.append(PhotonSource(p, energy * Constants.photons_per_GeV, t))

    record = MCRecord(
        "uniform_track",
        dejit_sources(sources),
        {"position": pos, "energy": energy, "track_len": track_len, "direction": direc},
    )
    hit_times = ak.sort(
        ak.Array(
            generate_photons(
                det.module_coords, det.module_efficiencies, List(sources), seed=seed
            )
        )
    )
    return hit_times, record


def generate_uniform_tracks(
    det,
    height,
    radius,
    nsamples,
    seed=31337,
):
    rng = np.random.RandomState(seed)
    # Safe length to that tracks will appear infinite
    # TODO: Calculate intersection with generation cylinder
    track_length = 3000

    positions = sample_cylinder_surface(height, radius, nsamples, rng)
    directions = sample_direction(nsamples, rng)

    # Sample amplitude uniform in log
    amplitudes = np.power(10, rng.uniform(0, 4, size=nsamples))

    events = []
    records = []

    for i, (pos, amp, direc) in enumerate(zip(positions, amplitudes, directions)):

        # shift pos back by half the length:
        pos = pos - track_length / 2 * direc

        event, record = generate_uniform_track(
            det, pos, direc, track_length, amp, 0, 10, seed + i
        )
        if ak.count(event) == 0:
            continue
        time_range = [
            ak.min(ak.flatten(event)) - 1000,
            ak.max(ak.flatten(event)) + 5000,
        ]
        noise = generate_noise(det, time_range)
        event = ak.sort(ak.concatenate([event, noise], axis=1))
        events.append(event)
        records.append(record)
    return events, records
