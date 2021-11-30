"""Event Generators."""
import logging

import awkward as ak
import numpy as np
from numba.typed import List
from tqdm.auto import trange

from .constants import Constants
from .detector import (
    generate_noise,
    sample_cylinder_surface,
    sample_cylinder_volume,
    sample_direction,
)
from .lightyield import make_realistic_cascade_source
from .mc_record import MCRecord
from .photon_propagation import PhotonSource, dejit_sources, generate_photons

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
    pos,
    t0,
    dir,
    energy,
    particle_id,
    seed=31337,
    pprop_func=generate_photons,
    pprop_extras=None,
    converter_func=make_realistic_cascade_source,
    converter_extras=None,
):
    """
    Generate a single cascade with given amplitude and position and return time of detected photons.

    Parameters:
        det: Detector
            Instance of Detector class
        pos: np.ndarray
            Position (x, y, z) of the cascade
        t0: float
            Time of the cascade
        dir: float
            Direction of the cascade
        energy: float
            Energy of the cascade
        particle_id: int
        seed: int
        pprop_func: function
            Function to calculate the photon signal
        pprop_extras: dict
        converter_func: function
            Function to calculate number of photons as function of energy
        converter_extras: dict
    """
    if pprop_extras is None:
        pprop_extras = {}

    if converter_extras is None:
        converter_extras = {}

    source_list = converter_func(pos, t0, dir, energy, particle_id, **converter_extras)
    record = MCRecord(
        "cascade",
        dejit_sources(source_list),
        {"energy": energy, "position": pos, "direction": dir, "time": t0},
    )

    propagation_result = pprop_func(
        det.module_coords,
        det.module_efficiencies,
        List(source_list),
        seed=seed,
        **pprop_extras
    )

    return propagation_result, record


def generate_cascades(
    det,
    height,
    radius,
    nsamples,
    seed=31337,
    log_emin=2,
    log_emax=6,
    pprop_func=generate_photons,
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


def generate_realistic_track(
    det,
    pos,
    direc,
    track_len,
    energy,
    t0=0,
    res=10,
    seed=31337,
    rng=np.random.RandomState(31337),
    propagator=None,
    pprop_func=generate_photons,
    pprop_extras=None,
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
    try:
        import proposal as pp
    except ImportError as e:
        logger.critical("Could not import proposal!")
        raise e

    sources = []

    if propagator is None:
        raise RuntimeError()

    if pprop_extras is None:
        pprop_extras = {}

    init_state = pp.particle.ParticleState()
    init_state.energy = energy * 1e3  # initial energy in MeV
    init_state.position = pp.Cartesian3D(0, 0, 0)
    init_state.direction = pp.Cartesian3D(0, 0, 1)
    track = propagator.propagate(init_state, track_len * 100)  # cm

    # harvest losses
    for loss in track.stochastic_losses():
        dist = loss.position.z / 100
        e_loss = loss.energy / 1e3
        dir = np.asarray([loss.direction.x, loss.direction.y, loss.direction.z])
        p = pos + dist * direc
        t = dist / Constants.c_vac + t0

        if np.linalg.norm(p) > det.outer_radius + 3 * Constants.lambda_abs:
            continue
        sources.append(PhotonSource(p, e_loss * Constants.photons_per_GeV, t, dir))

    record = MCRecord(
        "realistic_track",
        dejit_sources(sources),
        {
            "position": pos,
            "energy": energy,
            "track_len": track.track_propagated_distances()[-1] / 100,
            "direction": direc,
        },
    )

    if not sources:
        return ak.Array([[]]), record

    hit_times = ak.sort(
        ak.Array(
            pprop_func(
                det.module_coords,
                det.module_efficiencies,
                List(sources),
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
    seed=31337,
    propagator=None,
    log_emin=2,
    log_emax=6,
    pprop_func=generate_photons,
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
    seed=31337,
    propagator=None,
    log_emin=2,
    log_emax=6,
    pprop_func=generate_photons,
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
