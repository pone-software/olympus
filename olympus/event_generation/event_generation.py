"""Event Generators."""
from dataclasses import dataclass
import functools
import logging
import string
from time import time
from turtle import position
from typing import Any, Callable, Optional, List
from abc import ABC, abstractmethod


import awkward as ak
import jax.numpy as jnp
import numpy as np
from jax import random
from tqdm.auto import trange
from .constants import Constants, Defaults
from .detector import (
    Detector,
    generate_noise,
    sample_cylinder_surface,
    sample_cylinder_volume,
    sample_direction,
)
from .lightyield import make_pointlike_cascade_source, make_realistic_cascade_source
from .mc_record import MCRecord
from .photon_propagation.utils import source_array_to_sources
from .photon_source import PhotonSource
from .utils import get_event_times_by_rate, track_isects_cyl

logger = logging.getLogger(__name__)

_default_rng = Defaults.rng


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
    rng=_default_rng
):
return EventFactory.get_generator(asdasdasd)
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
    cylinder_height,
    cylinder_radius,
    nsamples,
    seed,
    log_emin,
    log_emax,
    particle_id,
    pprop_func,
    converter_func,
    noise_function=simulate_noise,
):
    """Generate a sample of cascades, randomly sampling the positions in a cylinder of given radius and length."""
    rng = np.random.RandomState(seed)
    key = random.PRNGKey(seed)

    events = []
    records = []

    for i in trange(nsamples):
        pos = sample_cylinder_volume(cylinder_height, cylinder_radius, 1, rng).squeeze()
        energy = _get_event_energy(log_emin=log_emin, log_emax=log_emax, rng=rng)
        dir = _get_event_direction(rng=rng)

        event_data = {
            "pos": pos,
            "dir": dir,
            "energy": energy,
            "time": 0,
            "particle_id": particle_id,
        }

        key, subkey = random.split(key)
        event, record = generate_cascade(
            det,
            event_data,
            subkey,
            pprop_func,
            converter_func,
        )
        if noise_function is not None:
            event, _ = noise_function(det, event)

        events.append(event)
        records.append(record)

    return events, records


def generate_muon_energy_losses(
    propagator,
    energy,
    track_len,
    position,
    direction,
    time,
    key,
    loss_resolution=0.2,
    cont_resolution=1,
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

    loss_map = {
        "brems": 11,
        "epair": 11,
        "hadrons": 211,
        "ioniz": 11,
        "photonuclear": 211,
    }

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

        loss_type_name = pp.particle.Interaction_Type(loss.type).name
        ptype = loss_map[loss_type_name]

        if e_loss < 1e3:
            spos, sdir, stime, sph = make_pointlike_cascade_source(
                p, t, dir, e_loss, ptype
            )
        else:
            key, subkey = random.split(key)
            spos, sdir, stime, sph = make_realistic_cascade_source(
                p,
                t,
                dir,
                e_loss,
                ptype,
                subkey,
                resolution=loss_resolution,
                moliere_rand=True,
            )

        aspos.append(spos)
        asdir.append(sdir)
        astime.append(stime)
        asph.append(sph)

    # distribute continuous losses uniformly along track
    # TODO: check if thats a good approximation
    # TODO: track segments

    cont_loss_sum = sum([loss.energy for loss in track.continuous_losses()]) / 1e3
    total_dist = track.track_propagated_distances()[-1] / 100
    loss_dists = np.arange(0, total_dist, cont_resolution)
    e_loss = cont_loss_sum / len(loss_dists)

    for ld in loss_dists:
        p = ld * direction + position
        t = np.linalg.norm(p - position) / Constants.c_vac + time

        spos, sdir, stime, sph = make_pointlike_cascade_source(
            p, t, direction, e_loss, 11
        )

        aspos.append(spos)
        asdir.append(sdir)
        astime.append(stime)
        asph.append(sph)

    if not aspos:
        return None, None, None, None, total_dist

    return (
        np.concatenate(aspos),
        np.concatenate(asdir),
        np.concatenate(astime),
        np.concatenate(asph),
        total_dist,
    )


def generate_realistic_track(
    det,
    event_data,
    key,
    pprop_func,
    proposal_prop,
    rng=_default_rng
):
    """
    Generate a realistic track using energy losses from PROPOSAL.

    Parameters:
        det: Detector
            Instance of Detector class
        event_data: dict
            Container of the event data
        seed: PRNGKey
        pprop_func: function
            Function to calculate the photon signal
        proposal_prop: function
            Propoposal propagator
    """

    if proposal_prop is None:
        raise RuntimeError()

    key, k1, k2 = random.split(key, 3)
    (
        source_pos,
        source_dir,
        source_time,
        source_photons,
        prop_dist,
    ) = generate_muon_energy_losses(
        proposal_prop,
        event_data["energy"],
        event_data["length"],
        event_data["pos"],
        event_data["dir"],
        event_data["time"],
        k1,
    )
    event_data["length"] = prop_dist

    if source_pos is None:
        return None, None

    # early mask sources that are out of reach

    dist_matrix = np.linalg.norm(
        source_pos[:, np.newaxis, ...] - det.module_coords[np.newaxis, ...], axis=-1
    )

    mask = np.any(dist_matrix < 300, axis=1)
    source_pos = source_pos[mask]
    source_dir = source_dir[mask]
    source_time = source_time[mask]
    source_photons = source_photons[mask]

    record = MCRecord(
        "realistic_track",
        source_array_to_sources(source_pos, source_dir, source_time, source_photons),
        event_data,
    )

    propagation_result = pprop_func(
        det.module_coords,
        det.module_efficiencies,
        source_pos,
        source_dir,
        source_time,
        source_photons,
        seed=k2,
    )
    return propagation_result, record

def generate_realistic_starting_track(
    det,
    event_data,
    key,
    pprop_func,
    proposal_prop,
    rng=_default_rng
): 
    
    inelas = rng.uniform(1e-6, 1 - 1e-6)

    raw_energy = event_data["energy"]
    track_event_data = event_data.copy()

    track_event_data['energy'] = inelas * raw_energy

    track, track_record = generate_realistic_track(
        det,
        track_event_data,
        key=key,
        proposal_prop=proposal_prop,
        pprop_func=pprop_func,
    )

    cascade_event_data = event_data.copy()

    cascade_event_data['energy'] = (1 - inelas) * raw_energy

    cascade, cascade_record = generate_cascade(
        det,
        event_data,
        key=key,
        pprop_func=pprop_func,
        converter_func=functools.partial(
            make_realistic_cascade_source, moliere_rand=True, resolution=0.2
        ),
    )

    if (ak.count(track) == 0) & (ak.count(cascade) == 0):
        event = ak.Array([])

    elif ak.count(track) == 0:
        event = cascade
    elif (ak.count(cascade)) == 0:
        event = track
    else:
        event = ak.sort(ak.concatenate([track, cascade], axis=1))
    record = track_record + cascade_record

    return event, record

def _generate_times_from_rate(
    rate: float,
    start_time: int,
    end_time: int,
    rng = _default_rng
) -> List[int]:
    times_det = get_event_times_by_rate(rate=rate, start_time=start_time, end_time=end_time, rng=rng)

    return ak.sort(ak.Array(times_det))

def _get_event_energy(log_emin: float, log_emax: float, rng=_default_rng) -> np.ndarray:
    return np.power(10, rng.uniform(log_emin, log_emax, size=1))

def _get_event_direction(rng=_default_rng) -> np.ndarray:
    return sample_direction(1, rng).squeeze()

def _generate_events(
    det: Detector,
    log_emin: float,
    log_emax: float,
    gen_func: Callable,
    pos_func: Callable,
    pprop_func: Callable,
    nsamples: Optional[int] = None,
    rate: Optional[float] = None,
    start_time: Optional[int] = 0,
    end_time: Optional[int] = None,
    seed: int = 0,
    proposal_prop=None,
):
    """Generate realistic muon tracks."""
    rng = np.random.RandomState(seed)
    key, subkey = random.split(random.PRNGKey(seed))

    events = []
    records = []

    if nsamples is None and (rate is None or end_time is None):
        raise ValueError('Either number of samples or time parameters must be set')

    time_based = nsamples is None

    if time_based:
        iterator_range = _generate_times_from_rate(rate=rate, start_time=start_time, end_time=end_time, seed=seed)
    else:
        iterator_range = trange(nsamples)
    
    cylinder_height= det.outer_cylinder[1] + 100
    cylinder_radius= det.outer_cylinder[0] + 50

    # TODO: Calculate intersection with generation cylinder
    track_length = 3000

    events = []
    records = []

    for i in iterator_range:
        pos = sample_cylinder_volume(cylinder_height=cylinder_height, cylinder_radius=cylinder_radius, n=1, rng=rng)
        energy = _get_event_energy(log_emin=log_emin, log_emax=log_emax, rng=rng)
        direc = _get_event_direction(rng=rng)

        event_data = {
            "pos": pos,
            "dir": direc,
            "energy": energy,
            "length": track_length,
        }

        if time_based:
            event_data["time"] = i
        else:
            event_data["time"] = 0

        event, record = gen_func(
            det,
            event_data,
            key=subkey,
            proposal_prop=proposal_prop,
            pprop_func=pprop_func,
        )

        events.append(event)
        records.append(record)

    return events, records




def generate_realistic_tracks(
    det,
    cylinder_height,
    cylinder_radius,
    nsamples,
    seed,
    log_emin,
    log_emax,
    pprop_func,
    proposal_prop=None,
    times=None
):
    """Generate realistic muon tracks."""
    rng = np.random.RandomState(seed)
    key = random.PRNGKey(seed)

    events = []
    records = []

    time_based = times is None

    if time_based:
        iterator_range = times
    else:
        pass

    for i in trange(nsamples):
        pos = sample_cylinder_surface(
            cylinder_height, cylinder_radius, 1, rng
        ).squeeze()
        energy = _get_event_energy(log_emin=log_emin, log_emax=log_emax, rng=rng)

        # determine the surface normal vectors given the samples position
        # surface normal always points out

        if pos[2] == cylinder_height / 2:
            # upper cap
            area_norm = np.array([0, 0, 1])
        elif pos[2] == -cylinder_height / 2:
            # lower cap
            area_norm = np.array([0, 0, -1])
        else:
            area_norm = np.array(pos, copy=True)
            area_norm[2] = 0
            area_norm /= np.linalg.norm(area_norm)

        orientation = 1
        # Rejection sampling to generate only inward facing tracks
        while orientation > 0:
            direc = _get_event_direction(rng=rng)
            orientation = np.dot(area_norm, direc)

        # shift pos back by half the length:
        # pos = pos - track_length / 2 * direc

        isec = track_isects_cyl(
            det._outer_cylinder[0], det._outer_cylinder[1], pos, direc
        )
        track_length = 3000
        if (isec[0] != np.nan) and (isec[1] != np.nan):
            track_length = isec[1] - isec[0] + 300

        event_data = {
            "pos": pos,
            "dir": direc,
            "energy": energy,
            "time": 0,
            "length": track_length,
        }

        key, subkey = random.split(key)
        result = generate_realistic_track(
            det,
            event_data,
            key=subkey,
            proposal_prop=proposal_prop,
            pprop_func=pprop_func,
        )

        event, record = result
        event, _ = simulate_noise(det, event)

        events.append(event)
        records.append(record)

    return events, records


def generate_realistic_starting_tracks(
    det,
    cylinder_height,
    cylinder_radius,
    nsamples,
    seed,
    log_emin,
    log_emax,
    pprop_func,
    proposal_prop=None,
):
    """Generate realistic starting tracks (cascade + track)."""
    rng = np.random.RandomState(seed)
    key, subkey = random.split(random.PRNGKey(seed))
    # Safe length to that tracks will appear infinite
    # TODO: Calculate intersection with generation cylinder
    track_length = 3000

    events = []
    records = []

    for i in trange(nsamples):
        pos = sample_cylinder_volume(cylinder_height, cylinder_radius, 1, rng).squeeze()
        energy = _get_event_energy(log_emin=log_emin, log_emax=log_emax, rng=rng)
        direc = _get_event_direction(rng=rng)
        inelas = rng.uniform(1e-6, 1 - 1e-6)

        event_data = {
            "pos": pos,
            "dir": direc,
            "energy": inelas * energy,
            "time": 0,
            "length": track_length,
        }

        track, track_record = generate_realistic_track(
            det,
            event_data,
            key=subkey,
            proposal_prop=proposal_prop,
            pprop_func=pprop_func,
        )

        event_data = {
            "pos": pos,
            "dir": direc,
            "energy": (1 - inelas) * energy,
            "time": 0,
            "length": track_length,
            "particle_id": 211,
        }

        cascade, cascade_record = generate_cascade(
            det,
            event_data,
            subkey,
            pprop_func,
            functools.partial(
                make_realistic_cascade_source, moliere_rand=True, resolution=0.2
            ),
        )

        if (ak.count(track) == 0) & (ak.count(cascade) == 0):
            event = ak.Array([])

        elif ak.count(track) == 0:
            event = cascade
        elif (ak.count(cascade)) == 0:
            event = track
        else:
            event = ak.sort(ak.concatenate([track, cascade], axis=1))
        record = track_record + cascade_record

        event, _ = simulate_noise(det, event)
        events.append(event)
        records.append(record)

    return events, records

class EventFactory():
    def get_generator(type):
        if type == 'cascade':
            return EventGenerator(VolumeInjector, PowerLawSpectrum, Propagator)
        return EventGenerato

class SurfaceInjector

class VolumeInjector

class TrackGenerator(EventGenerator)
    def __init__():
        self.injector = VolumeInjector
        self.propagator = TrackPropagator

@dataclass
class EventData:
    direction = np.ndarray
    energy = np.ndarray
    time = int
    length = float
    start_position = np.ndarray
    type = string
    particle_id = Optional[Any]

class EventGenerator(ABC):
    def __init__(
        self, 
        det: Detector, 
        log_emin: float, 
        log_emax: float,
        seed: Optional[int] = 0,
        rate: Optional[float] = None,
        proposal_prop: Optional[callable] = None,
        pprop_func: Optional[callable] = None,
    ) -> None:
        self.det = det
        self.log_emin = log_emin
        self.log_emax = log_emax
        self.seed = seed
        self.rng = np.random.RandomState(seed=seed)
        self.rate = rate
        self.proposal_prop=proposal_prop,
        self.pprop_func=pprop_func,

    @abstractmethod
    def _generate_single(event_data: dict):
        pass

    def generate(
        self,
        nsamples: Optional[int] = None, 
        start_time: Optional[int] = 0, 
        end_time: Optional[int] = None, 
        rate: Optional[float] = None):
        
        """Generate realistic muon tracks."""
        key, subkey = random.split(random.PRNGKey(self.seed))

        events = []
        records = []

        if nsamples is None and (rate is None or end_time is None):
            raise ValueError('Either number of samples or time parameters must be set')

        time_based = nsamples is None

        if time_based:
            if rate is None:
                rate = self.rate
            iterator_range = _generate_times_from_rate(rate=rate, start_time=start_time, end_time=end_time, seed=self.seed)
        else:
            iterator_range = trange(nsamples)
        
        cylinder_height= self.det.outer_cylinder[1] + 100
        cylinder_radius= self.det.outer_cylinder[0] + 50

        track_length = 3000

        events = []
        records = []

        for i in iterator_range:
            pos = sample_cylinder_volume(cylinder_height=cylinder_height, cylinder_radius=cylinder_radius, n=1, rng=self.rng)
            energy = _get_event_energy(log_emin=self.log_emin, log_emax=self.log_emax, rng=rng)
            direc = _get_event_direction(rng=self.rng)

            event_data = {
                "pos": pos,
                "dir": direc,
                "energy": energy,
                "length": track_length,
            }

            if time_based:
                event_data["time"] = i
            else:
                event_data["time"] = start_time

            event, record = self._generate_single(
                event_data,
                key=subkey,
            )

            events.append(event)
            records.append(record)

        return events, records

    def generate_per_timeframe(self, start_time: int, end_time: int, 
        rate: Optional[float] = None):
        return self._generate(start_time=start_time, end_time=end_time, rate=rate)

    def generate_nsamples(self, nsamples: int, start_time: Optional[int] = 0):
        return self._generate(nsamples=nsamples, start_time=start_time)


class EventTimelineGenerator():
    def __init__(self, det: Detector) -> None:
        self.det = det
        self.generators = []

    def add_generator(self, generator: EventGenerator):
        self.generators.append(generator)

    def generate(self, start_time: Optional[int] = 0, end_time: Optional[int] = None, nsamples: Optional[int] = None):
        events = []
        records = []

        for generator in self.generators:
            gen_events, gen_records = generator.generate(start_time=start_time, end_time=end_time, nsamples=nsamples)
            events += gen_events
            records += gen_records
        return events, records


    def generate_per_timeframe(self, start_time: int, end_time: int):
        return self.generate(start_time=start_time, end_time=end_time)

    def generate_nsamples(self, nsamples: int):
        return self.generate(nsamples=None)