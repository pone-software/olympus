from abc import ABC, abstractmethod
import dataclasses
from typing import Optional, Any
from jax import random
import awkward as ak
import copy

import numpy as np

from .lightyield import make_realistic_cascade_source
from .utils import proposal_setup

from .mc_record import MCRecord

from .event_generation import generate_muon_energy_losses

from .data import EventData
from .detector import Detector
from .constants import defaults
from .photon_propagation.utils import source_array_to_sources


class AbstractPropagator(ABC):
    def __init__(
            self,
            detector: Detector,
            photon_propagator: callable,
            name: str,
            rng: Optional[np.random.RandomState] = defaults['rng'],
            **kwargs
    ) -> None:
        super().__init__()
        self.name = name
        self.detector = detector
        self.photon_propagator = photon_propagator
        self.rng = rng

    @abstractmethod
    def _convert(self, event_data: EventData, k1: Any):
        pass

    def propagate(self, event_data: EventData):
        key, k1, k2 = random.split(event_data.key, 3)

        result = self._convert(event_data=event_data, k1=k1)

        source_pos = result[0]
        source_dir = result[1]
        source_time = result[2]
        source_photons = result[3]

        if len(result) > 4:
            event_data.length = result[4]

        if source_pos is None:
            return None, None

        # early mask sources that are out of reach

        dist_matrix = np.linalg.norm(
            source_pos[:, np.newaxis, ...]
            - self.detector.module_coords[np.newaxis, ...],
            axis=-1,
        )

        mask = np.any(dist_matrix < 300, axis=1)
        source_pos = source_pos[mask]
        source_dir = source_dir[mask]
        source_time = source_time[mask]

        record = MCRecord(
            self.name,
            source_array_to_sources(
                source_pos, source_dir, source_time, source_photons
            ),
            event_data,
        )

        propagation_result = self.photon_propagator(
            self.detector.module_coords,
            self.detector.module_efficiencies,
            source_pos,
            source_dir,
            source_time,
            source_photons,
            seed=k2,
        )

        return propagation_result, record


class TrackPropagator(AbstractPropagator):
    def __init__(
            self,
            detector: Detector,
            photon_propagator: callable,
            name: str = "track",
            rng: Optional[np.random.RandomState] = defaults['rng'],
            **kwargs
    ) -> None:
        super().__init__(detector, photon_propagator, name, rng, **kwargs)
        self.proposal_propagator = proposal_setup()

    def _convert(self, event_data: EventData, k1: str):
        return generate_muon_energy_losses(
            self.proposal_propagator,
            event_data.energy,
            event_data.length,
            event_data.start_position,
            event_data.direction,
            event_data.time,
            k1,
        )


class CascadePropagator(AbstractPropagator):
    def __init__(
            self,
            detector: Detector,
            photon_propagator: callable,
            name: str = "cascade",
            rng: Optional[np.random.RandomState] = defaults['rng'],
            resolution: float = 0.2,
            **kwargs
    ) -> None:
        super().__init__(
            detector=detector, photon_propagator=photon_propagator, rng=rng, name=name, **kwargs
        )
        self.resolution = resolution

    def _convert(self, event_data: EventData, k1: str):
        return make_realistic_cascade_source(
            event_data.start_position,
            event_data.time,
            event_data.direction,
            event_data.energy,
            event_data.particle_id,
            key=k1,
            moliere_rand=True,
            resolution=self.resolution,
        )


class StartingTrackPropagator(TrackPropagator, CascadePropagator):
    def __init__(
            self,
            detector: Detector,
            photon_propagator: callable,
            name: str = "track_starting",
            rng: Optional[np.random.RandomState] = defaults['rng'],
            resolution: float = 0.2,
            **kwargs
    ) -> None:
        super(StartingTrackPropagator, self).__init__(
            detector=detector,
            photon_propagator=photon_propagator,
            name=name,
            rng=rng,
            resolution=resolution,
            **kwargs)

    def propagate(self, event_data: EventData):
        inelas = self.rng.uniform(1e-6, 1 - 1e-6)
        track_event_data = dataclasses.replace(
            event_data, energy=inelas * event_data.energy
        )
        cascade_event_data = dataclasses.replace(
            event_data, energy=(1 - inelas) * event_data.energy
        )

        track, track_record = super(TrackPropagator, self).propagate(track_event_data)
        cascade, cascade_record = super(CascadePropagator, self).propagate(
            cascade_event_data
        )

        raw_energy = event_data.energy
        track_event_data = copy.copy(event_data)

        track_event_data.energy = inelas * raw_energy

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
