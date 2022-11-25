from abc import ABC, abstractmethod
import dataclasses
from typing import Optional, Tuple, List

import pandas as pd
from jax import random
import awkward as ak
import copy
import uuid

import numpy as np
import numpy.typing as npt

from ananke.models.event import EventRecord, Event, SourceRecord, SourceRecordCollection
from ananke.models.geometry import Vector3D
from .constants import defaults
from .lightyield import make_realistic_cascade_source
from .photon_propagation.interface import AbstractPhotonPropagator
from .utils import proposal_setup

from .event_generation import generate_muon_energy_losses

from .detector import Detector


class AbstractPropagator(ABC):
    def __init__(
            self,
            detector: Detector,
            photon_propagator: AbstractPhotonPropagator,
            name: str,
            seed: int = defaults['seed'],
            **kwargs
    ) -> None:
        super().__init__()
        self.name = name
        self.detector = detector
        self.photon_propagator = photon_propagator
        self.seed = seed

    @abstractmethod
    def _convert(self, event_record: EventRecord, k1: str) -> Tuple[
        SourceRecordCollection, Optional[float]]:
        pass

    def _convert_source_array_to_sources(
            self,
            sources_information: Tuple[
                npt.NDArray[np.float64],
                npt.NDArray[np.float64],
                npt.NDArray[np.float64],
                npt.NDArray[np.float64]
            ]
    ) -> SourceRecordCollection:
        source_locations = sources_information[0]
        source_orientations = sources_information[1]
        source_times = sources_information[2]
        source_number_of_photons = sources_information[3]

        # early mask sources that are out of reach

        module_locations = np.array(self.detector.module_locations)

        dist_matrix = np.linalg.norm(
            source_locations[:, np.newaxis, ...]
            - module_locations[np.newaxis, ...],
            axis=-1,
        )

        # only consider photon sources within a certain distance to the module
        mask = np.any(dist_matrix < 300, axis=1)
        source_locations = source_locations[mask]
        source_orientations = source_orientations[mask]
        source_times = source_times[mask]
        source_number_of_photons = source_number_of_photons[mask]

        source_collection = SourceRecordCollection()

        for index, location in enumerate(source_locations):
            source_collection.append(
                SourceRecord(
                    location=Vector3D.from_numpy(location),
                    orientation=Vector3D.from_numpy(source_orientations[index]),
                    time=source_times[index],
                    number_of_photons=source_number_of_photons[index]
                )
            )

        return source_collection

    def propagate(self, event_record: EventRecord) -> Event:
        key, k1, k2 = random.split(random.PRNGKey(self.seed), 3)

        sources, length = self._convert(event_record=event_record, k1=k1)
        event_record.length = length
        event = Event(
            ID=uuid.uuid4().int,
            detector=self.detector,
            event_record=event_record,
            sources=sources
        )

        event.hits = self.photon_propagator.propagate(sources)

        return event


class TrackPropagator(AbstractPropagator):
    def __init__(
            self,
            detector: Detector,
            photon_propagator: AbstractPhotonPropagator,
            name: str = "track",
            seed: int = defaults['seed'],
            **kwargs
    ) -> None:
        super().__init__(detector, photon_propagator, name, seed, **kwargs)
        self.proposal_propagator = proposal_setup()

    def _convert(self, event_record: EventRecord, k1: str) -> Tuple[
        pd.DataFrame, Optional[float]]:
        sources = generate_muon_energy_losses(
            self.proposal_propagator,
            event_record.energy,
            event_record.length,
            np.array(event_record.location),
            np.array(event_record.orientation),
            event_record.time,
            k1,
        )
        return pd.DataFrame(
            data=sources[:4],
            columns=['location', 'orientation', 'time', 'number_of_photons']
        ), sources[4]


class CascadePropagator(AbstractPropagator):
    def __init__(
            self,
            detector: Detector,
            photon_propagator: AbstractPhotonPropagator,
            name: str = "cascade",
            seed: int = defaults['seed'],
            resolution: float = 0.2,
            **kwargs
    ) -> None:
        super().__init__(
            detector=detector,
            photon_propagator=photon_propagator,
            seed=seed,
            name=name,
            **kwargs
        )
        self.resolution = resolution

    def _convert(self, event_record: EventRecord, k1: str) -> Tuple[
        SourceRecordCollection, Optional[float]]:
        sources = make_realistic_cascade_source(
            np.array(event_record.location),
            event_record.time,
            np.array(event_record.orientation),
            event_record.energy,
            event_record.particle_id,
            key=k1,
            moliere_rand=True,
            resolution=self.resolution,
        )

        source_collection = self._convert_source_array_to_sources(sources)
        return source_collection, None


class StartingTrackPropagator(TrackPropagator, CascadePropagator):
    def __init__(
            self,
            detector: Detector,
            photon_propagator: AbstractPhotonPropagator,
            name: str = "track_starting",
            seed: int = defaults['seed'],
            resolution: float = 0.2,
            **kwargs
    ) -> None:
        super(StartingTrackPropagator, self).__init__(
            detector=detector,
            photon_propagator=photon_propagator,
            name=name,
            seed=seed,
            resolution=resolution,
            **kwargs
        )
        self.rng = np.random.default_rng(seed)

    def propagate(self, event_record: EventRecord):
        inelas = self.rng.uniform(1e-6, 1 - 1e-6)
        track_event_record = dataclasses.replace(
            event_record, energy=inelas * event_record.energy
        )
        cascade_event_record = dataclasses.replace(
            event_record, energy=(1 - inelas) * event_record.energy
        )

        track, track_record = super(TrackPropagator, self).propagate(track_event_record)
        cascade, cascade_record = super(CascadePropagator, self).propagate(
            cascade_event_record
        )

        raw_energy = event_record.energy
        track_event_record = copy.copy(event_record)

        track_event_record.energy = inelas * raw_energy

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
