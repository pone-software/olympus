from abc import ABC, abstractmethod
from typing import Tuple, Any, List
import logging

import numpy as np
import numpy.typing as npt
import pandas as pd
from jax import random

from ananke.models.event import EventRecords, Sources
from ananke.models.geometry import Vectors3D
from ananke.models.detector import Detector
from ananke.schemas.event import SourceType
from .event_generation import generate_muon_energy_losses
from .lightyield import (
    make_realistic_cascade_source,
)
from .utils import proposal_setup
from ..configuration.generators import EventPropagatorConfiguration


# TODO: Configuration only needed for One Propagator
class AbstractPropagator(ABC):
    def __init__(
            self,
            detector: Detector,
            configuration: EventPropagatorConfiguration,
            seed: int
    ) -> None:
        super().__init__()
        self.configuration = configuration
        self.detector = detector
        self.seed = seed

    @abstractmethod
    def _records_to_sources(self, records: EventRecords, k1: Any) -> Sources:
        pass

    def _convert_to_sources(
            self,
            record_id: int,
            sources_information: Tuple[
                npt.ArrayLike,
                npt.ArrayLike,
                npt.ArrayLike,
                npt.ArrayLike,
            ]
    ) -> Sources:
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
        source_locations = Vectors3D.from_numpy(source_locations[mask])
        source_orientations = Vectors3D.from_numpy(source_orientations[mask])
        source_times = source_times[mask]
        source_number_of_photons = source_number_of_photons[mask]

        source_record_df = pd.concat(
            [
                source_locations.get_df_with_prefix('location_'),
                source_orientations.get_df_with_prefix('orientation_'),
            ], axis=1
        )

        source_record_df['record_id'] = record_id
        source_record_df['time'] = source_times
        source_record_df['number_of_photons'] = source_number_of_photons
        source_record_df['type'] = SourceType.STANDARD_CHERENKOV.value

        return Sources(df=source_record_df)

    def propagate(self, records: EventRecords) -> Sources:
        key, k1, k2 = random.split(random.PRNGKey(self.seed), 3)

        logging.info('Starting to propagate {} records'.format(len(records)))
        sources = self._records_to_sources(records=records, k1=k1)
        logging.info('Finished propagating {} records'.format(len(records)))

        return sources


class TrackPropagator(AbstractPropagator):
    def __init__(
            self,
            detector: Detector,
            configuration: EventPropagatorConfiguration,
            seed: int
    ) -> None:
        super().__init__(detector=detector, configuration=configuration, seed=seed)
        self.proposal_propagator = proposal_setup()

    def _records_to_sources(self, records: EventRecords, k1: str) -> Sources:
        source_records: List[Sources] = []
        number_of_records = len(records)
        for index, event_record in enumerate(records.df.itertuples()):
            sources = generate_muon_energy_losses(
                self.proposal_propagator,
                getattr(event_record, 'energy'),
                getattr(event_record, 'length'),
                np.array(getattr(event_record, 'location')),
                np.array(getattr(event_record, 'orientation')),
                getattr(event_record, 'time'),
                k1
            )

            event_source_records = self._convert_to_sources(
                getattr(event_record, 'record_id'),
                sources[:-1]
            )
            records.df.loc[index, 'length'] = sources[-1]
            source_records.append(event_source_records)
            logging.info('Propagated record {} of {}.'.format(index, number_of_records))

        return Sources.concat(source_records)


class CascadePropagator(AbstractPropagator):
    def __init__(
            self,
            detector: Detector,
            configuration: EventPropagatorConfiguration,
            seed: int
    ) -> None:
        super().__init__(detector=detector, configuration=configuration, seed=seed)
        self.resolution = configuration.resolution

    def _records_to_sources(self, records: EventRecords, k1: str) -> Sources:
        source_records: List[Sources] = []
        for event_record in records.df.itertuples():
            location = np.array(
                [
                    getattr(event_record, 'location_x'),
                    getattr(event_record, 'location_y'),
                    getattr(event_record, 'location_z'),
                ]
            )
            orientation = np.array(
                [
                    getattr(event_record, 'orientation_x'),
                    getattr(event_record, 'orientation_y'),
                    getattr(event_record, 'orientation_z'),
                ]
            )
            sources = make_realistic_cascade_source(
                location,
                getattr(event_record, 'time'),
                orientation,
                getattr(event_record, 'energy'),
                getattr(event_record, 'particle_id'),
                key=k1,
                moliere_rand=True,
                resolution=self.resolution
            )
            event_source_records = self._convert_to_sources(
                getattr(event_record, 'record_id'),
                sources
            )
            source_records.append(event_source_records)

        return Sources.concat(source_records)


class StartingTrackPropagator(TrackPropagator, CascadePropagator):
    def __init__(
            self,
            detector: Detector,
            configuration: EventPropagatorConfiguration,
            seed: int
    ) -> None:
        super().__init__(detector=detector, configuration=configuration, seed=seed)
        self.rng = np.random.default_rng(seed)

    def propagate(self, event_records: EventRecords) -> Sources:
        inelas = self.rng.uniform(1e-6, 1 - 1e-6)
        track_event_records = EventRecords(df=event_records.df.copy())
        cascade_event_records = EventRecords(df=event_records.df.copy())
        track_event_records.df.assign(energy=lambda x: inelas * x.energy)
        cascade_event_records.df.assign(energy=lambda x: (1 - inelas) * x.energy)

        track_sources = super(TrackPropagator, self).propagate(track_event_records)
        cascade_sources = super(CascadePropagator, self).propagate(
            cascade_event_records
        )

        return Sources.concat([track_sources, cascade_sources])
