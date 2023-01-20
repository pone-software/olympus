import uuid
from abc import ABC
from typing import Optional, Type, List

import pandas as pd
import numpy as np

from ananke.models.detector import Detector
from ananke.models.event import (
    EventRecords,
    Sources,
    Collection,
    NoiseRecords,
    Hits,
)
from ananke.models.geometry import Vectors3D
from ananke.schemas.event import EventType, NoiseType
from .detector import sample_direction
from .injectors import (
    AbstractInjector,
    SurfaceInjector,
    VolumeInjector,
)
from .photon_propagation.interface import AbstractPhotonPropagator
from .propagators import (
    AbstractPropagator,
    CascadePropagator,
    StartingTrackPropagator,
    TrackPropagator,
)
from .spectra import AbstractSpectrum, UniformSpectrum
from ..generators import AbstractGenerator
from ..constants import defaults


class EventGenerator(AbstractGenerator):
    def __init__(
            self,
            event_type: EventType,
            particle_id: int,
            injector: AbstractInjector,
            propagator: AbstractPropagator,
            photon_propagator: AbstractPhotonPropagator = None,
            spectrum: Optional[AbstractSpectrum] = None,
            *args,
            **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.injector = injector
        if spectrum is None:
            spectrum = UniformSpectrum(
                log_maximal_energy=kwargs["log_maximal_energy"],
                log_minimal_energy=kwargs["log_minimal_energy"],
            )
        self.spectrum = spectrum
        self.propagator = propagator
        self.photon_propagator = photon_propagator
        self.event_type = event_type
        self.particle_id = particle_id

    def generate(
            self,
            number_of_samples: int
    ) -> Collection:
        """Generate realistic muon tracks."""
        records = self.generate_records(number_of_samples)
        sources = self.propagate(records)
        if self.photon_propagator is None:
            raise ValueError('Photon Propagator is not defined')
        hits = self.photon_propagator.propagate(records, sources)

        collection = Collection(
            records=records,
            detector=self.propagator.detector,
            sources=sources,
            hits=hits
        )

        return collection

    def generate_records(self, number_of_samples) -> EventRecords:

        track_length = 3000

        locations = self.injector.get_positions(n=number_of_samples)
        energies = self.spectrum.get_energy(n=number_of_samples)
        orientations = Vectors3D.from_numpy(
            sample_direction(n_samples=number_of_samples, rng=self.rng)
        )
        ids = self._get_record_ids(number_of_samples)
        event_records_df = pd.concat(
            [
                locations.get_df_with_prefix('location_'),
                orientations.get_df_with_prefix('orientation_'),
            ],
            axis=1
        )

        event_records_df['record_id'] = ids
        event_records_df['energy'] = energies
        event_records_df['length'] = track_length
        # TODO: Make flexible start time possible
        event_records_df['time'] = 0
        event_records_df['type'] = self.event_type.value
        event_records_df['particle_id'] = self.particle_id

        event_records = EventRecords(df=event_records_df)

        return event_records

    def propagate(self, records: EventRecords) -> Sources:
        return self.propagator.propagate(records=records)


class CascadeGenerator(EventGenerator):
    def __init__(
            self,
            detector: Detector,
            *args,
            **kwargs
    ):
        injector = VolumeInjector(detector=detector, **kwargs)
        propagator = CascadePropagator(
            detector=detector, **kwargs
        )
        super().__init__(
            event_type=EventType.CASCADE,
            injector=injector, propagator=propagator, *args, **kwargs
        )


class TrackEventGenerator(EventGenerator):
    def __init__(
            self, detector: Detector,
            *args,
            **kwargs
    ):
        injector = SurfaceInjector(detector=detector, **kwargs)
        spectrum = UniformSpectrum(
            log_maximal_energy=kwargs["log_maximal_energy"],
            log_minimal_energy=kwargs["log_minimal_energy"],
        )
        propagator = TrackPropagator(
            detector=detector, **kwargs
        )
        super().__init__(
            event_type=EventType.REALISTIC_TRACK,
            injector=injector, spectrum=spectrum, propagator=propagator, *args, **kwargs
        )


class StartingTrackEventGenerator(EventGenerator):
    def __init__(
            self, detector: Detector,
            *args,
            **kwargs
    ):
        injector = VolumeInjector(detector=detector, **kwargs)
        propagator = StartingTrackPropagator(
            detector=detector, **kwargs
        )
        super().__init__(
            event_type=EventType.STARTING_TRACK,
            injector=injector, propagator=propagator, *args, **kwargs
        )


class NoiseGenerator(AbstractGenerator, ABC):
    """Abstract parent for all Noise Generators"""

    def __init__(
            self,
            noise_type: NoiseType,
            detector: Detector,
            start_time: int,
            duration: int,
            *args,
            **kwargs
    ):
        """Constructor of the noise generator class.

        Args:
            noise_type: type of the noise to be generated
            detector: detector to build Noise on
            start_time: start time of the noise
            duration: duration of the noise
        """
        super().__init__(*args, **kwargs)
        self.noise_type = noise_type
        self.detector = detector
        self.start_time = start_time
        self.duration = duration

    def generate_records(self, number_of_samples: int) -> NoiseRecords:
        """Generic generator of noise records.

        Args:
            number_of_samples: Amount of noise records to be generated.

        Returns:
            Generated noise records.
        """
        ids = self._get_record_ids(number_of_samples)

        noise_records_df = pd.DataFrame(
            {
                'record_id': ids
            }
        )

        noise_records_df['time'] = self.start_time
        noise_records_df['duration'] = self.duration
        noise_records_df['type'] = self.noise_type.value

        noise_records = NoiseRecords(df=noise_records_df)

        return noise_records


class ElectronicNoiseGenerator(NoiseGenerator):
    """Generates noise according to detector properties."""

    def __init__(
            self,
            *args,
            **kwargs
    ):
        """Constructor of the Electronic Noise."""
        super().__init__(noise_type=NoiseType.ELECTRICAL, *args, **kwargs)

    def _generate_hits(self, records: NoiseRecords) -> Hits:
        """Generates hits based on noise records.

        Args:
            records: Records to generate hits for

        Returns:
            hits of the records
        """
        noise_rates = self.detector.pmt_noise_rates
        number_of_records = len(records)
        number_of_pmts = len(self.detector)
        pmt_indices = self.detector.indices
        noise_rates_per_duration = noise_rates * self.duration
        poisson_number_of_photons = self.rng.poisson(
            noise_rates_per_duration,
            size=(number_of_records, number_of_pmts)
        ).T

        non_zero_indizes = np.asarray(poisson_number_of_photons > 0).nonzero()
        number_of_photons = poisson_number_of_photons[non_zero_indizes]

        end_time = self.start_time + self.duration

        hits = self.rng.uniform(self.start_time, end_time, np.sum(number_of_photons))

        hits_df = pd.DataFrame(
            {
                'time': hits,
                'pmt_id': 0,
                'record_id': 0,
                'string_id': 0,
                'module_id': 0,
            }
        )

        pmt_id_loc = hits_df.columns.get_loc('pmt_id')
        string_id_loc = hits_df.columns.get_loc('string_id')
        module_id_loc = hits_df.columns.get_loc('module_id')
        record_id_loc = hits_df.columns.get_loc('record_id')

        iterator = np.nditer(number_of_photons, flags=['f_index'])

        hits_index = 0

        for nop_per_pmt_and_record in iterator:
            current_record = records.df.iloc[non_zero_indizes[1][iterator.index]]
            current_pmt = pmt_indices.iloc[non_zero_indizes[0][iterator.index]]
            current_slice = slice(hits_index, nop_per_pmt_and_record + hits_index)
            hits_df.iloc[current_slice, pmt_id_loc] = current_pmt['pmt_id']
            hits_df.iloc[current_slice, module_id_loc] = current_pmt['module_id']
            hits_df.iloc[current_slice, string_id_loc] = current_pmt['string_id']
            hits_df.iloc[current_slice, record_id_loc] = current_record['record_id']
            hits_index += nop_per_pmt_and_record

        return Hits(df=hits_df)

    def generate(self, number_of_samples: int) -> Collection:
        """Generate Noise Collection with a given Number of samples.

        Args:
            number_of_samples: Amount of Records to be generated

        Returns:
            Collection with all the generated data.
        """
        records = self.generate_records(number_of_samples)
        hits = self._generate_hits(records)
        collection = Collection(
            detector=self.detector,
            records=records,
            hits=hits
        )

        return collection


class GeneratorFactory:
    def __init__(
            self,
            detector: Detector,
            seed: int = defaults['seed'],
            photon_propagator: AbstractPhotonPropagator = None
    ) -> None:
        self._builders = {
            "track": TrackEventGenerator,
            "starting_track": StartingTrackEventGenerator,
            "cascade": CascadeGenerator,
            # "noise": RandomNoiseGenerator
        }
        self.detector = detector
        self.photon_propagator = photon_propagator
        self.seed = seed

    def register_builder(self, key: str, builder: Type[AbstractGenerator]):
        self._builders[key] = builder

    def create(self, key, **kwargs) -> AbstractGenerator:
        builder = self._builders.get(key)
        if not builder:
            raise ValueError(key)
        return builder(
            detector=self.detector,
            photon_propagator=self.photon_propagator,
            seed=self.seed,
            **kwargs
        )


class GeneratorCollection:
    def __init__(self) -> None:
        self.generators = []

    def add(self, generator: AbstractGenerator):
        self.generators.append(generator)

    def generate(self, **kwargs) -> Collection:
        events_list: List[Collection] = []

        for generator in self.generators:
            events_list.append(generator.generate(**kwargs))

        return Collection.concat(events_list)
