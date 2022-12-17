import uuid
from abc import ABC, abstractmethod
from typing import Optional, Type, List
import numpy as np
import awkward as ak
import pandas as pd
from tqdm.auto import trange

from ananke.models.detector import Detector
from ananke.models.event import Events, EventRecords
from ananke.models.geometry import Vectors3D
from ananke.schemas.event import EventType
from .injectors import (
    AbstractInjector,
    SurfaceInjector,
    VolumeInjector,
)
from .photon_propagation.interface import AbstractPhotonPropagator
from .propagator import (
    AbstractPropagator,
    CascadePropagator,
    StartingTrackPropagator,
    TrackPropagator,
)
from .spectra import AbstractSpectrum, UniformSpectrum

from .detector import sample_direction
from .constants import defaults
from .utils import get_event_times_by_rate


# TODO: Consider changing everything to event collection


class AbstractGenerator(ABC):
    def __init__(
            self,
            seed: Optional[int] = defaults["seed"],
            rng: Optional[np.random.RandomState] = defaults["rng"],
            *args,
            **kwargs
    ):
        super().__init__()
        self.seed = seed
        self.rng = rng

    @abstractmethod
    def generate(self, start_time=0, end_time=None, **kwargs) -> Events:
        pass

    def generate_per_timeframe(
            self, start_time: int, end_time: int, rate: Optional[float] = None
    ) -> Events:
        return self.generate(start_time=start_time, end_time=end_time, rate=rate)

    def generate_nsamples(
            self,
            nsamples: int,
            start_time: Optional[int] = 0
    ) -> Events:
        return self.generate(nsamples=nsamples, start_time=start_time)


class EventGenerator(AbstractGenerator):
    def __init__(
            self,
            injector: AbstractInjector,
            spectrum: AbstractSpectrum,
            propagator: AbstractPropagator,
            event_type: EventType,
            rate: Optional[float] = None,
            particle_id: Optional[int] = None,
            *args,
            **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.injector = injector
        self.spectrum = spectrum
        self.propagator = propagator
        self.rate = rate
        self.event_type = event_type
        self.particle_id = particle_id

    def generate(
            self,
            n_samples: Optional[int] = None,
            start_time: Optional[int] = 0,
            end_time: Optional[int] = None,
    ) -> Events:
        """Generate realistic muon tracks."""

        if n_samples is None and (self.rate is None or end_time is None):
            raise ValueError("Either number of samples or time parameters must be set")

        time_based = n_samples is None

        if time_based:
            times_det = get_event_times_by_rate(
                rate=self.rate, start_time=start_time, end_time=end_time, rng=self.rng
            )

            start_times = ak.sort(ak.Array(times_det))
            n_samples = len(start_times)

        track_length = 3000
        iterator_range = trange(n_samples)

        locations = self.injector.get_positions(n=n_samples)
        energies = self.spectrum.get_energy(n=n_samples)
        orientations = Vectors3D.from_numpy(
            sample_direction(n_samples=n_samples, rng=self.rng)
            )
        ids = [uuid.uuid1().int >> 64 for x in range(n_samples)]
        event_records_df = pd.concat(
            [
                locations.get_df_with_prefix('location_'),
                orientations.get_df_with_prefix('orientation_'),
            ],
            axis=1
        )

        event_records_df['event_id'] = ids
        event_records_df['energy'] = energies
        event_records_df['length'] = track_length
        event_records_df['time'] = start_time
        event_records_df['type'] = self.event_type.value
        event_records_df['particle_id'] = self.particle_id

        event_records = EventRecords(df=event_records_df)

        events = self.propagator.propagate(event_records)

        # TODO: Drop Empty events

        return events


class CascadeEventGenerator(EventGenerator):
    def __init__(
            self,
            detector: Detector,
            photon_propagator: callable,
            *args,
            **kwargs
    ):
        injector = VolumeInjector(detector=detector, **kwargs)
        spectrum = UniformSpectrum(
            log_maximal_energy=kwargs["log_maximal_energy"],
            log_minimal_energy=kwargs["log_minimal_energy"],
        )
        propagator = CascadePropagator(
            detector=detector, photon_propagator=photon_propagator, **kwargs
        )
        super().__init__(
            event_type=EventType.CASCADE,
            injector=injector, spectrum=spectrum, propagator=propagator, *args, **kwargs
        )


class TrackEventGenerator(EventGenerator):
    def __init__(
            self, detector: Detector, photon_propagator: callable,
            *args,
            **kwargs
    ):
        injector = SurfaceInjector(detector=detector, **kwargs)
        spectrum = UniformSpectrum(
            log_maximal_energy=kwargs["log_maximal_energy"],
            log_minimal_energy=kwargs["log_minimal_energy"],
        )
        propagator = TrackPropagator(
            detector=detector, photon_propagator=photon_propagator, **kwargs
        )
        super().__init__(
            event_type=EventType.REALISTIC_TRACK,
            injector=injector, spectrum=spectrum, propagator=propagator, *args, **kwargs
        )


class StartingTrackEventGenerator(EventGenerator):
    def __init__(
            self, detector: Detector, photon_propagator: callable,
            *args,
            **kwargs
    ):
        injector = VolumeInjector(detector=detector, **kwargs)
        spectrum = UniformSpectrum(
            log_maximal_energy=kwargs["log_maximal_energy"],
            log_minimal_energy=kwargs["log_minimal_energy"],
        )
        propagator = StartingTrackPropagator(
            detector=detector, photon_propagator=photon_propagator, **kwargs
        )
        super().__init__(
            event_type=EventType.STARTING_TRACK,
            injector=injector, spectrum=spectrum, propagator=propagator, *args, **kwargs
        )


# class RandomNoiseGenerator(AbstractGenerator):
#     def __init__(
#             self,
#             detector: Detector,
#             *args,
#             **kwargs
#     ):
#         super().__init__(*args, **kwargs)
#         self.detector = detector
#
#     def generate(self, start_time=0, end_time=None, **kwargs) -> Tuple[List, List]:
#         time_range = [start_time, end_time]
#         return [generate_noise(self.detector, time_range, self.rng)], [MCRecord(
#             'noise', [], EventRecord(
#                 orientation=Vector3D(x=0, y=0, z=0),
#                 energy=0.0,
#                 location=Vector3D(x=0, y=0, z=0),
#                 time=0
#             )
#         )]


class GeneratorFactory:
    def __init__(
            self,
            detector: Detector,
            photon_propagator: AbstractPhotonPropagator = None
    ) -> None:
        self._builders = {
            "track": TrackEventGenerator,
            "starting_track": StartingTrackEventGenerator,
            "cascade": CascadeEventGenerator,
            # "noise": RandomNoiseGenerator
        }
        self.detector = detector
        self.photon_propagator = photon_propagator

    def register_builder(self, key: str, builder: Type[AbstractGenerator]):
        self._builders[key] = builder

    def create(self, key, **kwargs) -> AbstractGenerator:
        builder = self._builders.get(key)
        if not builder:
            raise ValueError(key)
        return builder(
            detector=self.detector, photon_propagator=self.photon_propagator, **kwargs
        )


class GeneratorCollection:
    def __init__(self, detector: Detector = None) -> None:
        self.generators = []
        self.detector = detector

    def add_generator(self, generator: AbstractGenerator):
        self.generators.append(generator)

    def generate(self, **kwargs) -> Events:
        events_list: List[Events] = []

        for generator in self.generators:
            events_list.append(generator.generate(**kwargs))

        return Events.concat(events_list)

    def generate_per_timeframe(self, start_time: int, end_time: int) -> Events:
        return self.generate(start_time=start_time, end_time=end_time)

    def generate_nsamples(self, nsamples: int) -> Events:
        return self.generate(nsamples=nsamples)
