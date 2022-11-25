from abc import ABC, abstractmethod
from typing import Optional, Tuple, Type, List
import numpy as np
import awkward as ak
from tqdm.auto import trange

from ananke.models.event import EventRecord, EventType, EventCollection
from ananke.models.geometry import Vector3D
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

from .mc_record import MCRecord
from .detector import Detector, sample_direction, generate_noise
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
    def generate(self, start_time=0, end_time=None, **kwargs) -> EventCollection:
        pass

    def generate_per_timeframe(
            self, start_time: int, end_time: int, rate: Optional[float] = None
    ) -> EventCollection:
        return self.generate(start_time=start_time, end_time=end_time, rate=rate)

    def generate_nsamples(
            self,
            nsamples: int,
            start_time: Optional[int] = 0
            ) -> EventCollection:
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
    ) -> EventCollection:
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

        events = EventCollection

        positions = self.injector.get_position(n=n_samples)
        energies = self.spectrum.get_energy(n=n_samples)
        directions = sample_direction(n_samples=n_samples, rng=self.rng)

        for i in iterator_range:

            if time_based:
                start_time = start_times[i]

            event_data = EventRecord(
                type=self.event_type,
                location=Vector3D.from_numpy(positions[i]),
                energy=energies[i],
                orientation=Vector3D.from_numpy(directions[i]),
                length=track_length,
                time=start_time,
                particle_id=self.particle_id,
            )

            event = self.propagator.propagate(event_record=event_data)

            if len(event.hits) > 0:
                events.append(event)

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


class RandomNoiseGenerator(AbstractGenerator):
    def __init__(
            self,
            detector: Detector,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.detector = detector

    def generate(self, start_time=0, end_time=None, **kwargs) -> Tuple[List, List]:
        time_range = [start_time, end_time]
        return [generate_noise(self.detector, time_range, self.rng)], [MCRecord(
            'noise', [], EventRecord(
                orientation=Vector3D(x=0, y=0, z=0),
                energy=0.0,
                location=Vector3D(x=0, y=0, z=0),
                time=0
            )
        )]


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
            "noise": RandomNoiseGenerator
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

    def generate(self, **kwargs) -> EventCollection:
        event_collection = EventCollection()

        for generator in self.generators:
            event_collection += generator.generate(**kwargs)

        return event_collection

    def generate_per_timeframe(self, start_time: int, end_time: int) -> EventCollection:
        return self.generate(start_time=start_time, end_time=end_time)

    def generate_nsamples(self, nsamples: int) -> EventCollection:
        return self.generate(nsamples=nsamples)
