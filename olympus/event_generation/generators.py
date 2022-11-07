from abc import ABC, abstractmethod
from typing import Optional, Tuple, Type, List
from jax import random
import numpy as np
import awkward as ak
from tqdm.auto import trange

from .injectors import (
    AbstractInjector,
    SurfaceInjector,
    VolumeInjector,
)
from .propagator import (
    AbstractPropagator,
    CascadePropagator,
    StartingTrackPropagator,
    TrackPropagator,
)
from .spectra import AbstractSpectrum, UniformSpectrum

from .mc_record import MCRecord
from .data import EventData
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

    def get_key(self) -> str:
        """Generate realistic muon tracks."""
        key, subkey = random.split(random.PRNGKey(self.seed))

        return key

    @abstractmethod
    def generate(self, start_time=0, end_time=None, **kwargs) -> Tuple[List, List]:
        pass

    def generate_per_timeframe(
            self, start_time: int, end_time: int, rate: Optional[float] = None
    ) -> Tuple[List, List]:
        return self.generate(start_time=start_time, end_time=end_time, rate=rate)

    def generate_nsamples(self, nsamples: int, start_time: Optional[int] = 0) -> Tuple[List, List]:
        return self.generate(nsamples=nsamples, start_time=start_time)


class EventGenerator(AbstractGenerator):
    def __init__(
            self,
            injector: AbstractInjector,
            spectrum: AbstractSpectrum,
            propagator: AbstractPropagator,
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
        self.particle_id = particle_id

    def generate(
            self,
            n_samples: Optional[int] = None,
            start_time: Optional[int] = 0,
            end_time: Optional[int] = None,
    ) -> Tuple[List, List]:

        """Generate realistic muon tracks."""
        key = self.get_key()

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

        events = []
        records = []

        positions = self.injector.get_position(n=n_samples)
        energies = self.spectrum.get_energy(n=n_samples)
        directions = sample_direction(n_samples=n_samples, rng=self.rng)

        for i in iterator_range:

            if time_based:
                start_time = start_times[i]

            event_data = EventData(
                start_position=positions[i],
                energy=energies[i],
                direction=directions[i],
                length=track_length,
                key=key,
                time=start_time,
                particle_id=self.particle_id,
            )

            event, record = self.propagator.propagate(event_data=event_data)

            if len(event) > 0:
                events.append(event)
                records.append(record)

        return events, records


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
        super().__init__(injector, spectrum, propagator, *args, **kwargs)


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
        super().__init__(injector, spectrum, propagator, *args, **kwargs)


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
        super().__init__(injector, spectrum, propagator, *args, **kwargs)


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
        return [generate_noise(self.detector, time_range, self.rng)], [MCRecord('noise', [], EventData(
            key=self.get_key(),
            direction=np.zeros((3,)),
            energy=0.0,
            start_position=np.zeros((3,)),
            time=0
        ))]


class GeneratorFactory:
    def __init__(self, detector: Detector, photon_propagator: callable = None) -> None:
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

    def generate(self, **kwargs) -> Tuple[List, List]:
        events = []
        records = []

        for generator in self.generators:
            generator_events, generator_records = generator.generate(**kwargs)
            events += generator_events
            records += generator_records

        return events, records

    def generate_per_timeframe(self, start_time: int, end_time: int) -> Tuple[List, List]:
        return self.generate(start_time=start_time, end_time=end_time)

    def generate_nsamples(self, nsamples: int) -> Tuple[List, List]:
        return self.generate(nsamples=nsamples)
