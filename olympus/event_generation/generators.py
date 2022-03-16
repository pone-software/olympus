from abc import ABC, abstractmethod
from tkinter.messagebox import NO
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

from .data import EventData
from .detector import Detector, sample_direction, generate_noise
from .constants import defaults
from .utils import get_event_times_by_rate


class AbstractGenerator(ABC):
    @abstractmethod
    def generate(self, start_time=0, end_time=None, **kwargs) -> Tuple[List, List]:
        pass

    @staticmethod
    def generate_histogram(events, records, stepsize=50):
        concatenated_events = ak.sort(ak.concatenate(events, axis=1))
        max = int(np.ceil(ak.max(concatenated_events)))
        min = int(np.floor(ak.min(concatenated_events)))
        bins = int(np.ceil((max - min) / stepsize))

        if bins == 0:
            bins = 10

        histograms = []

        for module in concatenated_events:
            histograms.append(np.histogram(module, bins=bins, range=(min, max))[0])

        return np.array(histograms), records


class EventGenerator(AbstractGenerator):
    def __init__(
        self,
        injector: Type[AbstractInjector],
        spectrum: Type[AbstractSpectrum],
        propagator: Type[AbstractPropagator],
        seed: Optional[int] = defaults["seed"],
        rate: Optional[float] = None,
        rng: Optional[np.random.RandomState] = defaults["rng"],
        particle_id: Optional[int] = None,
        **kwargs
    ) -> None:
        self.injector = injector
        self.spectrum = spectrum
        self.propagator = propagator
        self.seed = seed
        self.rng = rng
        self.rate = rate
        self.particle_id = particle_id

    def generate(
        self,
        n_samples: Optional[int] = None,
        start_time: Optional[int] = 0,
        end_time: Optional[int] = None,
    ) -> Tuple[List, List]:

        """Generate realistic muon tracks."""
        key, subkey = random.split(random.PRNGKey(self.seed))

        events = []
        records = []

        if n_samples is None and (self.rate is None or end_time is None):
            raise ValueError("Either number of samples or time parameters must be set")

        time_based = n_samples is None

        if time_based:
            if self.rate is None:
                rate = self.rate
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
                key=subkey,
                time=start_time,
                particle_id=self.particle_id,
            )

            event, record = self.propagator.propagate(event_data=event_data)

            if len(event) > 0:
                events.append(event)
                records.append(record)

        return events, records

    def generate_per_timeframe(
        self, start_time: int, end_time: int, rate: Optional[float] = None
    ):
        return self._generate(start_time=start_time, end_time=end_time, rate=rate)

    def generate_nsamples(self, nsamples: int, start_time: Optional[int] = 0):
        return self._generate(nsamples=nsamples, start_time=start_time)


class CascadeEventGenerator(EventGenerator):
    def __init__(
        self, detector: Detector, photon_propagator: callable, **kwargs
    ) -> None:
        injector = VolumeInjector(detector=detector, **kwargs)
        spectrum = UniformSpectrum(
            log_maximal_energy=kwargs["log_maximal_energy"],
            log_minimal_energy=kwargs["log_minimal_energy"],
        )
        propagator = CascadePropagator(
            detector=detector, photon_propagator=photon_propagator, **kwargs
        )
        super().__init__(injector, spectrum, propagator, **kwargs)


class TrackEventGenerator(EventGenerator):
    def __init__(
        self, detector: Detector, photon_propagator: callable, **kwargs
    ) -> None:
        injector = SurfaceInjector(detector=detector, **kwargs)
        spectrum = UniformSpectrum(
            log_maximal_energy=kwargs["log_maximal_energy"],
            log_minimal_energy=kwargs["log_minimal_energy"],
        )
        propagator = TrackPropagator(
            detector=detector, photon_propagator=photon_propagator, **kwargs
        )
        super().__init__(injector, spectrum, propagator, **kwargs)


class StartingTrackEventGenerator(EventGenerator):
    def __init__(
        self, detector: Detector, photon_propagator: callable, **kwargs
    ) -> None:
        injector = VolumeInjector(detector=detector, **kwargs)
        spectrum = UniformSpectrum(
            log_maximal_energy=kwargs["log_maximal_energy"],
            log_minimal_energy=kwargs["log_minimal_energy"],
        )
        propagator = StartingTrackPropagator(
            detector=detector, photon_propagator=photon_propagator, **kwargs
        )
        super().__init__(injector, spectrum, propagator, **kwargs)


class RandomNoiseGenerator(AbstractGenerator):
    def __init__(
        self,
        detector: Detector,
        rng: Optional[np.random.RandomState] = defaults["rng"],
        **kwargs
    ) -> None:
        super().__init__()
        self.detector = detector
        self.rng = rng

    def generate(self, start_time=0, end_time=None) -> Tuple[List, List]:
        time_range = [start_time, end_time]
        return [generate_noise(self.detector, time_range, self.rng)], []


class GeneratorFactory:
    def __init__(self, detector: Detector, photon_propagator: callable) -> None:
        self._builders = {
            "track": TrackEventGenerator,
            "starting_track": StartingTrackEventGenerator,
            "cascade": CascadeEventGenerator,
            "noise": RandomNoiseGenerator,
        }
        self.detector = detector
        self.photon_propagator = photon_propagator

    def register_builder(self, key: str, builder: Type[AbstractGenerator]):
        self._builders[key] = builder

    def create(self, key, **kwargs) -> Type[AbstractGenerator]:
        builder = self._builders.get(key)
        if not builder:
            raise ValueError(key)
        return builder(
            detector=self.detector, photon_propagator=self.photon_propagator, **kwargs
        )


class GeneratorCollection:
    def __init__(self) -> None:
        self.generators = []

    def add_generator(self, generator: Type[AbstractGenerator]):
        self.generators.append(generator)

    def generate(self, **kwargs) -> Tuple[List, List]:
        events = []
        records = []

        for generator in self.generators:
            generator_events, generator_records = generator.generate(**kwargs)
            events += generator_events
            records += generator_records

        return events, records

    def generate_per_timeframe(self, start_time: int, end_time: int):
        return self.generate(start_time=start_time, end_time=end_time)

    def generate_nsamples(self, nsamples: int):
        return self.generate(nsamples=None)
