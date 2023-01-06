import uuid
from typing import Optional, Type, List

import pandas as pd

from ananke.models.detector import Detector
from ananke.models.event import EventRecords, Sources, Collection
from ananke.models.geometry import Vectors3D
from ananke.schemas.event import EventType
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


class EventGenerator(AbstractGenerator):
    def __init__(
            self,
            event_type: EventType,
            particle_id: int,
            injector: AbstractInjector,
            propagator: AbstractPropagator,
            photon_propagator: AbstractPhotonPropagator,
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
        ids = [uuid.uuid1().int >> 64 for x in range(number_of_samples)]
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


class GeneratorFactory:
    def __init__(
            self,
            detector: Detector,
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

    def add(self, generator: AbstractGenerator):
        self.generators.append(generator)

    def generate(self, **kwargs) -> Collection:
        events_list: List[Collection] = []

        for generator in self.generators:
            events_list.append(generator.generate(**kwargs))

        return Collection.concat(events_list)
