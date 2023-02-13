"""Module containing all event generators."""
import logging
import os
import shutil
from abc import ABC
from typing import Type, Dict, Any, Union, Optional

import pandas as pd
import numpy as np

from ananke.models.detector import Detector
from ananke.models.collection import Collection
from ananke.models.event import (
    EventRecords,
    Sources,
    NoiseRecords,
    Hits,
)
from ananke.models.geometry import Vectors3D
from ananke.schemas.event import EventType, NoiseType, RecordType
from ananke.services.detector import DetectorBuilderService
from .detector import sample_direction
from .injectors import (
    AbstractInjector,
    SurfaceInjector,
    VolumeInjector,
)
from .photon_propagation.factory import get_photon_propagator
from .propagators import (
    AbstractPropagator,
    CascadePropagator,
    StartingTrackPropagator,
    TrackPropagator,
)
from .spectra import UniformSpectrum
from ..configuration.generators import (
    EventGeneratorConfiguration,
    NoiseGeneratorConfiguration,
    GeneratorConfiguration,
    DatasetConfiguration, DatasetStatus,
)
from ..generators import AbstractGenerator


class EventGenerator(AbstractGenerator[EventGeneratorConfiguration, EventType]):
    def __init__(
            self,
            injector: Type[AbstractInjector],
            propagator: Type[AbstractPropagator],
            *args,
            **kwargs
    ) -> None:
        super().__init__(
            *args,
            **kwargs
        )
        self.injector = injector(detector=self.detector)
        self.spectrum = UniformSpectrum(
            log_maximal_energy=self.configuration.spectrum.log_maximal_energy,
            log_minimal_energy=self.configuration.spectrum.log_minimal_energy,
            seed=self.configuration.seed
        )
        self.propagator = propagator(
            detector=self.detector,
            configuration=self.configuration.event_propagator,
            seed=self.configuration.seed
        )
        self.photon_propagator = get_photon_propagator(
            detector=self.detector,
            configuration=self.configuration.source_propagator
        )
        self.particle_id = self.configuration.particle_id

    def generate(
            self,
            collection: Collection,
            number_of_samples: int
    ) -> None:
        """Generate realistic events

        Args:
            collection: collection of the data
            number_of_samples: to generate
        """

        logging.info('Starting to generate {} samples'.format(number_of_samples))
        records = self.generate_records(number_of_samples)
        collection.set_records(records=records)
        collection.set_detector(detector=self.propagator.detector)
        sources = self.propagate(records)
        if self.photon_propagator is None:
            raise ValueError('Photon Propagator is not defined')
        collection.set_sources(sources=sources, cache=False)
        self.photon_propagator.propagate(collection=collection)

    def generate_records(self, number_of_samples: int) -> EventRecords:
        logging.info('Starting to generate {} records'.format(number_of_samples))
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
        event_records_df['time'] = self.configuration.start_time
        event_records_df['type'] = self.record_type.value
        event_records_df['particle_id'] = self.particle_id

        event_records = EventRecords(df=event_records_df)
        logging.info('Finished to generating {} records'.format(number_of_samples))

        return event_records

    def propagate(self, records: EventRecords) -> Sources:
        return self.propagator.propagate(records=records)


class CascadeEventGenerator(EventGenerator):
    def __init__(
            self,
            *args,
            **kwargs
    ):
        injector = VolumeInjector
        propagator = CascadePropagator
        super().__init__(
            *args,
            **kwargs,
            injector=injector,
            propagator=propagator
        )


class TrackEventGenerator(EventGenerator):
    def __init__(
            self,
            *args,
            **kwargs
    ):
        injector = SurfaceInjector
        propagator = TrackPropagator
        super().__init__(
            *args,
            **kwargs,
            injector=injector,
            propagator=propagator
        )


class StartingTrackEventGenerator(EventGenerator):
    def __init__(
            self,
            *args,
            **kwargs
    ):
        injector = VolumeInjector
        propagator = StartingTrackPropagator
        super().__init__(
            *args,
            **kwargs,
            injector=injector,
            propagator=propagator
        )


class NoiseGenerator(AbstractGenerator[NoiseGeneratorConfiguration, NoiseType], ABC):
    """Abstract parent for all Noise Generators."""

    def __init__(
            self,
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
        super().__init__(
            *args, **kwargs
        )
        self.start_time = self.configuration.start_time
        self.duration = self.configuration.duration

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
        noise_records_df['type'] = self.record_type.value

        noise_records = NoiseRecords(df=noise_records_df)

        return noise_records


class ElectronicNoiseGenerator(NoiseGenerator):
    """Generates noise according to detector properties."""

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
                'type': NoiseType.ELECTRICAL.value
            }
        )

        pmt_id_loc = hits_df.columns.get_loc('pmt_id')
        string_id_loc = hits_df.columns.get_loc('string_id')
        module_id_loc = hits_df.columns.get_loc('module_id')
        record_id_loc = hits_df.columns.get_loc('record_id')
        record_type_loc = hits_df.columns.get_loc('type')

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
            hits_df.iloc[current_slice, record_type_loc] = current_record['type']
            hits_index += nop_per_pmt_and_record

        return Hits(df=hits_df)

    def generate(
            self,
            collection: Collection,
            number_of_samples: int,
    ) -> Collection:
        """Generate Noise Collection with a given Number of samples.

        Args:
            collection_path: path to store the collection at
            number_of_samples: Amount of Records to be generated

        Returns:
            Collection with all the generated data.
        """
        records = self.generate_records(number_of_samples)
        hits = self._generate_hits(records)
        collection.set_detector(self.detector)
        collection.set_records(records=records)
        collection.set_hits(hits=hits)

        return collection


generators: Dict[Any, Type[AbstractGenerator]] = {
    EventType.REALISTIC_TRACK: TrackEventGenerator,
    EventType.STARTING_TRACK: StartingTrackEventGenerator,
    EventType.CASCADE: CascadeEventGenerator,
    NoiseType.ELECTRICAL: ElectronicNoiseGenerator,
}


def get_generator(
        detector: Detector,
        configuration: GeneratorConfiguration
) -> AbstractGenerator:
    try:
        generator_class: Type[AbstractGenerator] = generators[configuration.type]
    except:
        raise ValueError(
            '{} is not a valid generator type.'.format(configuration.type)
        )

    # TODO: Investigate potential side effects of "all records"
    return generator_class(
        record_type=RecordType(configuration.type),
        detector=detector,
        configuration=configuration
    )


def generate(
        configuration: Optional[DatasetConfiguration] = None,
        configuration_path: Optional[Union[str, bytes, os.PathLike]] = None,
        detector: Optional[Detector] = None,
) -> Collection:
    def save_configuration() -> None:
        """Saves current configuration."""
        with open(configuration_path, 'w') as f:
            f.write(configuration.json(indent=2))

    os.makedirs(configuration.data_path, exist_ok=True)

    if configuration is None:
        configuration = DatasetConfiguration.parse_file(configuration_path)
    else:
        configuration_path = os.path.join(configuration.data_path, 'configuration.json')
        save_configuration()
    data_path = os.path.join(configuration.data_path, 'data.h5')
    tmp_path = os.path.join(configuration.data_path, 'tmp')

    collection = Collection(data_path)

    if detector is None:
        detector_service = DetectorBuilderService()
        detector = detector_service.get(configuration=configuration.detector)
        collection.set_detector(detector)
    if configuration.status.value == DatasetStatus.NOT_STARTED:
        configuration.status.value = DatasetStatus.STARTED
        save_configuration()

    if configuration.status.value == DatasetStatus.COMPLETE:
        return collection

    try:
        iterations = range(
            configuration.status.current_index,
            len(configuration.generators)
        )
        for index in iterations:
            if index < configuration.status.current_index:
                continue
            generator_config = configuration.generators[index]
            generator = get_generator(
                detector=detector,
                configuration=generator_config.generator
            )
            generator_collection = Collection(
                data_path=os.path.join(
                    tmp_path,
                    '{index:n}.h5'.format(index=index)
                )
            )

            generator.generate(
                generator_collection,
                number_of_samples=generator_config.number_of_samples
            )
            collection.append(generator_collection)
            configuration.status.current_index = index + 1
            save_configuration()

        shutil.rmtree(tmp_path)
    except Exception as err:
        configuration.status.value = DatasetStatus.ERROR
        configuration.status.error_message = '[{}] {}'.format(
            type(err).__name__,
            str(err)
        )
        save_configuration()
        raise err

    configuration.status.value = DatasetStatus.COMPLETE
    save_configuration()

    return collection
