"""Module containing all event generators."""
import json
import logging
import os
from abc import ABC
from math import ceil
from multiprocessing import Pool, get_context
from typing import Type, Dict, Any, Union, Optional, List, Tuple

import pandas as pd
import numpy as np
import pyarrow as pa
from tqdm import tqdm

from ananke.models.detector import Detector
from ananke.models.collection import Collection
from ananke.models.event import (
    EventRecords,
    NoiseRecords,
    Hits,
)
from ananke.models.geometry import Vectors3D
from ananke.schemas.event import EventType, NoiseType, RecordType, HitSchema
from ananke.services.detector import DetectorBuilderService
from ananke.utils import save_configuration
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
    DatasetConfiguration, DatasetStatus, BioluminescenceGeneratorConfiguration,
)
from ..generators import (
    AbstractGenerator,
)


def generate_hit_per_file(
        file: str,
        record_id: int,
        record_type: RecordType,
        start_time: float,
        duration: float,
        string_module_indices: pd.DataFrame,
        rng: np.random.Generator
) -> Optional[Hits]:
    try:
        metadata: Optional[dict] = None
        dataset: Optional[pd.DataFrame] = None
        with pa.ipc.open_file(file) as reader:
            dataset = reader.read_pandas()
            metadata = json.loads(reader.schema.metadata[b'metadata_json'])

        number_of_hits = len(dataset.index)

        file_start_time = metadata['sources'][0]['time_range'][0]
        file_end_time = metadata['sources'][0]['time_range'][1]
        dataset['choice'] = rng.uniform(low=0.0, high=1.0, size=number_of_hits)
        dataset = dataset[dataset['total_weight'] >= dataset['choice']]

        hits_list: List[Hits] = []

        for indices in string_module_indices.itertuples():
            pick_start_time = rng.uniform(
                file_start_time,
                file_end_time - duration
            )
            pick_end_time = pick_start_time + duration
            module_hits_df = dataset[
                (dataset['time'] >= pick_start_time) &
                (dataset['time'] < pick_end_time)
                ].copy()

            if len(module_hits_df.index) == 0:
                continue

            module_hits_df['string_id'] = getattr(indices, 'string_id')
            module_hits_df['module_id'] = getattr(indices, 'module_id')
            module_hits_df['record_id'] = record_id
            module_hits_df['type'] = record_type.value
            module_hits_df['time'] = module_hits_df['time'] - pick_start_time + start_time

            module_hits = Hits(
                df=module_hits_df[[
                    'time',
                    'record_id',
                    'type',
                    'string_id',
                    'module_id',
                    'pmt_id'
                ]]
            )

            hits_list.append(module_hits)

        return Hits.concat(hits_list)
    except Exception as exc:
        logging.error(exc)
        return None


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

    def generate_records(
            self,
            collection: Collection,
            number_of_samples: int,
    ) -> None:
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

        collection.set_records(event_records, append=True)

    def generate_sources(
            self,
            collection: Collection,
    ) -> None:
        """Generates and sets the sources to further evaluate.

        Args:
            collection: collection to generate for
        """
        record_without_sources = collection.get_records_with_sources(
            record_type=self.record_type,
            invert=True
        )
        if record_without_sources is None:
            raise ValueError('No records to generate sources')
        event_records = EventRecords(df=record_without_sources.df)
        sources = self.propagator.propagate(records=event_records)
        collection.set_sources(sources)

    def generate_hits(
            self,
            collection: Collection,
    ) -> None:
        """Generates and sets the hits to further evaluate.

        Args:
            collection: collection to generate for
        """
        if self.photon_propagator is None:
            raise ValueError('Photon Propagator is not defined')
        self.photon_propagator.propagate(
            collection=collection,
            record_type=self.record_type
        )


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

    def generate_records(
            self,
            collection: Collection,
            number_of_samples: int,
    ) -> None:
        """Generic generator of noise records.

        Args:
            collection: collection to generate for
            number_of_samples: Amount of samples to be generated
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

        collection.set_records(records=noise_records, append=True)


class ElectronicNoiseGenerator(NoiseGenerator):
    """Generates noise according to detector properties."""

    def generate_hits(
            self,
            collection: Collection,
    ) -> None:
        """Generates hits based on noise records.

        Args:
            collection: collection to generate for
        """
        noise_rates = self.detector.pmt_noise_rates
        records = collection.get_records_with_hits(
            record_type=self.record_type,
            invert=True
        )
        if records is None:
            raise ValueError('No records to generate hits')
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

        hits = Hits(df=hits_df)

        collection.set_hits(hits=hits)


class JuliaBioluminescenceGenerator(NoiseGenerator):
    """Generator for Bioluminescence Records based on C. Haacks Julia Hits."""

    configuration: BioluminescenceGeneratorConfiguration

    def __init__(
            self,
            *args,
            **kwargs
    ):
        """Constructor for the Julia Bioluminescence Generator

        Args:
            args: Args to pass to Noise Generator
            kwargs: KwArgs to pass to Noise Generator
        """
        super().__init__(
            *args,
            **kwargs
        )
        self.hits_paths = self.__get_hits_paths()

    def __get_hits_paths(self) -> List[str]:
        directory = os.fsencode(self.configuration.julia_data_path)
        hits_paths = []
        for file in os.listdir(directory):
            file_decoded = os.fsdecode(file)
            filename = os.path.join(
                self.configuration.julia_data_path,
                file_decoded
            )
            if file_decoded.endswith(".arrow"):
                number_of_sources = int(file_decoded.split('_')[1])
                if self.configuration.number_of_sources is None \
                        or self.configuration.number_of_sources == number_of_sources:
                    hits_paths.append(filename)

        return hits_paths

    def generate_records(
            self,
            collection: Collection,
            number_of_samples: int,
    ) -> None:
        """Generates and sets the records to further evaluate.

        Args:
            collection: collection to generate for
            number_of_samples: Amount of samples to be generated
        """
        # TODO: Implement per sample draft of paper
        super().generate_records(
            collection=collection,
            number_of_samples=number_of_samples
        )

    def generate_hits(
            self,
            collection: Collection,
    ) -> None:
        """Generates and sets the hits to further evaluate.

        Args:
            collection: collection to generate for
        """
        logging.info('Starting to generate {} hits'.format(self.record_type))
        records = collection.get_records_with_hits(
            record_type=self.record_type,
            invert=True
        )
        number_of_records = len(records)
        string_module_indices = self.detector.df[self.detector.id_columns[0:2]]
        string_module_indices.drop_duplicates(inplace=True)

        multiprocessing_args = []

        for index, record_id in records.record_ids.items():
            current_file = str(self.rng.choice(self.hits_paths))
            multiprocessing_args.append(
                (
                    current_file,
                    record_id,
                    self.record_type,
                    self.configuration.start_time,
                    self.configuration.duration,
                    string_module_indices,
                    self.rng
                )
            )
        batch_size = self.configuration.batch_size
        with tqdm(total=ceil(number_of_records / batch_size), mininterval=.5) as pbar:
            for idx in range(0, number_of_records, batch_size):
                with Pool() as pool:
                    result = pool.starmap(
                            generate_hit_per_file,
                            multiprocessing_args[idx:idx + batch_size]
                    )
                    result_hits = Hits.concat(result)
                    if result_hits is not None:
                        collection.set_hits(hits=result_hits)
                pbar.update()
            # logging.info('Starting to generate {} hits'.format(self.record_type))


generators: Dict[Any, Type[AbstractGenerator]] = {
    EventType.REALISTIC_TRACK: TrackEventGenerator,
    EventType.STARTING_TRACK: StartingTrackEventGenerator,
    EventType.CASCADE: CascadeEventGenerator,
    NoiseType.ELECTRICAL: ElectronicNoiseGenerator,
    NoiseType.BIOLUMINESCENCE: JuliaBioluminescenceGenerator,
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
    def _local_save_configuration() -> None:
        """Saves current configuration."""
        save_configuration(configuration_path, configuration)

    os.makedirs(configuration.data_path, exist_ok=True)
    configuration_path = os.path.join(configuration.data_path, 'configuration.json')

    if configuration is None:
        configuration = DatasetConfiguration.parse_file(configuration_path)
    else:
        _local_save_configuration()
    data_path = os.path.join(configuration.data_path, 'data.h5')

    collection = Collection(data_path)

    if detector is None:
        detector_service = DetectorBuilderService()
        detector = detector_service.get(configuration=configuration.detector)

    if collection.get_detector() is None:
        collection.set_detector(detector)
    else:
        collection_detector = collection.get_detector()
        if not collection_detector.df.equals(detector.df):
            raise ValueError('Cannot generate on top of different detector')

    if configuration.status.value == DatasetStatus.NOT_STARTED:
        configuration.status.value = DatasetStatus.STARTED
        _local_save_configuration()

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

            generator.generate(
                collection,
                number_of_samples=generator_config.number_of_samples,
                drop_empty_records=generator_config.drop_empty_records
            )
            configuration.status.current_index = index + 1
            _local_save_configuration()
    except Exception as err:
        configuration.status.value = DatasetStatus.ERROR
        configuration.status.error_message = '[{}] {}'.format(
            type(err).__name__,
            str(err)
        )
        _local_save_configuration()
        raise err

    configuration.status.value = DatasetStatus.COMPLETE
    _local_save_configuration()

    return collection
