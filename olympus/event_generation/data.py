from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Union, Tuple
import logging
import os

import pickle

import numpy as np
import awkward as ak
import pandas as pd
from awkward.partition import every

from .detector import Detector
from .mc_record import MCRecord
from .constants import defaults
from .photon_source import PhotonSource


@dataclass
class EventData:
    direction: np.ndarray
    energy: np.ndarray
    time: int
    start_position: np.ndarray
    key: Optional[str] = None
    length: Optional[float] = None
    particle_id: Optional[int] = None

@dataclass
class EventRecord:
    type: str
    sources: List[PhotonSource]
    data: EventData

    @classmethod
    def from_mc_record(cls, mc_record: MCRecord) -> EventRecord:
        info = mc_record.mc_info[0]
        if not isinstance(info, EventData):
            info = EventData(
                direction=info["dir"],
                energy=info["energy"],
                time=info["time"],
                start_position=info["pos"],
            )
        return cls(type=mc_record.event_type, sources=mc_record.sources, data=info)

    def to_dict(self):
        return {
            'type' : self.type,
            'energy': self.data.energy,
            'time': self.data.time,
            'pos_x': self.data.start_position[0],
            'pos_y': self.data.start_position[1],
            'pos_z': self.data.start_position[2],
            'dir_x': self.data.direction[0],
            'dir_y': self.data.direction[1],
            'dir_z': self.data.direction[2],
        }


@dataclass
class EventCollection:
    events: List[ak.Array]
    records: List[Union[EventRecord, MCRecord]]
    detector: Optional[Detector] = None

    rng: Optional[np.random.RandomState] = defaults["rng"]
    seed: Optional[int] = defaults["seed"]

    def __post_init__(self):
        for x in range(len(self.records)):
            if isinstance(self.records[x], MCRecord):
                self.records[x] = EventRecord.from_mc_record(self.records[x])

    def __len__(self):
        return len(self.events)

    def __getitem__(self, item):
        return EventCollection(detector=self.detector, events=self.events[item], records=self.records[item], rng=self.rng, seed=self.seed)

    def generate_histogram(self, step_size=50, start_time=None, end_time=None) -> Tuple[np.ndarray, pd.DataFrame, np.ndarray]:

        events_to_account = []
        records_to_account = []
        if self.detector is None:
            raise ValueError('detector is not provided.')
        number_of_modules = len(self.detector.modules)

        if start_time is not None and end_time is not None:
            short_collection = self.get_within_timeframe(start_time, end_time)
            events_to_enumerate = short_collection.events
            records_to_enumerate = short_collection.records
        else:
            events_to_enumerate = self.events
            records_to_enumerate = self.records

        for event_loop_index, event in enumerate(events_to_enumerate):
            if len(event) == number_of_modules:
                events_to_account.append(event)
                records_to_account.append(records_to_enumerate[event_loop_index])

        if not len(events_to_account):
            logging.warning('No events to generate Histogram')

        if start_time is None:
            start_time = int(np.floor(ak.min(events_to_account)))
        if end_time is None:
            end_time = int(np.ceil(ak.max(events_to_account)))
        bins = int(np.ceil((end_time - start_time) / step_size))

        if bins == 0:
            bins = 10

        # Ensure second dimension is at number of modules
        regular_array = ak.to_regular(events_to_account, axis=1)

        partitioned_events_to_account = ak.repartition(regular_array, 100, highlevel=False)

        histogram = np.zeros([number_of_modules, bins])
        timings = np.arange(0, bins * step_size, step_size)

        for partition in every(partitioned_events_to_account):
            # Ensure second dimension is at number of modules
            regular_array = ak.to_regular(partition, axis=1)

            concatenated_events = ak.concatenate(regular_array, axis=1)

            for module_index, module in enumerate(concatenated_events):
                histogram[module_index] += np.histogram(ak.to_numpy(module), bins=bins, range=(start_time, end_time))[0]

        slim_records_data = self.get_info_as_panda(records_to_account)
        times = np.arange(start_time, end_time, step_size)

        return histogram, slim_records_data, times

    def redistribute(
        self,
        start_time: int,
        end_time: int = None,
        rate: float = None,
        rng: np.random.RandomState = None,
    ):
        nr_events = len(self)
        logging.info('Start redistribute events (start %s, end %s, rate %s)', start_time, end_time, rate)

        if rng is None:
            rng = self.rng

        if rate is not None:
            if end_time is not None:
                raise ValueError('Rate and End Time cannot be provided at the same time')
            end_time = start_time + np.ceil(nr_events / rate)

        if end_time is not None:
            times_det = rng.random_integers(start_time, end_time, nr_events)
        else:
            raise ValueError('Either rate or end time have to be provided')

        for index, record in enumerate(self.records):
            start_time = record.data.time
            end_time = times_det[index]

            for source in record.sources:
                source.time = source.time - start_time + end_time

            self.events[index] = self.events[index] - start_time + end_time
            record.data.time = end_time
        logging.info('Finish redistribute events (start %s, end %s, rate %s)', start_time, end_time, rate)

    def get_within_timeframe(self, start_time: int, end_time: int = None) -> EventCollection:
        events = []
        records = []

        logging.info('Start collecting events within %d and %d (%d Events available)', start_time, end_time, len(self))
        for index, record in enumerate(self.records):
            record_time = record.data.time
            if start_time <= record_time and (end_time is None or record_time < end_time):
                events.append(self.events[index])
                records.append(record)

        logging.info('Finish collecting events within %d and %d (%d Events collected)', start_time, end_time, len(records))


        return EventCollection(records=records, events=events, detector=self.detector, rng=self.rng, seed=self.seed)

    def get_info_as_panda(self, records=None) -> pd.DataFrame:
        records_list = []
        if records is None:
            records = self.records
        for record in records:
            records_list.append(record.to_dict())

        return pd.DataFrame(records_list)

    def save(self, path, batch_size: int=100, filename: str ='part_{index}.pickle'):
        self.to_folder(path, batch_size, filename)
        df = self.get_info_as_panda()
        df.to_csv(os.path.join(path, 'records.csv'))

    def to_pickle(self, filename):
        file = open(filename, 'wb')
        pickle.dump(self, file)
        file.close()

    def to_folder(self, path, batch_size=100, filename='part_{index}.pickle'):
        logging.info('Starting to pickle to folder %s with batch_size %s', path, batch_size)

        is_exist = os.path.exists(path)

        if is_exist:
            logging.warning('Folder %s already exists', path)
        else:
            os.makedirs(path)

        nr_events = len(self)

        if not nr_events:
            logging.warning('Just pickled an empty collection.')

        loop_index = start_index = 0

        while start_index <= nr_events:
            batch_events = self.events[start_index:start_index + batch_size]
            batch_records = self.records[start_index:start_index + batch_size]
            current_collection = EventCollection(events=batch_events, records=batch_records, detector=self.detector, rng=self.rng, seed=self.seed)
            current_collection.to_pickle(os.path.join(path, filename.format(index = loop_index)))
            loop_index += 1
            start_index = loop_index * batch_size


    @classmethod
    def from_pickle(cls, filename) -> EventCollection:
        logging.info('Start to load file %s', filename)

        event_collection = cls.from_pickles([filename])
        logging.info('Finish to load file %s', filename)
        return event_collection

    @classmethod
    def from_pickles(cls, filenames: List):
        if len(filenames) == 0:
            logging.warning('Imported empty collection')
            return EventCollection(events=[], records=[])

        final_collection = None
        for filename in filenames:
            with open(filename, 'rb') as f:
                result = pickle.load(f)

            if not isinstance(result, EventCollection):
                result = EventCollection(events=result[0], records=result[1])

            if final_collection is None:
                final_collection = result
            else:
                if result.detector != final_collection.detector:
                    logging.warning('Imported multiple detectors')
                if result.seed != final_collection.seed:
                    logging.warning('Imported multiple seeds')
                final_collection.events += result.events
                final_collection.records += result.records

            logging.info('File %s loaded', f)

        return final_collection

    @classmethod
    def from_folder(cls, folder):
        filenames = []
        logging.info('Start to load folder %s', folder)
        for file in os.listdir(folder):
            if file.endswith(".pickle"):
                filenames.append(os.path.join(folder, file))

        event_collection = cls.from_pickles(filenames)
        logging.info('Finish to load folder %s', folder)

        return event_collection
