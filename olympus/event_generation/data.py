from __future__ import annotations
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from typing import Optional, List, Tuple, Type
import logging
import os

import pickle

import numpy as np
import awkward as ak
from awkward.partition import every

from jax import random

from .detector import Detector
from .mc_record import MCRecord
from .constants import defaults


@dataclass
class EventData:
    key: str
    direction: np.ndarray
    energy: np.ndarray
    time: int
    start_position: np.ndarray
    length: Optional[float] = None
    particle_id: Optional[int] = None


@dataclass
class EventCollection:
    events: List[ak.Array]
    records: List[MCRecord]
    detector: Optional[Detector] = None

    rng: Optional[np.random.RandomState] = defaults["rng"]
    seed: Optional[int] = defaults["seed"]

    def __len__(self):
        return len(self.events)

    # def generate_histogram(self, step_size=50, number_of_modules=None, event_index=None) -> Tuple[np.ndarray, np.ndarray]:
    def generate_histogram(self, step_size=50, start_time=None, end_time=None, number_of_modules=None, event_index=None) -> np.ndarray:

        events_to_account = []
        records_to_account = []
        if number_of_modules is None and self.detector is None:
            raise ValueError('Either number of modules or detector is not provided.')

        if self.detector is not None and number_of_modules is None:
            number_of_modules = len(self.detector.modules)

        short_collection = self.get_within_timeframe(start_time, end_time)

        events_to_enumerate = short_collection.events

        if event_index is not None:
            events_to_enumerate = [events_to_enumerate[event_index]]

        for event_loop_index, event in enumerate(events_to_enumerate):
            if len(event) == number_of_modules:
                events_to_account.append(event)
                records_to_account.append(short_collection.records[event_loop_index])

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

        for partition in every(partitioned_events_to_account):
            # Ensure second dimension is at number of modules
            regular_array = ak.to_regular(partition, axis=1)

            concatenated_events = ak.concatenate(regular_array, axis=1)

            for module_index, module in enumerate(concatenated_events):
                histogram[module_index] += np.histogram(ak.to_numpy(module), bins=bins, range=(start_time, end_time))[0]

        slim_records_data = []

        # TODO Split in two different functions
        for record in records_to_account:
            info = record.mc_info
            if isinstance(info, list):
                info = info[0]
            slim_records_data.append({
                'time': info.time,
                'event': record.event_type
            })

        # record_histograms = []
        #
        # for record in self.records:
        #     for source in record.sources:
        #         record_histograms.append(np.histogram([source.time], bins=bins, range=(min,max))[0])

        return histogram, slim_records_data

        # return np.array(histograms), np.array(record_histograms)

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
            info = record.mc_info

            if not isinstance(info, EventData):
                if isinstance(info, list):
                    info = info[0]
                if "key" not in info:
                    key, subkey = random.split(random.PRNGKey(self.seed))
                obj_info = EventData(
                    key=subkey,
                    direction=info["dir"],
                    energy=info["energy"],
                    time=info["time"],
                    start_position=info["pos"],
                )

                if "length" in info:
                    obj_info.length = info["length"]

                if "particle_id" in info:
                    obj_info.particle_id = info["particle_id"]

                info = obj_info

                record.mc_info = obj_info

            start_time = info.time
            end_time = times_det[index]

            for source in record.sources:
                source.time = source.time - start_time + end_time

            self.events[index] = self.events[index] - start_time + end_time
            info.time = end_time

            record.mc_info = info
        logging.info('Finish redistribute events (start %s, end %s, rate %s)', start_time, end_time, rate)

    def get_within_timeframe(self, start_time: int, end_time: int = None) -> EventCollection:
        events = []
        records = []

        logging.info('Start collecting events within %d and %d (%d Events available)', start_time, end_time, len(self))
        for index, record in enumerate(self.records):
            # TODO: Remove or properly add list behaviour
            info = record.mc_info
            if isinstance(info, list):
                info = info[0]
            record_time = info.time
            if start_time <= record_time and (end_time is None or record_time < end_time):
                events.append(self.events[index])
                records.append(record)

        logging.info('Finish collecting events within %d and %d (%d Events collected)', start_time, end_time, len(records))


        return EventCollection(records=records, events=events, detector=self.detector, rng=self.rng, seed=self.seed)

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
        events = []
        records = []

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
                # TODO: Check why errors
                # if result.rng != final_collection.rng:
                #     logging.warning('Imported multiple rngs')
                final_collection.events += result.events
                final_collection.records += result.records

            logging.debug('File %s loaded', f)

        # TODO: Make multiprocessing work again
        # global load_single_pickle
        # def load_single_pickle(filename) -> EventCollection:
        #     with open(filename, 'rb') as f:
        #         data = pickle.load(f)
        #         f.close()
        #     logging.debug('File %s loaded', f)
        #
        #     if isinstance(data, EventCollection):
        #         return data
        #
        #     return EventCollection(events=data[0], records=data[1])
        #
        # p = Pool(cpu_count(), maxtasksperchild=1000)
        #
        # combined = p.map(load_single_pickle, filenames)
        #
        # p.close()
        #
        # if len(combined) == 0:
        #     logging.warning('Imported empty collection')
        #     return EventCollection(events=[], records=[])
        #
        # final_collection = combined[0]
        #
        # for result in combined[1:]:
        #     if result.detector != final_collection.detector:
        #         logging.warning('Imported multiple detectors')
        #     if result.seed != final_collection.seed:
        #         logging.warning('Imported multiple seeds')
        #     # TODO: Check why errors
        #     # if result.rng != final_collection.rng:
        #     #     logging.warning('Imported multiple rngs')
        #     final_collection.events += result.events
        #     final_collection.records += result.records

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
