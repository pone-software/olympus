from dataclasses import dataclass
from typing import Optional, List, Tuple
import os

import pickle

import numpy as np
import awkward as ak

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

    def generate_histogram(self, step_size=50, number_of_modules=None) -> Tuple[np.ndarray, np.ndarray]:
        events_to_account = []
        records_to_account = []
        if number_of_modules is None and self.detector is None:
            raise ValueError('Either number of modules or detector is not provided.')

        if self.detector is not None and number_of_modules is None:
            number_of_modules = len(self.detector.modules)
        
        for index, event in enumerate(self.events):
            if len(event) == number_of_modules:
                events_to_account.append(event)
                records_to_account.append(self.records[index])

        concatenated_events = ak.sort(ak.concatenate(events_to_account, axis=1))
        max = int(np.ceil(ak.max(concatenated_events)))
        min = int(np.floor(ak.min(concatenated_events)))
        bins = int(np.ceil((max - min) / step_size))

        if bins == 0:
            bins = 10

        histograms = []

        for module in concatenated_events:
            histograms.append(np.histogram(module, bins=bins, range=(min, max))[0])

        record_histograms = []

        for record in self.records:
            for source in record.sources:
                record_histograms.append(np.histogram([source.time], bins=bins, range=(min,max))[0])

        return np.array(histograms), np.array(record_histograms)

    def redistribute(
        self,
        start_time: int,
        new_time: int = None,
        rate: float = None,
        rng: np.random.RandomState = None,
    ):
        nr_events = len(self.events)  

        if rng is None:
            rng = self.rng
        
        if rate is not None:
            new_time = start_time + np.ceil(nr_events * rate)
        if new_time is not None:
            times_det = rng.random_integers(start_time, new_time, nr_events)
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
            new_time = times_det[index]

            for source in record.sources:
                source.time = source.time - start_time + new_time

            self.events[index] = self.events[index] - start_time + new_time
            info.time = new_time

            record.mc_info = info

    @classmethod
    def from_pickle(cls, filename):
        return cls.from_pickles([filename])

    @classmethod
    def from_pickles(cls, filenames: List):
        events = []
        records = []
        for filename in filenames:
            with open(filename, 'rb') as f:
                data = pickle.load(f)

            events += data[0]
            records += data[1]

        return cls(events=events, records=records)

    @classmethod
    def from_folder(cls, folder):
        filenames = []
        for file in os.listdir(folder):
            if file.endswith(".pickle"):
                filenames.append(os.path.join(folder, file))

        return cls.from_pickles(filenames)
