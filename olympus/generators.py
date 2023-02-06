"""Modula containing the interface for generators."""
import os
import uuid
from abc import ABC, abstractmethod
from typing import List, TypeVar, Generic, Union

import numpy as np

from ananke.models.detector import Detector
from ananke.models.collection import Collection
from ananke.models.event import Records
from ananke.schemas.event import RecordType
from olympus.configuration.generators import GeneratorConfiguration

_GeneratorConfiguration = TypeVar(
    '_GeneratorConfiguration',
    bound=GeneratorConfiguration
)

_GeneratorRecordType = TypeVar(
    '_GeneratorRecordType',
    bound=RecordType
)


class AbstractGenerator(ABC, Generic[_GeneratorConfiguration, _GeneratorRecordType]):
    def __init__(
            self,
            record_type: _GeneratorRecordType,
            detector: Detector,
            configuration: _GeneratorConfiguration,
    ):
        """Abstract parent to all detector based generators.

        TODO: Decide whether to include record type here or in generate method.

        Args:
            record_type: Type of the record
            detector: Detector to generate records for.
            configuration: Configuration of the records to generate.
        """
        super(AbstractGenerator, self).__init__()

        self.record_type = record_type
        self.configuration = configuration
        self.detector = detector
        self.rng = np.random.default_rng(configuration.seed)

    @abstractmethod
    def generate(
            self,
            collection: Collection,
            number_of_samples: int,
    ) -> None:
        """Generates a full collection.

        Args:
            collection: collection to generate for
            number_of_samples: Amount of samples to be generated
        """
        pass

    @abstractmethod
    def generate_records(
            self,
            number_of_samples: int,
    ) -> Records:
        """Generates the raw records to further evaluate.

        Args:
            number_of_samples: Amount of records to be generated.

        Returns:
            Records to be processed.
        """
        pass

    def _get_record_ids(self, number_of_samples: int) -> List[int]:
        """Generates record uuid1 integer ids for generated objects.

        Args:
            number_of_samples: Amount of record ids

        Returns:
            List of integer ids.
        """
        if self.configuration.fix_uuids:
            clock = self.configuration.seed
        else:
            clock = None
        return [uuid.uuid1(clock_seq=clock).int >> 64 for x in range(number_of_samples)]
