"""Modula containing the interface for generators."""
import logging
from abc import ABC, abstractmethod
from typing import TypeVar, Generic

import numpy as np

from ananke.models.detector import Detector
from ananke.models.collection import Collection
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

    def generate(
            self,
            collection: Collection,
            number_of_samples: int,
            drop_empty_records: bool = True,
            recompress: bool = True,
            append: bool = False
    ):
        """Generates a full collection.

        Args:
            collection: collection to generate for
            number_of_samples: Amount of samples to be generated
            drop_empty_records: Ensure only records with hits count
            recompress: Recompress collection after complete creation.
            append: generate full number of samples anyway
        """
        self._generate(
            collection=collection,
            number_of_samples=number_of_samples,
            drop_empty_records=drop_empty_records,
            recursion_depth=0,
            append=append
        )
        if recompress:
            collection.storage.optimize()

    def _generate(
            self,
            collection: Collection,
            number_of_samples: int,
            drop_empty_records: bool,
            recursion_depth: int,
            append: bool = False
    ) -> None:
        """Generates a full collection.

        Args:
            collection: collection to generate for
            number_of_samples: Amount of samples to be generated
            drop_empty_records: Ensure only records with hits count
            recursion_depth: Recursive stopping mechanism
            append: generate full number of samples anyway
        """
        # get valid number of current samples
        if drop_empty_records:
            collection.drop_no_hit_records()
        current_records = collection.storage.get_records()
        if current_records is None:
            current_samples = 0
        else:
            current_samples = len(current_records)
        del current_records

        # get number of samples to add depending on first call
        if recursion_depth == 0:
            if append:
                number_of_samples = number_of_samples + current_samples
            else:
                number_of_samples = number_of_samples
        samples_to_generate = number_of_samples - current_samples

        # No more samples to add, return
        if samples_to_generate <= 0:
            logging.info(
                'Finished to generate {} {}'.format(
                    samples_to_generate,
                    self.record_type
                )
            )
            return

        # Now lets get started
        logging.info(
            'Starting to generate {} {}'.format(samples_to_generate, self.record_type)
        )
        self.generate_records(
            collection=collection,
            number_of_samples=samples_to_generate
        )
        self.generate_sources(collection=collection)
        self.generate_hits(collection=collection)

        if recursion_depth == 500:
            raise RecursionError(
                'Record Generation not deterministic for {} levels'.format(
                    recursion_depth
                )
            )

        # let's do it all again (will stop if enough records)
        self._generate(
            collection=collection,
            number_of_samples=number_of_samples,
            drop_empty_records=drop_empty_records,
            recursion_depth=recursion_depth + 1,
            append=append
        )

    @abstractmethod
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
        pass

    def generate_sources(
            self,
            collection: Collection,
    ) -> None:
        """Generates and sets the sources to further evaluate.

        Args:
            collection: collection to generate for
        """
        pass

    @abstractmethod
    def generate_hits(
            self,
            collection: Collection,
    ) -> None:
        """Generates and sets the hits to further evaluate.

        Args:
            collection: collection to generate for
        """
        pass
