import uuid
from abc import ABC, abstractmethod
from typing import Optional, List

import numpy as np

from ananke.models.event import Collection, Records
from olympus.constants import defaults


class AbstractGenerator(ABC):
    def __init__(
            self,
            seed: Optional[int] = defaults["seed"],
            *args,
            **kwargs
    ):
        super().__init__()
        self.rng = np.random.default_rng(seed)

    @abstractmethod
    def generate(
            self,
            number_of_samples: int,
    ) -> Collection:
        """Generates a full collection.

        Args:
            number_of_samples: Amount of samples to be generated

        Returns:
            Collection containing all generated information.
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

    @staticmethod
    def _get_record_ids(number_of_samples: int) -> List[int]:
        """Generates record uuid1 integer ids for generated objects.

        Args:
            number_of_samples: Amount of record ids

        Returns:
            List of integer ids.
        """
        return [uuid.uuid1().int >> 64 for x in range(number_of_samples)]

