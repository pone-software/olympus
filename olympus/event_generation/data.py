
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class EventData:
    key: str
    direction: np.ndarray
    energy: np.ndarray
    time: int
    start_position: np.ndarray
    length: Optional[float]
    particle_id: Optional[int]
