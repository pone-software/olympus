from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class EventData:
    direction: np.ndarray
    energy: np.ndarray
    time: int
    start_position: np.ndarray
    key: Optional[str] = None
    length: Optional[float] = None
    particle_id: Optional[int] = None
