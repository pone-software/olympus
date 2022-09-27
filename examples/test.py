
import os
import numpy as np

from olympus.event_generation.data import EventCollection
from olympus.event_generation.detector import make_triang, Detector, make_line
from olympus.plotting.plotting import plot_timeline


filename = os.path.join(os.path.dirname(__file__), '../../data/tracks/events_track_0.pickle')

events = EventCollection.from_pickle(filename)

rng = np.random.RandomState(31338)
oms_per_line = 20
dist_z = 50  # m
dark_noise_rate = 16 * 1e4 * 1e-9  # 1/ns

pmts_per_module = 16
pmt_cath_area_r = 75e-3 / 2  # m
module_radius = 0.21  # m

# Calculate the relative area covered by PMTs
efficiency = (
    pmts_per_module * (pmt_cath_area_r) ** 2 * np.pi / (4 * np.pi * module_radius**2)
)
det = Detector(make_line(0, 0, 20, 50, rng, dark_noise_rate, 0, efficiency=efficiency))
# det = make_triang(100, 20, dist_z, dark_noise_rate, rng, efficiency)

events.detector = det

events.redistribute(0, 10000)


plot_timeline(event_collection=events, draw_records=True)