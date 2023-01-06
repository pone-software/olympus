import numpy as np
import pandas as pd

from ananke.configurations.detector import DetectorConfiguration
from ananke.models.detector import Detector
from ananke.models.geometry import Vectors3D
from ananke.models.event import EventRecords, Sources
from ananke.schemas.detector import DetectorSchema
from olympus.event_generation.generators import GeneratorCollection, GeneratorFactory
from olympus.event_generation.medium import MediumEstimationVariant, Medium
from olympus.event_generation.photon_propagation.mock_photons import MockPhotonPropagator

dark_noise_rate = 16 * 1e-5  # 1/ns
pmt_cath_area_r = 75e-3 / 2  # m
module_radius = 0.21  # m
efficiency = 0.42 # Christian S. Number

number_of_steps = 18
spherical_orientations = np.zeros((number_of_steps, 3))

spherical_orientations[:, 0] = 1
angle_range = np.linspace(0, -1* np.pi, number_of_steps)
spherical_orientations[:, 2] = angle_range
spherical_orientations_df = pd.DataFrame(spherical_orientations, columns=('norm', 'phi', 'theta'))
cartesian_orientations = Vectors3D.from_spherical(spherical_orientations_df)

module_1_df = cartesian_orientations.get_df_with_prefix('pmt_orientation_')
module_1_df['pmt_id'] = range(number_of_steps)
module_1_df['pmt_efficiency'] = efficiency
module_1_df['pmt_area'] = pmt_cath_area_r
module_1_df['pmt_noise_rate'] = dark_noise_rate
module_1_df['pmt_location_x'] = 0.0
module_1_df['pmt_location_y'] = 0.0
module_1_df['pmt_location_z'] = 0.0
module_1_df['module_id'] = 0
module_1_df['module_radius'] = module_radius
module_1_df['module_location_x'] = 0.0
module_1_df['module_location_y'] = 0.0
module_1_df['module_location_z'] = 0.0
module_1_df['string_id'] = 0
module_1_df['string_location_x'] = 0.0
module_1_df['string_location_y'] = 0.0
module_1_df['string_location_z'] = 0.0

module_1_detector = Detector(df=module_1_df)
module_1_detector.df.head()
number_of_steps = 36
spherical_orientations = np.zeros((number_of_steps, 3))

spherical_orientations[:, 0] = 1
angle_range = np.linspace(0, 2*np.pi, number_of_steps)
spherical_orientations[:, 1] = angle_range
spherical_orientations[:, 2] = np.pi / 2
spherical_orientations_df = pd.DataFrame(spherical_orientations, columns=('norm', 'phi', 'theta'))
cartesian_orientations = Vectors3D.from_spherical(spherical_orientations_df)

module_2_df = cartesian_orientations.get_df_with_prefix('pmt_orientation_')
module_2_df['pmt_id'] = range(number_of_steps)
module_2_df['pmt_efficiency'] = efficiency
module_2_df['pmt_area'] = pmt_cath_area_r
module_2_df['pmt_noise_rate'] = dark_noise_rate
module_2_df['pmt_location_x'] = 0.0
module_2_df['pmt_location_y'] = 0.0
module_2_df['pmt_location_z'] = 0.0
module_2_df['module_id'] = 1
module_2_df['module_radius'] = module_radius
module_2_df['module_location_x'] = 0.0
module_2_df['module_location_y'] = 0.0
module_2_df['module_location_z'] = 0.0
module_2_df['string_id'] = 0
module_2_df['string_location_x'] = 0.0
module_2_df['string_location_y'] = 0.0
module_2_df['string_location_z'] = 0.0

module_2_detector = Detector(df=module_2_df)
module_2_detector.df.head()

number_of_steps = 36
spherical_orientations = np.zeros((number_of_steps, 3))

spherical_orientations[:, 0] = 1
angle_range = np.linspace(0, 2*np.pi, number_of_steps)
spherical_orientations[:, 1] = angle_range
spherical_orientations[:, 2] = np.pi / 4
spherical_orientations_df = pd.DataFrame(spherical_orientations, columns=('norm', 'phi', 'theta'))
cartesian_orientations = Vectors3D.from_spherical(spherical_orientations_df)

module_3_df = cartesian_orientations.get_df_with_prefix('pmt_orientation_')
module_3_df['pmt_id'] = range(number_of_steps)
module_3_df['pmt_efficiency'] = efficiency
module_3_df['pmt_area'] = pmt_cath_area_r
module_3_df['pmt_noise_rate'] = dark_noise_rate
module_3_df['pmt_location_x'] = 0.0
module_3_df['pmt_location_y'] = 0.0
module_3_df['pmt_location_z'] = 0.0
module_3_df['module_id'] = 2
module_3_df['module_radius'] = module_radius
module_3_df['module_location_x'] = 0.0
module_3_df['module_location_y'] = 0.0
module_3_df['module_location_z'] = 0.0
module_3_df['string_id'] = 0
module_3_df['string_location_x'] = 0.0
module_3_df['string_location_y'] = 0.0
module_3_df['string_location_z'] = 0.0

module_3_detector = Detector(df=module_3_df)
module_3_detector.df.head()

distance_between_pmts = 5
maximum_distance = 100
number_of_steps = int(maximum_distance / distance_between_pmts)
cartesian_locations = np.zeros((number_of_steps, 3))
cartesian_locations[:, 0] = np.arange(0, maximum_distance, distance_between_pmts)
cartesian_locations = Vectors3D.from_numpy(cartesian_locations)

module_4_df = cartesian_locations.get_df_with_prefix('pmt_location_')
module_4_df['pmt_id'] = range(number_of_steps)
module_4_df['pmt_efficiency'] = efficiency
module_4_df['pmt_area'] = pmt_cath_area_r
module_4_df['pmt_noise_rate'] = dark_noise_rate
module_4_df['pmt_orientation_x'] = -1.0
module_4_df['pmt_orientation_y'] = 0.0
module_4_df['pmt_orientation_z'] = 0.0
module_4_df['module_id'] = 3
module_4_df['module_radius'] = module_radius
module_4_df['module_location_x'] = 0.0
module_4_df['module_location_y'] = 0.0
module_4_df['module_location_z'] = 0.0
module_4_df['string_id'] = 0
module_4_df['string_location_x'] = 0.0
module_4_df['string_location_y'] = 0.0
module_4_df['string_location_z'] = 0.0

module_4_detector = Detector(df=module_4_df)
module_4_detector.df.head()

detector = Detector.concat([
    module_1_detector,
    module_2_detector,
    module_3_detector,
    module_4_detector,
])
len(detector.df.index)

number_of_steps = 18
spherical_orientations = np.zeros((number_of_steps, 3))

spherical_orientations[:, 0] = 1
angle_range = np.linspace(0, np.pi, number_of_steps)
spherical_orientations[:, 2] = angle_range
spherical_orientations_df = pd.DataFrame(spherical_orientations, columns=('norm', 'phi', 'theta'))
cartesian_orientations = Vectors3D.from_spherical(spherical_orientations_df)

events_df = cartesian_orientations.get_df_with_prefix('orientation_')
events_df['record_id'] = range(number_of_steps)
events_df['time'] = 0.0
events_df['location_x'] = -5.0
events_df['location_y'] = 0.0
events_df['location_z'] = 0.0

sources_df = events_df.copy()

events_df['energy'] = 10000
events_df['length'] = 3000
events_df['particle_id'] = 11
events_df['type'] = 'cascade'

sources_df['type'] = 'cherenkov'
sources_df['number_of_photons'] = 1000000

event_records = EventRecords(df=events_df)
sources = Sources(df=sources_df)

event_records.df.head()

medium = Medium(MediumEstimationVariant.PONE_OPTIMISTIC)

photon_propagator = MockPhotonPropagator(
    detector=detector,
    medium=medium,
    angle_resolution=18000,
)

hits = photon_propagator.propagate(event_records, sources)
hits.df.head()