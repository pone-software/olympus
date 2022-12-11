"""Module containing code for mock photon source propagation."""
from typing import Union, Tuple

import numpy as np
import jax.numpy as jnp

from ananke.models.detector import Detector
from ananke.models.event import SourceRecordCollection, HitCollection
from olympus.event_generation.photon_propagation.interface import \
    AbstractPhotonPropagator


def unit_vector(
        vector: Union[jnp.ndarray, np.typing.NDArray, Tuple[float, float, float]],
        axis: Union[int, Tuple[int, int]] = 0
) -> jnp.ndarray:
    """ Returns the unit vector of the vector.

    Args:
        vector: vector to calculate unit vector of.
        axis: axis of the operation on input.

    Returns:
        Unit vector in direction of input vector.

    """
    norm = jnp.linalg.norm(vector, axis=axis)
    return jnp.divide(vector, jnp.expand_dims(norm, axis=axis))


def angle_between(
        v1: Union[jnp.ndarray, np.typing.NDArray],
        v2: Union[jnp.ndarray, np.typing.NDArray],
        axis: Tuple[int, int]
):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793

    Args:
        v1: First vector
        v2: Second vector
        axis: axis by which to calculate

    Returns:
        Angle between vectors in Radians
    """
    v1_u = unit_vector(v1, axis=axis[0])
    v2_u = unit_vector(v2, axis=axis[1])

    return jnp.arccos(jnp.clip(jnp.einsum('...i, ...i', v1_u, v2_u), -1.0, 1.0))


class MockPhotonPropagator(AbstractPhotonPropagator):
    """Class enabling basic photon propagation to detector modules."""

    def __init__(
            self,
            detector: Detector,
            angle_resolution: int = 18000,
            **kwargs
    ) -> None:
        super().__init__(detector=detector, **kwargs)

        self.angle_resolution = angle_resolution

        pmt_locations = self.detector_df[[
            'pmt_x',
            'pmt_y',
            'pmt_z',
        ]].to_numpy(np.float32)
        self.pmt_positions = jnp.array(pmt_locations)

        pmt_orientations = self.detector_df[[
            'pmt_orientation_x',
            'pmt_orientation_y',
            'pmt_orientation_z',
        ]].to_numpy(np.float32)
        self.pmt_orientations = jnp.array(pmt_orientations)

        pmt_areas = self.detector_df[
            'pmt_area'
        ].to_numpy(np.float32)
        self.pmt_areas = jnp.array(pmt_areas)

        pmt_efficiencies = self.detector_df[
            'pmt_efficiency'
        ].to_numpy(np.float32)
        self.pmt_efficiencies = jnp.array(pmt_efficiencies)

        pmt_areas = self.detector_df[
            'pmt_efficiency'
        ].to_numpy(np.float32)
        self.pmt_radius = jnp.sqrt(jnp.array(pmt_areas) / jnp.pi)

        module_radius = self.detector_df[
            'module_radius'
        ].to_numpy(np.float32)
        self.module_radius = jnp.array(module_radius)

    @staticmethod
    def __get_unit(vector: jnp.ndarray, axis: int):
        norm = jnp.linalg.norm(vector, axis=axis)
        return jnp.divide(vector, jnp.expand_dims(norm, axis=axis))

    def __calculate_orthogonals(self, source_orientations: jnp.ndarray) -> jnp.ndarray:
        orthogonal = jnp.cross(
            jnp.expand_dims(self.pmt_orientations, axis=1),
            source_orientations
        )
        return jnp.cross(source_orientations, orthogonal)

    @staticmethod
    def __arccos(dot_products: jnp.ndarray) -> jnp.ndarray:
        return jnp.arccos(jnp.clip(dot_products, -1.0, 1.0))

    def __calculate_pmt_to_source_angles(
            self,
            pmt_to_source: jnp.ndarray
    ) -> jnp.ndarray:
        per_source_and_pmt_unit = self.__get_unit(pmt_to_source, axis=2)
        per_pmt_unit = self.__get_unit(self.pmt_orientations, axis=1)
        dot_products = jnp.einsum(
            'ijk,ik->ij',
            per_source_and_pmt_unit,
            per_pmt_unit
        )

        return self.__arccos(dot_products=dot_products)

    @classmethod
    def __calculate_source_orientation_angles(
            cls,
            pmt_to_source: jnp.ndarray,
            source_orientations: jnp.ndarray
    ) -> jnp.ndarray:
        per_source_and_pmt_unit = cls.__get_unit(pmt_to_source, axis=2)
        source_orientation_unit = cls.__get_unit(source_orientations, axis=1)
        dot_products = jnp.einsum(
            'ijk,jk->ij',
            per_source_and_pmt_unit,
            source_orientation_unit
        )

        return cls.__arccos(dot_products=dot_products)

    def __calculate_angular_yield(
            self,
            number_of_modules: int,
            number_of_sources: int,
            source_orientations: jnp.ndarray,
            source_angle_distributions: jnp.ndarray,
            pmt_to_source: jnp.ndarray,
            pmt_to_source_distances: jnp.ndarray,
    ) -> jnp.ndarray:
        pmt_to_source_angles = self.__calculate_pmt_to_source_angles(pmt_to_source)
        source_orientation_angles = self.__calculate_source_orientation_angles(
            - pmt_to_source,  # we want to know angle between source and the pmt.
            source_orientations
        )
        pmt_opening_angles = 2 * jnp.arcsin(
            jnp.divide(
                jnp.expand_dims(self.pmt_radius, axis=1),
                pmt_to_source_distances
            )
        )
        pmt_opening_angles_length = pmt_opening_angles / jnp.pi * self.angle_resolution
        source_orientation_angles_index = jnp.rint(
            source_orientation_angles / jnp.pi * self.angle_resolution
        ).astype('int16')
        # this is the maximum size of the arrays to consider
        max_opening_angle_length = int(jnp.ceil(jnp.max(pmt_opening_angles_length)))
        angle_indices = jnp.indices((max_opening_angle_length,)) - \
                        jnp.rint(max_opening_angle_length / 2)
        angle_indices = jnp.add(
            jnp.tile(angle_indices, (number_of_modules, number_of_sources, 1)),
            jnp.expand_dims(source_orientation_angles_index, axis=2)
        ).astype('int16')
        # Get Angular distribution values for each pmt
        angular_distribution_per_pmt = jnp.take(
            jnp.tile(source_angle_distributions, (number_of_modules, 1, 1)),
            angle_indices
        )
        # repeat n times to get squared "target" area
        angular_distribution_per_pmt = jnp.repeat(
            jnp.expand_dims(angular_distribution_per_pmt, axis=3),
            max_opening_angle_length,
            axis=3
        )
        # calculate rotation of ellipsis
        # first find pmt normal projection on target area
        unit_pmt_to_source = unit_vector(pmt_to_source, axis=2)
        pmt_normal_to_source_normal_distance = jnp.einsum(
            'ijk,ik->ij',
            unit_pmt_to_source,
            self.pmt_orientations
        )
        projected_pmt_normal = jnp.subtract(
            jnp.expand_dims(self.pmt_orientations, axis=1),
            jnp.multiply(
                jnp.expand_dims(pmt_normal_to_source_normal_distance, axis=2),
                unit_pmt_to_source
            )
        )
        # second get second vector of target area basis orthogonal to source direction
        second_target_area_basis = jnp.cross(source_orientations, pmt_to_source)
        # third calculate ellipsis properties
        ellipsis_a_axis = pmt_opening_angles_length / 2
        ellipsis_b_axis = jnp.multiply(
            jnp.cos(pmt_to_source_angles),
            ellipsis_a_axis
        )
        ellipsis_angles = angle_between(
            projected_pmt_normal,
            second_target_area_basis,
            axis=(2, 2)
        )

        center_index = int(jnp.floor(max_opening_angle_length / 2))
        target_indices = jnp.arange(max_opening_angle_length) \
                             .astype('int16') - center_index
        target_dimensions = (max_opening_angle_length, max_opening_angle_length)
        x = jnp.broadcast_to(
            jnp.square(target_indices),
            target_dimensions
        )
        y = x.T
        broadcast_dimensions = (
                                   number_of_modules, number_of_sources
                               ) + target_dimensions
        target_full = jnp.einsum('i,j->ij', target_indices, target_indices)
        broad_x = jnp.broadcast_to(
            x,
            broadcast_dimensions
        )
        broad_y = jnp.broadcast_to(
            y,
            broadcast_dimensions
        )
        broad_target_full = jnp.broadcast_to(
            target_full,
            broadcast_dimensions
        )

        # formula: https://www.maa.org/external_archive/joma/Volume8/Kalman/General.html
        # TODO: Check Sin Cos of Ellipsis
        x_part = jnp.multiply(
            jnp.add(
                jnp.square(jnp.divide(jnp.cos(ellipsis_angles), ellipsis_a_axis)),
                jnp.square(jnp.divide(jnp.sin(ellipsis_angles), ellipsis_b_axis))
            )[..., jnp.newaxis, jnp.newaxis],
            broad_x
        )
        xy_part = 2 * jnp.multiply(
            jnp.multiply(
                jnp.multiply(jnp.cos(ellipsis_angles), jnp.sin(ellipsis_angles)),
                jnp.subtract(
                    jnp.reciprocal(jnp.square(ellipsis_a_axis)),
                    jnp.reciprocal(jnp.square(ellipsis_b_axis))
                )
            )[..., jnp.newaxis, jnp.newaxis],
            broad_target_full
        )
        y_part = jnp.multiply(
            jnp.add(
                jnp.square(jnp.divide(jnp.sin(ellipsis_angles), ellipsis_a_axis)),
                jnp.square(jnp.divide(jnp.cos(ellipsis_angles), ellipsis_b_axis))
            )[..., jnp.newaxis, jnp.newaxis],
            broad_y
        )
        ellipsis_mask = x_part + xy_part + y_part <= 1
        angular_distribution_per_pmt = jnp.where(
            ellipsis_mask,
            angular_distribution_per_pmt,
            0
        )
        yield_per_source_and_pmt = jnp.einsum(
            '...ij->...',
            angular_distribution_per_pmt
        ) * jnp.square(180 / self.angle_resolution)

        # we only have yield at pmts facing the source
        # TODO: Calculate Efficiency based on Angle and PMT
        mask_condition = pmt_to_source_angles < jnp.pi / 2

        yield_per_source_and_pmt = jnp.where(
            mask_condition,
            yield_per_source_and_pmt,
            0
        )

        return yield_per_source_and_pmt

    def propagate(
            self, sources: SourceRecordCollection,
            seed: int = 1337
    ) -> HitCollection:
        sources_df = sources.to_pandas()
        number_of_sources = len(sources_df.index)
        number_of_modules = len(self.detector_df.index)

        source_locations = sources_df[[
            'location_x',
            'location_y',
            'location_z',
        ]].to_numpy(np.float32)
        source_locations = jnp.array(source_locations)

        source_orientations = sources_df[[
            'orientation_x',
            'orientation_y',
            'orientation_z',
        ]].to_numpy(np.float32)
        source_orientations = jnp.array(source_orientations)

        source_photons = sources_df[[
            'number_of_photons'
        ]].to_numpy(np.float32)
        source_photons = jnp.array(source_photons)

        source_times = sources_df[[
            'time'
        ]].to_numpy(np.float32)
        source_times = jnp.array(source_times)

        source_angle_distributions = sources.angle_distributions

        hits = HitCollection()

        # 1. Calculate angle between PMT and Source direction
        expanded_pmt_positions = jnp.tile(
            jnp.expand_dims(self.pmt_positions, axis=1),
            (1, len(sources_df.index), 1)
        )
        pmt_to_source = source_locations - expanded_pmt_positions
        pmt_to_source_distances = jnp.linalg.norm(pmt_to_source, axis=2)

        # 2. Mask sources in direction of PMT (180°)
        angular_yield = self.__calculate_angular_yield(
            number_of_modules=number_of_modules,
            number_of_sources=number_of_sources,
            source_orientations=source_orientations,
            source_angle_distributions=source_angle_distributions,
            pmt_to_source=pmt_to_source,
            pmt_to_source_distances=pmt_to_source_distances
        )
        # 3. Calculate Yield based on distance

        # 4. Calculate Number of Photons
        photons_per_pmt_per_source = jnp.multiply(angular_yield, source_photons.T)

        # 5. Calculate Arrival Time
        travel_time = pmt_to_source_distances / self.c_medium
        times_per_pmt_per_source = jnp.add(travel_time, source_times.T)

        # 5. Distribute Yield using Gamma distribution

        return hits
