"""Module containing code for mock photon source propagation."""
from typing import Union, Tuple, Optional

import numpy as np
import jax.numpy as jnp

from jax import jit

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

    return jnp.arccos(jnp.clip(jnp.matmul(v1_u, jnp.transpose(v2_u)), -1.0, 1.0))


class MockPhotonPropagator(AbstractPhotonPropagator):
    """Class enabling basic photon propagation to detector modules."""

    def __init__(
            self,
            detector: Detector,
            **kwargs
    ) -> None:
        super().__init__(detector=detector, **kwargs)

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

    def propagate(
            self, sources: SourceRecordCollection,
            seed: int = 1337
    ) -> HitCollection:
        sources_df = sources.to_pandas()
        number_of_sources = len(sources_df.index)

        source_locations = sources_df[[
            'location_x',
            'location_y',
            'location_z',
        ]].to_numpy(np.float32)
        source_positions = jnp.array(source_locations)

        source_orientations = sources_df[[
            'orientation_x',
            'orientation_y',
            'orientation_z',
        ]].to_numpy(np.float32)
        source_orientations = jnp.array(source_orientations)

        hits = HitCollection()

        # 1. Calculate angle between PMT and Source direction
        expanded_pmt_positions = jnp.tile(
            jnp.expand_dims(self.pmt_positions, axis=1),
            (1, len(sources_df.index), 1)
        )
        pmt_to_source = source_positions - expanded_pmt_positions
        pmt_to_source_distances = np.linalg.norm(pmt_to_source, axis=2)
        pmt_to_source_angles = self.__calculate_pmt_to_source_angles(pmt_to_source)
        source_orientation_angles = self.__calculate_source_orientation_angles(
            - pmt_to_source, # we want to know angle between source and the pmt.
            source_orientations
            )

        # 2. Mask sources in direction of PMT (180°)
        # mask_condition = pmt_to_source_angles < np.pi / 2
        # pmt_to_source_angles = np.where(mask_condition, pmt_to_source_angles, 0)
        # source_orientation_angles = np.where(mask_condition, source_orientation_angles, 0)

        # 3. Calculate Yield based on distance


        # 4. Calculate Yield based on angle on sphere

        # 5. Distribute Yield using Gamma distribution

        return hits
