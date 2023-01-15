"""Module containing code for mock photon source propagation."""
from multiprocessing import Pool
from typing import Union, Tuple, List

import numpy as np
import numpy.typing as npt
import jax.numpy as jnp
import pandas as pd
import logging

from ananke.models.detector import Detector
from ananke.models.event import Sources, Hits, EventRecords
from olympus.event_generation.lightyield import fennel_angle_distribution_function
from olympus.event_generation.medium import Medium
from olympus.event_generation.photon_propagation.interface import \
    AbstractPhotonPropagator


def func(record_id, pmt_photons, pmt_times, pmt_indices, rng) -> Hits:
    logging.log(logging.INFO,
                "Processed PMT {} for Record {}".format(
                    pmt_indices, record_id
                ))

    hit_times = []

    for index, photons in enumerate(pmt_photons):
        if photons == 0:
            continue
        # TODO: Check passing of RNG generator
        # TODO: Flatten everything
        hit_times.append(rng.gamma(
            pmt_times[index],
            1.0,
            photons
        ))

    string_id = pmt_indices[0]
    module_id = pmt_indices[1]
    pmt_id = pmt_indices[2]
    records_hits_df = pd.DataFrame(
        {
            'time': hit_times
        }
    )

    records_hits_df['string_id'] = string_id
    records_hits_df['module_id'] = module_id
    records_hits_df['pmt_id'] = pmt_id
    records_hits_df['record_id'] = record_id

    return Hits(df=records_hits_df)


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
    norm = jnp.expand_dims(jnp.linalg.norm(vector, axis=axis), axis=axis)
    # TODO: Investigate Bug with jnp division precision
    return jnp.array(
        np.divide(
            vector,
            norm,
            out=np.zeros_like(vector),
            where=norm != 0
        )
    )


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
            medium: Medium,
            angle_resolution: int = 18000,
            **kwargs
    ) -> None:
        super().__init__(detector=detector, medium=medium, **kwargs)
        # TODO: Migrate Angle and Wavelength away
        self.angle_resolution = angle_resolution

        pmt_locations = self.detector.pmt_locations.to_numpy(np.float32)
        self.pmt_positions = jnp.array(pmt_locations)

        self.pmt_indices = self.detector.indices

        pmt_orientations = self.detector.pmt_orientations.to_numpy(np.float32)
        self.pmt_orientations = jnp.array(pmt_orientations)

        pmt_areas = self.detector.pmt_areas.to_numpy(np.float32)
        self.pmt_areas = jnp.array(pmt_areas)
        self.pmt_radius = jnp.sqrt(jnp.array(pmt_areas) / jnp.pi)

        pmt_efficiencies = self.detector.pmt_efficiencies.to_numpy(np.float32)
        self.pmt_efficiencies = jnp.array(pmt_efficiencies)

        module_radius = self.detector.module_radius.to_numpy(np.float32)
        self.module_radius = jnp.array(module_radius)

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
        per_source_and_pmt_unit = unit_vector(pmt_to_source, axis=2)
        per_pmt_unit = unit_vector(self.pmt_orientations, axis=1)
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
        per_source_and_pmt_unit = unit_vector(pmt_to_source, axis=2)
        source_orientation_unit = unit_vector(source_orientations, axis=1)
        dot_products = jnp.einsum(
            'ijk,jk->ij',
            per_source_and_pmt_unit,
            source_orientation_unit
        )

        return cls.__arccos(dot_products=dot_products)

    def __get_ellipsis_mask(
            self,
            number_of_modules: int,
            number_of_sources: int,
            pmt_to_source: jnp.ndarray,
            pmt_to_source_angles: jnp.ndarray,
            pmt_opening_angles_length: int,
            max_opening_angles_length: int,
            source_orientations: jnp.ndarray
    ) -> jnp.ndarray:
        """Calculates the mask of the ellipsis for each pmt and source.

        Args:
            number_of_modules: Number of modules
            number_of_sources: Number of sources
            pmt_to_source: Vectors from PMTs to Sources
            pmt_to_source_angles: Angles between PMTs and Sources
            pmt_opening_angles_length: Number of indices to account for of PMT
            max_opening_angles_length: Ceiled number of PMT opening angles
            source_orientations: Vectors representing source orientations

        Returns:

        """
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

        center_index = int(jnp.floor(max_opening_angles_length / 2))
        target_indices = jnp.arange(max_opening_angles_length) \
                             .astype('int16') - center_index
        target_dimensions = (max_opening_angles_length, max_opening_angles_length)
        x = jnp.broadcast_to(
            jnp.square(target_indices),
            target_dimensions
        )
        broadcast_dimensions = (
                                   number_of_modules, number_of_sources
                               ) + target_dimensions
        target_full = jnp.einsum('i,j->ij', target_indices, target_indices)
        broad_x = jnp.broadcast_to(
            x,
            broadcast_dimensions
        )
        broad_y = jnp.broadcast_to(
            x.T,
            broadcast_dimensions
        )
        broad_target_full = jnp.broadcast_to(
            target_full,
            broadcast_dimensions
        )

        # formula: https://www.maa.org/external_archive/joma/Volume8/Kalman/General.html
        # TODO: Check Sin Cos of Ellipsis
        ellipsis_mask = jnp.multiply(
            jnp.add(
                jnp.square(jnp.divide(jnp.sin(ellipsis_angles), ellipsis_a_axis)),
                jnp.square(jnp.divide(jnp.cos(ellipsis_angles), ellipsis_b_axis))
            )[..., jnp.newaxis, jnp.newaxis],
            broad_x
        ) + 2 * jnp.multiply(
            jnp.multiply(
                jnp.multiply(jnp.cos(ellipsis_angles), jnp.sin(ellipsis_angles)),
                jnp.subtract(
                    jnp.reciprocal(jnp.square(ellipsis_a_axis)),
                    jnp.reciprocal(jnp.square(ellipsis_b_axis))
                )
            )[..., jnp.newaxis, jnp.newaxis],
            broad_target_full
        ) + jnp.multiply(
            jnp.add(
                jnp.square(jnp.divide(jnp.cos(ellipsis_angles), ellipsis_a_axis)),
                jnp.square(jnp.divide(jnp.sin(ellipsis_angles), ellipsis_b_axis))
            )[..., jnp.newaxis, jnp.newaxis],
            broad_y
        )
        return (ellipsis_mask <= 1).astype('int16')

    def __calculate_angular_yield(
            self,
            number_of_modules: int,
            number_of_sources: int,
            source_orientations: jnp.ndarray,
            source_angle_distribution: jnp.ndarray,
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
                self.pmt_radius,
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
            jnp.tile(source_angle_distribution, (number_of_modules, 1)),
            angle_indices
        )
        ellipsis_mask = self.__get_ellipsis_mask(
            number_of_modules=number_of_modules,
            source_orientations=source_orientations,
            pmt_to_source=pmt_to_source,
            number_of_sources=number_of_sources,
            pmt_to_source_angles=pmt_to_source_angles,
            pmt_opening_angles_length=pmt_opening_angles_length,
            max_opening_angles_length=max_opening_angle_length
        )
        yield_per_source_and_pmt = jnp.einsum(
            '...ij,...i->...',
            ellipsis_mask,
            angular_distribution_per_pmt
        )
        # TODO: Check Integral

        # we only have yield at pmts facing the source
        mask_condition = (pmt_to_source_angles < jnp.pi / 2).astype('int16')

        yield_per_source_and_pmt = jnp.einsum(
            'ij, ij->ij',
            mask_condition,
            yield_per_source_and_pmt
        )

        return yield_per_source_and_pmt

    def __calculate_distance_yield(
            self,
            pmt_to_source_distances: jnp.ndarray
    ) -> jnp.ndarray:
        absorption_length = self.medium.get_absolute_length(self.default_wavelengths)

        return jnp.exp(-1.0 * jnp.divide(pmt_to_source_distances, absorption_length))

    def _get_angle_distribution(
            self, energy: float, particle_id: int
    ) -> jnp.ndarray:

        angles = jnp.linspace(0, 180, self.angle_resolution)
        n_photon = self.medium.get_refractive_index(self.default_wavelengths)
        angle_distribution = fennel_angle_distribution_function(
            energy=energy,
            particle_id=particle_id
        )
        source_angle_distribution = angle_distribution(
            angles,
            n_photon
        )

        source_angle_distribution = jnp.array(source_angle_distribution)

        # norm all angle distributions according to a cylinder
        normation_factor = jnp.sum(source_angle_distribution)
        normation_factor = normation_factor * self.angle_resolution

        source_angle_distribution = jnp.divide(source_angle_distribution, normation_factor)

        return source_angle_distribution

    def propagate(
            self,
            records: EventRecords,
            sources: Sources,
            seed: int = 1337,
            use_multiprocessing: bool = True
    ) -> Union[Hits, Tuple[Hits, jnp.array]]:
        if len(sources) == 0:
            return Hits()
        hits_list = []
        for record_index, record in records.df.iterrows():
            record_sources = sources.get_by_record(record.record_id)
            number_of_sources = len(record_sources)
            number_of_pmts = len(self.detector)

            source_locations = record_sources.locations.to_numpy(np.float32)
            source_locations = jnp.array(source_locations)

            source_orientations = record_sources.orientations.to_numpy(np.float32)
            source_orientations = jnp.array(source_orientations)

            source_photons = record_sources.number_of_photons.to_numpy(np.float32)
            source_photons = jnp.array(source_photons)

            source_times = record_sources.times.to_numpy(np.float32)
            source_times = jnp.array(source_times)

            # 1. Calculate angle between PMT and Source direction
            expanded_pmt_positions = jnp.tile(
                jnp.expand_dims(self.pmt_positions, axis=1),
                (1, number_of_sources, 1)
            )
            pmt_to_source = source_locations - expanded_pmt_positions
            pmt_to_source_distances = jnp.linalg.norm(pmt_to_source, axis=2)

            source_angle_distribution = self._get_angle_distribution(
                energy=record.energy,
                particle_id=record.particle_id
            )

            # 2. Mask sources in direction of PMT (180°)
            angular_yield = self.__calculate_angular_yield(
                number_of_modules=number_of_pmts,
                number_of_sources=number_of_sources,
                source_orientations=source_orientations,
                pmt_to_source=pmt_to_source,
                source_angle_distribution=source_angle_distribution,
                pmt_to_source_distances=pmt_to_source_distances
            )
            # 3. Calculate Yield based on distance
            distance_yield = self.__calculate_distance_yield(
                pmt_to_source_distances=pmt_to_source_distances
            )

            # 4. Calculate Yield based on PMT Efficiency
            efficiency_yield = self.pmt_efficiencies

            # 4. Calculate Number of Photons
            total_yield = jnp.multiply(angular_yield, distance_yield)
            total_yield = jnp.multiply(total_yield, efficiency_yield)
            photons_per_pmt_per_source = jnp.multiply(total_yield, source_photons.T)
            photons_per_pmt_per_source = jnp.rint(photons_per_pmt_per_source).astype(
                'int16'
            )

            # 5. Calculate Arrival Time
            travel_time = pmt_to_source_distances / self.medium.get_c_medium_photons(
                self.default_wavelengths
            )
            times_per_pmt_per_source = jnp.add(travel_time, source_times.T)

            # 5. Distribute Yield using Gamma distribution

            multiprocessing_args = []

            for pmt_index, pmt_photons in enumerate(photons_per_pmt_per_source):
                if jnp.max(pmt_photons) > 0:
                    multiprocessing_args.append((
                        record.record_id,
                        pmt_photons,
                        times_per_pmt_per_source[pmt_index],
                        self.pmt_indices.iloc[pmt_index],
                        self.rng
                    ))

            if not use_multiprocessing:
                for args in multiprocessing_args:
                    hits_list += func(*args)
            else:
                with Pool() as pool:
                    hits_list += pool.starmap(func, multiprocessing_args)

        all_hits = Hits.concat(hits_list)

        return all_hits
