"""Module containing code for mock photon source propagation."""
import numpy as np
import jax.numpy as jnp

from ananke.models.detector import Detector
from ananke.models.event import SourceRecordCollection, HitCollection
from olympus.event_generation.photon_propagation.interface import AbstractPhotonPropagator


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
        self.pmt_positions = jnp.array(pmt_orientations)

        pmt_areas = self.detector_df[
            'pmt_area'
        ].to_numpy(np.float32)
        self.pmt_areas = jnp.array(pmt_areas)

        pmt_efficiencies = self.detector_df[
            'pmt_efficiency'
        ].to_numpy(np.float32)
        self.pmt_efficiencies = jnp.array(pmt_efficiencies)


    def propagate(
            self, sources: SourceRecordCollection,
            seed: int = 1337
    ) -> HitCollection:
        sources_df = sources.to_pandas()

        source_locations = sources_df[[
            'location_x',
            'location_y',
            'location_z',
        ]].to_numpy(np.float32)
        self.source_positions = jnp.array(source_locations)

        source_orientations = sources_df[[
            'orientation_x',
            'orientation_y',
            'orientation_z',
        ]].to_numpy(np.float32)
        self.source_positions = jnp.array(source_orientations)

        hits = HitCollection()

        return hits
