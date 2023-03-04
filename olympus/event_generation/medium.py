from enum import Enum

import numpy.typing as npt

from olympus.constants import Constants
from hyperion.medium import medium_collections


class MediumEstimationVariant(Enum):
    PONE_OPTIMISTIC = 'pone_optimistic'
    PONE_PESSIMISTIC = 'pone_pessimistic'


class Medium:
    def __init__(self, variant: MediumEstimationVariant) -> None:
        self._funcs = medium_collections[variant.value]

    def get_refractive_index(self, wavelengths: npt.ArrayLike) -> npt.ArrayLike:
        """Gets the refractive index/indices for given wavelength(s).

        Args:
            wavelengths: wavelength(s) for which to evaluate

        Returns:
            refractive index/indices as scalar or array depending on input.
        """
        return self._funcs[0](wavelengths)

    def get_attenuation_length(self, wavelengths: npt.ArrayLike) -> npt.ArrayLike:
        """Gets the attenuation length(s) for given wavelength(s).

        Args:
            wavelengths: wavelength(s) for which to evaluate

        Returns:
            attenuation length(s) as scalar or array depending on input.
        """
        return self._funcs[1](wavelengths)

    def get_scattering_length(self, wavelengths: npt.ArrayLike) -> npt.ArrayLike:
        """Gets the scattering length(s) for given wavelength(s).

        Args:
            wavelengths: wavelength(s) for which to evaluate

        Returns:
            scattering length(s) as scalar or array depending on input.
        """
        return self._funcs[2](wavelengths)

    def get_absolute_length(self, wavelengths: npt.ArrayLike) -> npt.ArrayLike:
        """Gets the length(s) for given wavelength(s) (scattering and attenuation).

        Args:
            wavelengths: wavelength(s) for which to evaluate

        Returns:
            length(s) as scalar or array depending on input.
        """
        return self._funcs[3](wavelengths)

    def get_c_medium_photons(self, wavelengths: npt.ArrayLike) -> npt.ArrayLike:
        """Gets the light speed for given wavelength(s).

        Args:
            wavelengths: wavelength(s) for which to evaluate

        Returns:
            light speed as scalar or array depending on input.
        """
        return Constants.c_vac / self.get_refractive_index(wavelengths)
