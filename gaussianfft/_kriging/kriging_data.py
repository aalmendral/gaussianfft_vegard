from dataclasses import dataclass

import numpy as np

import gaussianfft


@dataclass
class KrigingData:
    """Grid, variogram, and observations shared by the kriging methods.

    Observation arrays are normalized to floating point and validated on
    construction. Do not modify the data after constructing a kriging instance,
    which caches covariance factors and grid coordinates derived from it.
    """

    variogram: gaussianfft.Variogram
    nx: int
    dx: float
    ny: int
    dy: float
    nz: int
    dz: float
    obs_locations: np.ndarray
    obs_values: np.ndarray
    obs_uncertainties: np.ndarray

    def __post_init__(self):
        self.obs_locations = np.asarray(self.obs_locations, dtype=float)
        self.obs_values = np.asarray(self.obs_values, dtype=float)
        self.obs_uncertainties = np.asarray(self.obs_uncertainties, dtype=float)
        self._validate_inputs()

    @property
    def ndims(self) -> int:
        return 3 if self.nz > 1 else (2 if self.ny > 1 else 1)

    @property
    def grid_shape(self) -> tuple[int, ...]:
        return (self.nx, self.ny, self.nz)[:self.ndims]

    def _validate_inputs(self):
        if self.obs_locations.ndim != 2:
            raise ValueError(
                f"obs_locations must be a 2D array with shape (N, {self.ndims}), got {self.obs_locations.shape}"
            )
        if self.obs_locations.shape[1] != self.ndims:
            raise ValueError(
                f"obs_locations must have {self.ndims} columns, got {self.obs_locations.shape[1]}"
            )
        n = self.obs_locations.shape[0]
        if self.obs_values.shape != (n,):
            raise ValueError(f"obs_values must have shape ({n},), got {self.obs_values.shape}")
        if self.obs_uncertainties.shape != (n,):
            raise ValueError(f"obs_uncertainties must have shape ({n},), got {self.obs_uncertainties.shape}")
        if np.any(self.obs_uncertainties < 0):
            raise ValueError("obs_uncertainties must be non-negative")
        sizes = np.array(self.grid_shape)
        steps = np.array((self.dx, self.dy, self.dz)[:self.ndims])
        grid_max = (sizes - 1) * steps
        if np.any(self.obs_locations < 0) or np.any(self.obs_locations > grid_max):
            raise ValueError("obs_locations has values outside grid bounds")
