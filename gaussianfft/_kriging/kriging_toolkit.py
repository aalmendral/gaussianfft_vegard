from __future__ import annotations

import abc
from dataclasses import dataclass

import numpy as np
try:
    import scipy.linalg
    import scipy.ndimage
except ImportError:
    raise ImportError("Scipy is not installed, and is required to use these utilities")

import gaussianfft

class FactorizationError(ValueError):
    def __init__(self):
        super().__init__(
            "Failed to factorize observation covariance matrix. "
            "The matrix is likely not positive definite. This can happen for "
            "nearly duplicate observation locations with too small observation "
            "uncertainties. Consider increasing obs_uncertainties or removing "
            "duplicate/near-duplicate observations."
        )

# Each cross-covariance chunk holds (chunk_size × n_obs) float64 values.
# Chunking keeps peak memory bounded while BLAS still operates on full batches.
_MEMORY_BUDGET_BYTES = 10 * 1024 * 1024 * 1024  # 10 GB


# ---------------------------------------------------------------------------
# Module-level utilities
# ---------------------------------------------------------------------------

def _grid_coordinates(grid: GridSpec, ndims):
    if len(grid.shape) != ndims:
        raise ValueError(f"obs_locations must have {len(grid.shape)} columns for this grid")
    axes = [
        origin + np.arange(size) * spacing
        for size, spacing, origin in zip(grid.shape, grid.spacing, grid.origin)
    ]
    return np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, ndims)


def _build_obs_cov_matrix(variogram, obs_locations, obs_uncertainties):
    n = len(obs_locations)
    cov = np.empty((n, n))
    for i in range(n):
        correlations = variogram.corr_array(obs_locations[i] - obs_locations[i:])
        cov[i, i:] = correlations
        cov[i:, i] = correlations
    cov[np.diag_indices(n)] += obs_uncertainties ** 2 + 1e-10
    return cov


def _cholesky_factor(cov):
    """Factorize an observation covariance matrix, reporting non-positive definiteness."""
    try:
        return scipy.linalg.cho_factor(cov)
    except scipy.linalg.LinAlgError as exc:
        raise FactorizationError() from exc


def _chunk_size(n_grid, n_obs):
    """Grid points per batch so peak working memory stays within _MEMORY_BUDGET_BYTES.

    Each chunk iteration allocates ~4 arrays of shape (chunk, n_obs):
    K, beta, weights, and temporaries.  We divide the budget by 4 to account
    for this.  When the whole grid fits in one chunk (the common 2D case)
    the loop body executes exactly once, identical to the old fully-vectorised path.
    """
    arrays_per_chunk = 4
    return min(n_grid, max(1, _MEMORY_BUDGET_BYTES // (arrays_per_chunk * n_obs * 8)))


def _build_cross_cov_matrix(variogram, grid_coords, obs_locations):
    """Cross-covariance between supplied grid points and observations."""
    cov = np.empty((len(grid_coords), len(obs_locations)))
    for j, obs_location in enumerate(obs_locations):
        cov[:, j] = variogram.corr_array(grid_coords - obs_location)
    return cov


def _obs_interp_coords(obs_locations, grid: GridSpec):
    relative_locations = obs_locations - grid.origin
    grid_max = (np.asarray(grid.shape) - 1) * grid.spacing
    if np.any(relative_locations < 0) or np.any(relative_locations > grid_max):
        raise ValueError("obs_locations has values outside grid bounds for simulation")
    return (relative_locations / grid.spacing).T


def _interpolate_at_obs(field, obs_coords):
    return scipy.ndimage.map_coordinates(field, obs_coords, order=1)


def _simulate_uncond(variogram, grid: GridSpec):
    grid_args = tuple(value for axis in zip(grid.shape, grid.spacing) for value in axis)
    raw = gaussianfft.simulate(variogram, *grid_args)
    return np.asarray(raw).reshape(grid.shape, order='F')


def _require_fitted(system):
    if system is None:
        raise RuntimeError("Call fit(obs, variogram) before predict or simulate")
    return system


# ---------------------------------------------------------------------------
# Kriging classes
# ---------------------------------------------------------------------------


@dataclass
class GridSpec:
    """Regular grid with axes in x, y, z order and an optional world-space origin."""

    shape: tuple[int, ...]
    spacing: tuple[float, ...]
    origin: np.ndarray | None = None

    def __post_init__(self):
        self.shape = tuple(self.shape)
        self.spacing = tuple(self.spacing)
        if not 1 <= len(self.shape) <= 3:
            raise ValueError("grid_shape must contain 1, 2, or 3 dimensions")
        if len(self.spacing) != len(self.shape):
            raise ValueError("grid_spacing must have the same length as grid_shape")
        if any(
            not isinstance(size, (int, np.integer))
            or isinstance(size, (bool, np.bool_)) or size <= 0
            for size in self.shape
        ):
            raise ValueError("grid_shape must contain positive integers")
        if not np.all(np.isfinite(self.spacing)) or np.any(np.asarray(self.spacing) <= 0):
            raise ValueError("grid_spacing must contain positive finite values")
        self.origin = (
            np.zeros(len(self.shape)) if self.origin is None
            else np.array(self.origin, dtype=float, copy=True)
        )
        if self.origin.shape != (len(self.shape),) or not np.all(np.isfinite(self.origin)):
            raise ValueError("origin must contain one finite value per grid dimension")


@dataclass
class ObservationData:
    """Observation coordinates, values, and error standard deviations in field units."""

    locations: np.ndarray
    values: np.ndarray
    uncertainties: np.ndarray

    def __post_init__(self):
        self.locations = np.array(self.locations, dtype=float, copy=True)
        self.values = np.array(self.values, dtype=float, copy=True)
        self.uncertainties = np.array(self.uncertainties, dtype=float, copy=True)
        if self.locations.ndim != 2:
            raise ValueError("obs_locations must be a 2D array")
        if not 1 <= self.locations.shape[1] <= 3:
            raise ValueError("obs_locations must have 1, 2, or 3 columns")
        count = len(self.locations)
        if count == 0:
            raise ValueError("at least one observation is required")
        if self.values.shape != (count,):
            raise ValueError(f"obs_values must have shape ({count},), got {self.values.shape}")
        if self.uncertainties.shape != (count,):
            raise ValueError(
                f"obs_uncertainties must have shape ({count},), got {self.uncertainties.shape}"
            )
        if not all(np.all(np.isfinite(array)) for array in (
            self.locations, self.values, self.uncertainties,
        )):
            raise ValueError("observations must contain only finite values")
        if np.any(self.uncertainties < 0):
            raise ValueError("obs_uncertainties must be non-negative")


class _ObservationSystem:
    """Grid-independent observation snapshot and covariance factorization."""

    def __init__(self, obs: ObservationData, variogram: gaussianfft.Variogram):
        if not isinstance(variogram, gaussianfft.Variogram):
            raise TypeError("variogram must be a gaussianfft.Variogram instance")
        self.obs = ObservationData(obs.locations, obs.values, obs.uncertainties)
        self.variogram = variogram
        covariance = _build_obs_cov_matrix(variogram, self.obs.locations, self.obs.uncertainties)
        self.factor = _cholesky_factor(covariance)


class Kriging(abc.ABC):
    """Kriging interface allowing preparation to be deferred until evaluation."""

    @abc.abstractmethod
    def fit(self, obs: ObservationData, variogram: gaussianfft.Variogram) -> Kriging:
        """Accept observations and a configured variogram, returning this estimator.

        Preparation may be deferred. Refit after changing the variogram; no
        variogram parameters are estimated by this interface.
        """

    @abc.abstractmethod
    def predict(
        self, grid: GridSpec, calculate_stddev=False,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Return a grid-shaped prediction and optional standard deviation."""

    @abc.abstractmethod
    def simulate(self, grid: GridSpec, n_sim=1) -> list[np.ndarray]:
        """Generate conditional fields on a grid covering all observation locations."""


class SimpleKriging(Kriging):
    """Simple kriging with a known constant scalar mean."""

    def __init__(self, mean: float = 0.0):
        if np.ndim(mean) != 0 or not np.isrealobj(mean):
            raise ValueError("mean must be a finite scalar")
        self.mean = float(mean)
        if not np.isfinite(self.mean):
            raise ValueError("mean must be a finite scalar")
        self._system = None

    def fit(self, obs: ObservationData, variogram: gaussianfft.Variogram) -> SimpleKriging:
        self._system = _ObservationSystem(obs, variogram)
        return self

    def predict(
        self, grid: GridSpec, calculate_stddev=False,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        system = _require_fitted(self._system)
        grid_coords = _grid_coordinates(grid, system.obs.locations.shape[1])
        residuals = system.obs.values - self.mean
        correction, stddev = self._krige(grid, grid_coords, residuals, calculate_stddev)
        correction += self.mean
        return correction, stddev

    def simulate(self, grid: GridSpec, n_sim=1) -> list[np.ndarray]:
        system = _require_fitted(self._system)
        grid_coords = _grid_coordinates(grid, system.obs.locations.shape[1])
        obs_coords = _obs_interp_coords(system.obs.locations, grid)
        results = []

        for _ in range(n_sim):
            uncond = _simulate_uncond(system.variogram, grid)
            uncond_at_obs = _interpolate_at_obs(uncond, obs_coords)
            obs_residuals = (system.obs.values - self.mean) - uncond_at_obs

            correction, _ = self._krige(grid, grid_coords, obs_residuals)
            results.append(self.mean + uncond + correction)
        return results

    def _krige(self, grid, grid_coords, residuals, calculate_stddev=False):
        """Apply simple kriging to mean-adjusted observation or simulation residuals."""
        system = _require_fitted(self._system)
        weights = scipy.linalg.cho_solve(system.factor, residuals)
        n_grid = len(grid_coords)
        n_obs = len(system.obs.locations)
        chunk = _chunk_size(n_grid, n_obs)
        correction_flat = np.empty(n_grid)
        variance_flat = np.empty(n_grid) if calculate_stddev else None
        for start in range(0, n_grid, chunk):
            end = min(start + chunk, n_grid)
            cross_cov = _build_cross_cov_matrix(
                system.variogram, grid_coords[start:end], system.obs.locations,
            )
            correction_flat[start:end] = cross_cov @ weights
            if calculate_stddev:
                alpha = scipy.linalg.cho_solve(system.factor, cross_cov.T)
                variance_flat[start:end] = 1.0 - np.sum(cross_cov * alpha.T, axis=1)

        correction = correction_flat.reshape(grid.shape)
        stddev = None
        if calculate_stddev:
            stddev = np.sqrt(np.maximum(variance_flat, 0.0)).reshape(grid.shape)
        return correction, stddev


class OrdinaryKriging(Kriging):
    """Ordinary kriging with an unknown constant mean estimated during fit.

    The unbiasedness constraint (weights sum to 1) is enforced via a Lagrange
    multiplier.  Weights and drift coefficient are computed on-the-fly in
    chunks via the Schur complement of C, using only the Cholesky factor of C.
    Beyond the covariance factor, only O(n_obs) scalars are stored after fit::

        alpha  = C^{-1} 1                          (n_obs,)  stored
        s      = 1^T alpha                          scalar    stored
        -- per chunk (chunk_size grid points at a time) --
        beta   = C^{-1} K^T                         (n_obs, cs)
        mu(x)  = (alpha^T K^T - 1) / s             (cs,)  [drift coefficient]
        w(x)   = beta - alpha * mu(x)               (n_obs, cs)

    Kriging mean:     z*(x) = w(x)^T z_obs
    Kriging variance: sigma^2(x) = C(0) - k(x)^T w(x) - mu(x)
    """

    def __init__(self):
        self._system = None
        self._alpha = np.empty(0)
        self._s = 0.0
        self.estimated_mean = None

    def fit(self, obs: ObservationData, variogram: gaussianfft.Variogram) -> OrdinaryKriging:
        system = _ObservationSystem(obs, variogram)
        n_obs = len(system.obs.locations)
        ones = np.ones(n_obs)
        self._alpha = scipy.linalg.cho_solve(system.factor, ones)
        self._s = float(self._alpha.sum())
        gamma = scipy.linalg.cho_solve(system.factor, system.obs.values)
        self.estimated_mean = float(gamma.sum() / self._s)
        self._system = system
        return self

    def _ok_weights_chunk(self, cross_cov):
        """Return ordinary-kriging weights and drift for a cross-covariance chunk."""
        system = _require_fitted(self._system)
        beta = scipy.linalg.cho_solve(system.factor, cross_cov.T)
        drift = (self._alpha @ cross_cov.T - 1.0) / self._s
        return beta - np.outer(self._alpha, drift), drift

    def predict(
        self, grid: GridSpec, calculate_stddev=False,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        system = _require_fitted(self._system)
        grid_coords = _grid_coordinates(grid, system.obs.locations.shape[1])
        return self._krige(grid, grid_coords, system.obs.values, calculate_stddev)

    def simulate(self, grid: GridSpec, n_sim=1) -> list[np.ndarray]:
        system = _require_fitted(self._system)
        grid_coords = _grid_coordinates(grid, system.obs.locations.shape[1])
        obs_coords = _obs_interp_coords(system.obs.locations, grid)
        results = []

        for _ in range(n_sim):
            uncond = _simulate_uncond(system.variogram, grid)
            uncond_at_obs = _interpolate_at_obs(uncond, obs_coords)
            obs_residuals = system.obs.values - uncond_at_obs

            correction, _ = self._krige(grid, grid_coords, obs_residuals)
            results.append(uncond + correction)
        return results

    def _krige(self, grid, grid_coords, residuals, calculate_stddev=False):
        """Apply ordinary kriging to observation values or simulation residuals."""
        system = _require_fitted(self._system)
        n_grid = len(grid_coords)
        n_obs = len(system.obs.locations)
        chunk = _chunk_size(n_grid, n_obs)
        correction_flat = np.empty(n_grid)
        variance_flat = np.empty(n_grid) if calculate_stddev else None
        for start in range(0, n_grid, chunk):
            end = min(start + chunk, n_grid)
            cross_cov = _build_cross_cov_matrix(
                system.variogram, grid_coords[start:end], system.obs.locations,
            )
            weights, drift = self._ok_weights_chunk(cross_cov)
            correction_flat[start:end] = weights.T @ residuals
            if calculate_stddev:
                variance_flat[start:end] = 1.0 - np.sum(cross_cov * weights.T, axis=1) - drift

        correction = correction_flat.reshape(grid.shape)
        stddev = None
        if calculate_stddev:
            stddev = np.sqrt(np.maximum(variance_flat, 0.0)).reshape(grid.shape)
        return correction, stddev
