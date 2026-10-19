import numpy as np
try:
    import scipy.linalg
    import scipy.ndimage
except ImportError:
    raise ImportError("Scipy is not installed, and is required to use these utilities")

import gaussianfft
from gaussianfft._kriging.kriging_data import KrigingData

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

def _grid_coordinates(nx, dx, ny, dy, nz, dz, ndims):
    x = np.arange(nx) * dx
    if ndims == 1:
        return x.reshape(-1, 1)
    y = np.arange(ny) * dy
    if ndims == 2:
        xx, yy = np.meshgrid(x, y, indexing='ij')
        return np.column_stack([xx.ravel(), yy.ravel()])
    z = np.arange(nz) * dz
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    return np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])


def _build_obs_cov_matrix(variogram, obs_locations, obs_uncertainties):
    n = len(obs_locations)
    cov = np.empty((n, n))
    for i in range(n):
        correlations = variogram.corr_array(obs_locations[i] - obs_locations[i:])
        cov[i, i:] = correlations
        cov[i:, i] = correlations
    cov[np.diag_indices(n)] += obs_uncertainties ** 2 + 1e-10
    return cov


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


def _obs_interp_coords(obs_locations, dx, dy, dz, ndims):
    steps = np.array((dx, dy, dz)[:ndims])
    return (obs_locations / steps).T


def _interpolate_at_obs(field, obs_coords):
    return scipy.ndimage.map_coordinates(field, obs_coords, order=1)


def _simulate_uncond(data: KrigingData):
    if data.ndims == 1:
        raw = gaussianfft.simulate(data.variogram, data.nx, data.dx)
    elif data.ndims == 2:
        raw = gaussianfft.simulate(data.variogram, data.nx, data.dx, data.ny, data.dy)
    else:
        raw = gaussianfft.simulate(
            data.variogram, data.nx, data.dx, data.ny, data.dy, data.nz, data.dz
        )
    return np.array(raw).reshape(data.grid_shape, order='F')


# ---------------------------------------------------------------------------
# Kriging classes
# ---------------------------------------------------------------------------


class SimpleKriging:
    """Simple kriging using KrigingData and a known scalar or grid-shaped mean."""

    def __init__(self, data: KrigingData, *, mean=0.0):
        self._data = data
        self.mean = mean

        if isinstance(self.mean, np.ndarray) and self.mean.shape != data.grid_shape:
            raise ValueError(
                f"mean must be a scalar or have shape {data.grid_shape}, got {self.mean.shape}"
            )

        cov = _build_obs_cov_matrix(data.variogram, data.obs_locations, data.obs_uncertainties)
        try:
            self._cho_factor = scipy.linalg.cho_factor(cov)
        except scipy.linalg.LinAlgError as exc:
            raise FactorizationError() from exc

        self._grid_coords = _grid_coordinates(
            data.nx, data.dx, data.ny, data.dy, data.nz, data.dz, data.ndims
        )
        self._obs_coords = _obs_interp_coords(
            data.obs_locations, data.dx, data.dy, data.dz, data.ndims
        )

    def _mean_components(self):
        if isinstance(self.mean, np.ndarray):
            return _interpolate_at_obs(self.mean, self._obs_coords), self.mean
        return self.mean, self.mean

    def predict(self, calculate_stddev=False):
        mean_at_obs, mean_field = self._mean_components()
        residuals = self._data.obs_values - mean_at_obs
        correction, stddev = self._krige(residuals, calculate_stddev=calculate_stddev)
        correction += mean_field
        return correction, stddev

    def simulate(self, n_sim=1):
        results = []

        for _ in range(n_sim):
            mean_at_obs, mean_field = self._mean_components()
            uncond = _simulate_uncond(self._data)
            uncond_at_obs = _interpolate_at_obs(uncond, self._obs_coords)
            obs_residuals = (self._data.obs_values - mean_at_obs) - uncond_at_obs

            correction, _ = self._krige(obs_residuals)
            results.append(mean_field + uncond + correction)
        return results

    def _krige(self, residuals, calculate_stddev=False):
        """Apply simple kriging to mean-adjusted observation or simulation residuals."""
        weights = scipy.linalg.cho_solve(self._cho_factor, residuals)
        n_grid = self._grid_coords.shape[0]
        n_obs = len(self._data.obs_locations)
        chunk = _chunk_size(n_grid, n_obs)
        correction_flat = np.empty(n_grid)
        variance_flat = np.empty(n_grid) if calculate_stddev else None
        for start in range(0, n_grid, chunk):
            end = min(start + chunk, n_grid)
            K = _build_cross_cov_matrix(
                self._data.variogram,
                self._grid_coords[start:end],
                self._data.obs_locations,
            )
            correction_flat[start:end] = K @ weights
            if calculate_stddev:
                alpha = scipy.linalg.cho_solve(self._cho_factor, K.T)     # (n_obs, cs)
                variance_flat[start:end] = 1.0 - np.sum(K * alpha.T, axis=1)

        correction = correction_flat.reshape(self._data.grid_shape)
        stddev = None
        if calculate_stddev:
            stddev = np.sqrt(np.maximum(variance_flat, 0.0)).reshape(self._data.grid_shape)
        return correction, stddev


class OrdinaryKriging:
    """Ordinary Kriging: estimates an unknown constant mean from KrigingData.

    The unbiasedness constraint (weights sum to 1) is enforced via a Lagrange
    multiplier.  Weights and drift coefficient are computed on-the-fly in
    chunks via the Schur complement of C, using only the Cholesky factor of C.
    Only O(n_obs) scalars are stored after construction::

        alpha  = C^{-1} 1                          (n_obs,)  stored
        s      = 1^T alpha                          scalar    stored
        -- per chunk (chunk_size grid points at a time) --
        beta   = C^{-1} K^T                         (n_obs, cs)
        mu(x)  = (alpha^T K^T - 1) / s             (cs,)  [drift coefficient]
        w(x)   = beta - alpha * mu(x)               (n_obs, cs)

    Kriging mean:     z*(x) = w(x)^T z_obs
    Kriging variance: sigma^2(x) = C(0) - k(x)^T w(x) - mu(x)
    """

    def __init__(self, data: KrigingData):
        self._data = data

        cov = _build_obs_cov_matrix(data.variogram, data.obs_locations, data.obs_uncertainties)
        try:
            self._cho_factor = scipy.linalg.cho_factor(cov)
        except scipy.linalg.LinAlgError as exc:
            raise FactorizationError() from exc

        self._grid_coords = _grid_coordinates(
            data.nx, data.dx, data.ny, data.dy, data.nz, data.dz, data.ndims
        )
        self._obs_coords = _obs_interp_coords(
            data.obs_locations, data.dx, data.dy, data.dz, data.ndims
        )

        # Schur complement scalars — O(n_obs) only, no grid-sized arrays stored.
        n_obs = len(data.obs_locations)
        ones = np.ones(n_obs)
        self._alpha = scipy.linalg.cho_solve(self._cho_factor, ones)  # (n_obs,)  C^{-1} 1
        self._s = float(self._alpha.sum())

        # BLUE estimate of the unknown constant mean: mu_hat = (1^T C^{-1} z) / s
        gamma = scipy.linalg.cho_solve(self._cho_factor, data.obs_values)
        self.estimated_mean: float = float(gamma.sum() / self._s)

    def _ok_weights_chunk(self, K):
        """Return OK weights for a chunk K (chunk_size, n_obs) -> w (n_obs, chunk_size)."""
        beta = scipy.linalg.cho_solve(self._cho_factor, K.T)              # (n_obs, cs)
        drift = (self._alpha @ K.T - 1.0) / self._s                      # (cs,)
        return beta - np.outer(self._alpha, drift), drift

    def predict(self, calculate_stddev=False):
        return self._krige(self._data.obs_values, calculate_stddev=calculate_stddev)

    def simulate(self, n_sim=1):
        results = []

        for _ in range(n_sim):
            uncond = _simulate_uncond(self._data)
            uncond_at_obs = _interpolate_at_obs(uncond, self._obs_coords)
            obs_residuals = self._data.obs_values - uncond_at_obs

            correction, _ = self._krige(obs_residuals)
            results.append(uncond + correction)
        return results

    def _krige(self, residuals, calculate_stddev=False):
        """Apply ordinary kriging to observation values or simulation residuals."""
        n_grid = self._grid_coords.shape[0]
        n_obs = len(self._data.obs_locations)
        chunk = _chunk_size(n_grid, n_obs)
        correction_flat = np.empty(n_grid)
        variance_flat = np.empty(n_grid) if calculate_stddev else None
        for start in range(0, n_grid, chunk):
            end = min(start + chunk, n_grid)
            K = _build_cross_cov_matrix(
                self._data.variogram,
                self._grid_coords[start:end],
                self._data.obs_locations,
            )
            w, drift = self._ok_weights_chunk(K)
            correction_flat[start:end] = w.T @ residuals
            if calculate_stddev:
                variance_flat[start:end] = 1.0 - np.sum(K * w.T, axis=1) - drift

        correction = correction_flat.reshape(self._data.grid_shape)
        stddev = None
        if calculate_stddev:
            stddev = np.sqrt(np.maximum(variance_flat, 0.0)).reshape(self._data.grid_shape)
        return correction, stddev
