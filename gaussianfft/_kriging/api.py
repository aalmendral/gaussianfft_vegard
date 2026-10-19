from typing import Union

from numpy import ndarray

from gaussianfft._kriging.kriging_data import KrigingData
from gaussianfft._kriging.kriging_toolkit import SimpleKriging, OrdinaryKriging


_METHODS = {
    'SimpleKriging': SimpleKriging,
    'OrdinaryKriging': OrdinaryKriging,
}


def _build_kriging(method, data: KrigingData, mean):
    cls = _METHODS.get(method)
    if cls is None:
        raise ValueError(
            f"Unknown kriging method: {method!r}. Choose from {list(_METHODS)}"
        )
    if method == 'SimpleKriging':
        return SimpleKriging(data, mean=mean)
    elif method == 'OrdinaryKriging':
        return OrdinaryKriging(data)
    else:
        raise ValueError(f"Unsupported kriging method: {method!r}")


def _build_kriging_data(
        variogram, grid_shape: tuple[int, ...], grid_spacing: tuple[float, ...],
        obs_locations, obs_values, obs_uncertainties,
) -> KrigingData:
    if not 1 <= len(grid_shape) <= 3:
        raise ValueError("grid_shape must contain 1, 2, or 3 dimensions")
    if len(grid_spacing) != len(grid_shape):
        raise ValueError("grid_spacing must have the same length as grid_shape")
    nx, ny, nz = grid_shape + (1,) * (3 - len(grid_shape))
    dx, dy, dz = grid_spacing + (1.0,) * (3 - len(grid_spacing))
    return KrigingData(variogram, nx, dx, ny, dy, nz, dz,
                       obs_locations, obs_values, obs_uncertainties)


def predict(
        variogram, grid_shape: tuple[int, ...], grid_spacing: tuple[float, ...],
        obs_locations, obs_values, obs_uncertainties, *,
        method='SimpleKriging', mean: Union[float, ndarray] = 0.0, calculate_stddev=False,
):
    """Predict on a grid defined by shape and spacing tuples in x, y, z order."""
    data = _build_kriging_data(
        variogram, grid_shape, grid_spacing, obs_locations, obs_values, obs_uncertainties,
    )
    k = _build_kriging(method, data, mean)
    return k.predict(calculate_stddev=calculate_stddev)


def simulate(
        variogram, grid_shape: tuple[int, ...], grid_spacing: tuple[float, ...],
        obs_locations, obs_values, obs_uncertainties, *,
        method='SimpleKriging', mean: Union[float, ndarray] = 0.0, n_sim=1,
):
    """Simulate on a grid defined by shape and spacing tuples in x, y, z order."""
    data = _build_kriging_data(
        variogram, grid_shape, grid_spacing, obs_locations, obs_values, obs_uncertainties,
    )
    k = _build_kriging(method, data, mean)
    return k.simulate(n_sim=n_sim)