from gaussianfft._kriging.grid_spec import GridSpec
from gaussianfft._kriging.kriging_toolkit import OrdinaryKriging, SimpleKriging
from gaussianfft._kriging.observation_data import ObservationData

_METHODS = {
    'SimpleKriging': SimpleKriging,
    'OrdinaryKriging': OrdinaryKriging,
}


def _build_kriging(method, obs, variogram, mean):
    cls = _METHODS.get(method)
    if cls is None:
        raise ValueError(
            f"Unknown kriging method: {method!r}. Choose from {list(_METHODS)}"
        )
    if method == 'SimpleKriging':
        estimator = cls(mean=mean)
    else:
        estimator = cls()
    estimator.fit(obs, variogram)
    return estimator


def predict(
        variogram, grid_shape: tuple[int, ...], grid_spacing: tuple[float, ...],
        obs_locations, obs_values, obs_uncertainties, *,
        method='SimpleKriging', mean: float = 0.0, calculate_stddev=False,
):
    """Predict on a grid defined by shape and spacing tuples in x, y, z order."""
    grid = GridSpec(grid_shape, grid_spacing)
    obs = ObservationData(obs_locations, obs_values, obs_uncertainties)
    estimator = _build_kriging(method, obs, variogram, mean)
    return estimator.predict(grid, calculate_stddev=calculate_stddev)


def simulate(
        variogram, grid_shape: tuple[int, ...], grid_spacing: tuple[float, ...],
        obs_locations, obs_values, obs_uncertainties, *,
        method='SimpleKriging', mean: float = 0.0, n_sim=1,
):
    """Simulate on a grid defined by shape and spacing tuples in x, y, z order."""
    grid = GridSpec(grid_shape, grid_spacing)
    obs = ObservationData(obs_locations, obs_values, obs_uncertainties)
    estimator = _build_kriging(method, obs, variogram, mean)
    return estimator.simulate(grid, n_sim=n_sim)