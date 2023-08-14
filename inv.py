import numpy as np
from numba import njit


def invert(xyz: np.ndarray, inv_center: np.ndarray) -> np.ndarray:
    """
    Invert `xyz` with respect to `inv_center`

    :param xyz:        Nx3 coordinate array
    :param inv_center: 1x3 coordinate array
    """
    return 2 * inv_center - xyz


@njit(cache=True, error_model="numpy")
def invert_numba(xyz: np.ndarray, inv_center: np.ndarray) -> np.ndarray:
    """
    Invert `xyz` with respect to `inv_center`

    :param xyz:        Nx3 coordinate array
    :param inv_center: 1x3 coordinate array
    """
    return 2 * inv_center - xyz
