"""Matrix operations and utilities."""

import numpy as np


def spectral_radius(A):
    """
    Return the spectral radius of matrix `A`.

    Parameters
    ----------
    A : ndarray
        Matrix to compute spectral radius for

    Returns
    -------
    float
        Maximum absolute eigenvalue of A
    """
    return np.max(abs(np.linalg.eigvals(A)))
