"""NumPy utilities for random number generation and operations."""

import numpy as np


def get_rng(seed=0, loc='Not specified'):
    """
    Get a random number generator with specified seed.

    Parameters
    ----------
    seed : int
        Random seed
    loc : str
        Location/context for debugging

    Returns
    -------
    np.random.RandomState
        Random number generator
    """
    print("<< RNG {} >> {}".format(seed, loc))
    return np.random.RandomState(seed)


def relu(x):
    """
    ReLU activation function.

    Parameters
    ----------
    x : ndarray
        Input array

    Returns
    -------
    ndarray
        max(0, x)
    """
    return np.maximum(0, x)
