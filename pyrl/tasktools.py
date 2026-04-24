"""
Utilities for defining and working with cognitive tasks.
"""
from collections import OrderedDict
import numpy as np


def to_map(*args):
    """Convert list of names to ordered dictionary mapping names to indices."""
    if isinstance(args[0], list):
        l = args[0]
    else:
        l = args

    od = OrderedDict()
    for i, v in enumerate(l):
        od[v] = i

    return od


def get_idx(t, time_range):
    """Get indices where t is within time_range."""
    start, end = time_range
    return list(np.where((start <= t) & (t < end))[0])


def get_epochs_idx(dt, epochs):
    """Convert epoch time ranges to index ranges."""
    t = np.linspace(0, epochs['tmax'], int(epochs['tmax']/dt)+1)
    return t, {k: get_idx(t, v) for k, v in epochs.items() if k != 'tmax'}


def choice(rng, a):
    """Select random element from array."""
    return a[rng.choice(len(a))]


def unravel_index(i, dims):
    """Multi-dimensional index from flat index."""
    return list(np.unravel_index(i % np.prod(dims), dims, order='F'))


def uniform(rng, dt, xmin, xmax):
    """Random duration that's a multiple of dt."""
    return (rng.uniform(xmin, xmax)//dt)*dt


def divide(x, y):
    """Safe division that returns 0 for division by zero."""
    try:
        z = x/y
        if np.isnan(z):
            raise ZeroDivisionError
        return z
    except ZeroDivisionError:
        return 0


def correct_2AFC(perf):
    """Calculate decision and correct rates for 2-alternative forced choice."""
    p_decision = perf.n_decision/perf.n_trials
    p_correct  = divide(perf.n_correct, perf.n_decision)
    return p_decision, p_correct


class Task:
    """Generic task base class."""
    def __init__(self):
        pass
