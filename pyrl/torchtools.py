"""PyTorch utilities (replacement for theanotools.py)."""

import numpy as np
import torch


def asarray(x, dtype=torch.float32):
    """Convert to torch tensor with specified dtype."""
    if isinstance(x, torch.Tensor):
        return x.to(dtype)
    return torch.tensor(np.asarray(x), dtype=dtype)


def zeros(shape, dtype=torch.float32, device=None):
    """Create zero tensor."""
    return torch.zeros(shape, dtype=dtype, device=device)


def shared(x, name=None, device=None):
    """Create a tensor (equivalent to Theano shared variable)."""
    if isinstance(x, torch.Tensor):
        return x.to(device) if device else x
    return torch.tensor(np.asarray(x, dtype=np.float32), device=device)


def clipping_multiplier(norm, max_norm):
    """
    Multiplier for renormalizing a vector.

    Parameters
    ----------
    norm : torch.Tensor
        Norm of the gradient
    max_norm : float
        Maximum allowed norm

    Returns
    -------
    torch.Tensor
        Multiplier to clip gradient
    """
    return torch.where(norm > max_norm, max_norm / norm, torch.ones_like(norm))


def choice(rng, a, size=1, replace=True, p=None):
    """
    A version of `numpy.random.RandomState.choice` that works with `float32`.

    Parameters
    ----------
    rng : np.random.RandomState
        Random number generator
    a : int or array-like
        Population to sample from
    size : int
        Number of samples
    replace : bool
        Whether to sample with replacement
    p : array-like, optional
        Probabilities

    Returns
    -------
    int or ndarray
        Sampled indices
    """
    # Format and Verify input
    if isinstance(a, int):
        if a > 0:
            pop_size = a  # population size
        else:
            raise ValueError("a must be greater than 0")
    else:
        a = np.array(a, ndmin=1, copy=False)
        if a.ndim != 1:
            raise ValueError("a must be 1-dimensional")
        pop_size = a.size
        if pop_size == 0:
            raise ValueError("a must be non-empty")

    if p is not None:
        p = np.array(p, dtype=p.dtype, ndmin=1, copy=False)
        if p.ndim != 1:
            raise ValueError("p must be 1-dimensional")
        if p.size != pop_size:
            raise ValueError("a and p must have same size")
        if np.any(p < 0):
            raise ValueError("probabilities are not non-negative")
        if not np.allclose(p.sum(), 1):
            raise ValueError("probabilities do not sum to 1")

    # Actual sampling
    if replace:
        if p is not None:
            cdf = p.cumsum()
            cdf /= cdf[-1]
            uniform_samples = rng.rand(size)
            idx = cdf.searchsorted(uniform_samples, side='right')
        else:
            idx = rng.randint(0, pop_size, size=size)
    else:
        if size > pop_size:
            raise ValueError(''.join(["Cannot take a larger sample than ",
                                      "population when 'replace=False'"]))

        if p is not None:
            if np.sum(p > 0) < size:
                raise ValueError("Fewer non-zero entries in p than size")
            n_uniq = 0
            p = p.copy()
            found = np.zeros(size, dtype=np.int32)
            while n_uniq < size:
                x = rng.rand(size - n_uniq)
                if n_uniq > 0:
                    p[found[0:n_uniq]] = 0
                cdf = np.cumsum(p)
                cdf /= cdf[-1]
                new = cdf.searchsorted(x, side='right')
                new = np.unique(new)
                found[n_uniq:n_uniq + new.size] = new
                n_uniq += new.size
            idx = found
        else:
            idx = rng.permutation(pop_size)[:size]

    assert len(idx) == 1
    return idx[0]


# Output activations
def relu(x):
    """ReLU activation."""
    return torch.relu(x)


def softmax(x, temp=1, dim=-1):
    """Softmax with temperature."""
    y = torch.exp(x / temp)
    return y / y.sum(dim=dim, keepdim=True)


def log_softmax(x, temp=1, dim=-1):
    """Log softmax with temperature."""
    y = x / temp
    y = y - y.max(dim=dim, keepdim=True)[0]
    return y - torch.log(torch.exp(y).sum(dim=dim, keepdim=True))


def normalization(x):
    """Normalize by sum of squares."""
    x2 = torch.square(x) + 1e-6
    return x2 / torch.sum(x2, dim=-1, keepdim=True)


def normalization3(x):
    """3D normalization."""
    sh = x.shape
    x = x.reshape((sh[0] * sh[1], sh[2]))
    y = normalization(x)
    y = y.reshape(sh)
    return y


def get_processor_type():
    """Check if GPU is available."""
    return 'cuda' if torch.cuda.is_available() else 'cpu'
