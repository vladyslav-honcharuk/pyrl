"""Data partitioning and analysis utilities."""

import numpy as np


def partition(X, Y, nbins=None, Xedges=None):
    """
    Partition data into bins for analysis.

    Parameters
    ----------
    X : array-like
        X values
    Y : array-like
        Y values corresponding to X
    nbins : int, optional
        Number of bins to create
    Xedges : array-like, optional
        Explicit bin edges

    Returns
    -------
    Xbins : list of arrays
        X values in each bin
    Ybins : list of arrays
        Y values in each bin
    Xedges : ndarray
        Bin edges
    binsizes : ndarray
        Number of samples in each bin
    """
    assert nbins is not None or Xedges is not None
    assert len(X) == len(Y)

    X = np.asarray(X)
    Y = np.asarray(Y)

    if Xedges is None:
        idx = np.argsort(X)
        Xsorted = X[idx]
        Ysorted = Y[idx]

        inc = len(X) // nbins
        p = [i * inc for i in range(nbins + 1)]
        if p[-1] != len(X):
            p[-1] = len(X)
        assert len(p) == nbins + 1

        Xbins = [Xsorted[p[i]:p[i+1]] for i in range(nbins)]
        Ybins = [Ysorted[p[i]:p[i+1]] for i in range(nbins)]

        Xedges = np.array([Xsorted[0]]
                          + [(Xbins[i][-1] + Xbins[i+1][0]) / 2 for i in range(nbins-1)]
                          + [Xsorted[-1]])
    else:
        nbins = len(Xedges) - 1
        wbins = [np.where((Xedges[i] <= X) & (X < Xedges[i+1]))[0]
                 for i in range(nbins-1)]
        wbins.append(np.where((Xedges[nbins-1] <= X) & (X <= Xedges[nbins]))[0])

        Xbins = [X[w] for w in wbins]
        Ybins = [Y[w] for w in wbins]

    binsizes = np.array([len(Xbin) for Xbin in Xbins])
    assert(np.sum(binsizes) == len(X))

    return Xbins, Ybins, Xedges, binsizes
