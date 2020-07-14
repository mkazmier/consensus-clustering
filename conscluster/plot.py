import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from .conscluster import cdf, cdf_diff


def plot_consensus_matrix(M, **kwargs):
    '''Visualize a consensus matrix as a heatmap.

    This function can be used to quickly visually inspect the stability of the
    clusters. A perfectly stable clustering forms a sharp block diagonal
    structure with all entries close to either 1 or 0. The consensus matrix
    entries are re-ordered using hierarchical clustering with consensus score
    as distance measure to emphasize the block diagonal structure.

    Parameters
    ----------
    M : np.ndarray, shape (n_samples, n_samples)
        The consensus matrix as returned by `consensus_clustering`.
    kwargs, optional
        Additional keyword arguments to `seaborn.heatmap`.

    Returns
    -------
    heatmap
        The plotted heatmap
    '''
    hc = hierarchy.linkage(squareform(np.ones_like(M) - M), method='single', optimal_ordering=True)
    dend = hierarchy.dendrogram(hc, no_plot=True)
    ids = dend['leaves']
    return sns.heatmap(M[ids][:, ids], cmap='Reds', **kwargs)


def plot_pdf(M, **kwargs):
    '''Plot the empirical probability density function of the consensus scores.

    This function can be used to visually inspect the stability of clustering.
    A stable clustering corresponds to a bimodal PDF with sharp peaks at 0 and 1.
    PDF close to uniform indicates lack of stable cluster structure. See paper
    for examples.

    Parameters
    ----------
    M : np.ndarray, shape (n_samples, n_samples)
        The consensus matrix as returned by `consensus_clustering`.
    kwargs, optional
        Additional keyword arguments to `seaborn.heatmap`.

    Returns
    -------
    pdf
        The plotted PDF.
    '''
    return sns.distplot(M[np.triu_indices_from(M)], **kwargs)


def plot_cdfs(M_list, ax=None, labels=None):
    '''Plot the empirical CDFs for a sequence of consensus matrices.

    This function can be used to visually inspect the stability of clustering
    and select the optimal number of clusters.
    The number of clusters leading to the most stable clustering should appear
    the most step-like (see fig. 3 in the paper for examples).

    Parameters
    ----------
    M_list : list of np.ndarray
        A sequence of consensus matrices generated using different number of
        clusters.
    ax : matplotlib.axes.Axes, optional
        The axes to plot the CDFs on. If None, a new set of axes is created.
    labels : list or sequence, optional
        The labels to use for the CDFs. If None, it's assumed that `M_list` is
        a sequence corresponding to increasing number of clusters, starting
        from 2.

    Returns
    -------
    ax 
        The plotted CDFs.
    '''
    if not ax:
        fig, ax = plt.subplots(1)
    if not labels:
        labels = range(2, len(M_list) + 2)
    for i, M in zip(labels, M_list):
        xx, f = cdf(M)
        ax.plot(xx, f, label='{} clusters'.format(i))
    ax.legend()
    return ax


def plot_cdf_diff(M_list, ax=None, labels=None):
    '''Plot the difference in areas under empirical CDFs for a sequence of
    consensus matrices.

    This function can be used to visually inspect the stability of clustering
    and select the optimal number of clusters.
    The number of clusters leading to the most stable clustering will give
    the highest relative increase in area under CDF. See fig. 3 in the paper
    for examples with 1 and 3 clusters.

    Parameters
    ----------
    M_list : list of np.ndarray
        A sequence of consensus matrices generated using different number of
        clusters.
    ax : matplotlib.axes.Axes, optional
        The axes to plot the CDFs on. If None, a new set of axes is created.
    labels : list or sequence, optional
        The labels to use for the CDFs. If None, it's assumed that `M_list` is
        a sequence corresponding to increasing number of clusters, starting
        from 2.

    Returns
    -------
    ax 
        The plotted CDF differences.
    '''
    if not ax:
        fig, ax = plt.subplots(1)
    if not labels:
        labels = range(2, len(M_list) + 2)
    diffs = cdf_diff(M_list)
    plt.plot(labels, diffs)
    return ax
