# -*- coding: utf-8 -*-
from ..imports import *


def plot_dendogram(data, cols=None, front_size=16):
    if cols is None:
        cols = data.columns
    # Redundant Features
    corr = np.nan_to_num(np.round(spearmanr(data[cols]).correlation, 4))

    for i in range(corr.shape[0]):
        corr[i, i] = 1

    corr_condensed = hc.distance.squareform(1 - corr)
    z = hc.linkage(corr, method='average')
    _, ax = plt.subplots(figsize=(16, 10))
    dendrogram = hc.dendrogram(
        z,
        labels=cols,
        orientation='left',
        leaf_font_size=front_size)
    return ax


def display_time_split(index_list):
    """ Display how the time series data is split

    """

    fig, ax = plt.subplots(figsize=(9, 2.5))
    ax_list = []
    for idxs in index_list:
        # Plot training and test indices
        l1 = ax.scatter(idxs, [1] * len(idxs), c='royalblue', marker='_', lw=6)
    l2 = ax.scatter(val_idx, [1] * len(val_idx), c='orange', marker='_', lw=6)
    l3 = ax.scatter(test_idx, [1] * len(test_idx), c='red', marker='_', lw=6)
    ax.legend([l1, l2, l3], ['Training', 'Validation', 'Test'])
    ax.set(ylim=[0.8, 1.5], title='Split behavior', xlabel='date')
    plt.xticks(rotation=90)
