# -*- coding: utf-8 -*-
from ..imports import *
from ..gen_functions import *


def plot_dendogram(data:pd.core.frame.DataFrame, cols=None, front_size=16,filename=None):
    """Plot hierarchical relationship between features in the data. 
    
    Args:
        data: dataframe from dataset.data
        cols(optional): columns to use for making the dendogram. If None, use the columns of the data dataFrame  
        front_size(optional): plot front size [default:None]
        filename(optional): if not None, save plot as filename[default:None]

    """
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
    
    if filename:
        plt.savefig(filename)


def display_time_split(index_list):
    """ Display how the time series data is split.

    Args: 
        index_list: a list of index in each split 

    """
    lenght = len(index_list)
    if lenght==3:
        colors = ['royalblue','orange','red']
        label_list = ['Training', 'Validation', 'Test']
    
    elif lenght == 4:
        colors = ['royalblue','orange','red','maroon']
        label_list = ['Training', 'Validation1', 'Validation2', 'Test']
    
    else:
        colors = get_color(color_length= length, cmap=cm.jet)

    fig, ax = plt.subplots(figsize=(9, 2.5))
    ax_list = []
    for idxs, color in zip(index_list,colors):
        # Plot training and test indices
        l1 = ax.scatter(idxs, [1] * len(idxs), c=color, marker='_', lw=6)
        ax_list.append(l1)
     
    ax.legend(ax_list, label_list)
    ax.set(ylim=[0.8, 1.5], title='Split behavior', xlabel='date')
    plt.xticks(rotation=90)


def show_fea_imp(fea_imp,x_log=False, filename=None,title=''):
    """Display feature of importance in a bar plot 

    Args: 
        imp_df: important dataframe 
        x_log: if True, plot x axis in a long scale
        filename: filename to save figure as 
        title: figure title
    
    """
    plt.rcParams.update({'font.size': 14})


    if 'imp_std' in fea_imp.columns:
        fea_imp.drop(0).plot('index','importance',kind='barh',xerr='imp_std',figsize=(5,8),linewidth=1,edgecolor='black',legend=False)
    else:
        fea_imp.drop(0).plot('index','importance',kind='barh',figsize=(5,8),linewidth=1,edgecolor='black', legend=False)

    if x_log:
        plt.xscale('log')

    plt.title(title)
    plt.xlabel('importance index')
    plt.tight_layout()
    if filename:
        plt.savefig(filename)