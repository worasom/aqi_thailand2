# -*- coding: utf-8 -*-
from ..imports import *
from ..gen_functions import *

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
        fea_imp.plot('index','importance',kind='barh',xerr='imp_std',figsize=(5,8),linewidth=1,edgecolor='black',legend=False)
    else:
        fea_imp.plot('index','importance',kind='barh',figsize=(5,8),linewidth=1,edgecolor='black', legend=False)

    if x_log:
        plt.xscale('log')

    plt.title(title)
    plt.xlabel('importance index')
    plt.tight_layout()
    if filename:
        plt.savefig(filename)


def plot_model_perf(dataset, model, split_list=[0.7, 0.3],xlim=[], to_save=True):
    """Plot model performance over training and test data
    Args:
        dataset: dataset object
        model: fitted model 
        split_list(optional): train/test spliting ratio[default:[0.7,0.3]]
        xlim(optional): if passed, use these value for xlim
        filename(optional): if not None, save the figure using the filename 
    """

     
    dataset.split_data(split_ratio=split_list)
    xtrn, ytrn, x_cols = dataset.get_data_matrix(use_index=dataset.split_list[0], x_cols=dataset.x_cols)
    xtest, ytest, _ = dataset.get_data_matrix(use_index=dataset.split_list[1], x_cols=dataset.x_cols)

    plt.figure(figsize=(10, 4))

    # plot data 
    plt.plot(dataset.split_list[0], ytrn, marker='.', label='data(blue)',linewidth=0,alpha=1, markersize=3,color='royalblue')
    plt.plot(dataset.split_list[1],ytest, marker='.',linewidth=0,alpha=1, markersize=3,color='royalblue')

    plt.plot(dataset.split_list[0],model.predict(xtrn), marker='.', label='train(green)',linewidth=0,alpha=0.3, markersize=2,color='lime')
    plt.plot(dataset.split_list[1],model.predict(xtest), marker='.', label='test(red)',linewidth=0,alpha=0.3, markersize=2, color='red')


    plt.legend()
    plt.title(f'Model Perfromance on Predicting {dataset.monitor}')
    plt.xlabel('date')
    plt.ylabel(dataset.monitor)

    if len(xlim)>0:
        plt.xlim(xlim)

    if to_save:
        poll_name = dataset.monitor.replace('.','')
        plt.savefig(dataset.report_folder + f'{poll_name}_model_perfomance.png')