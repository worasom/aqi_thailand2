# -*- coding: utf-8 -*-
from ..imports import *
from ..gen_functions import *
from ..models.predict_model import *
from .vis_data import *

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
    """Plot model performance over training and test data.
    - plot actual againt prediction for train and test set
    - take the daily average and plot actual again prediction for training and test set
    - return error of the training and test set as dataframe 

    Args:
        dataset: dataset object
        model: fitted model 
        split_list(optional): train/test spliting ratio[default:[0.7,0.3]]
        xlim(optional): if passed, use these value for xlim
        filename(optional): if not None, save the figure using the filename 

    Returns:
        trn_pred_df: actual, prediction of the training set (no averaging)
        test_pred_df: actual, prediction of the test set (no averaging)
    """
    dataset.split_data(split_list)
    ytrn_pred_df = cal_error(dataset, model, data_index=dataset.split_list[0])
    ytest_pred_df = cal_error(dataset, model, data_index=dataset.split_list[1])
     

    _, ax = plt.subplots(2, 1, figsize=(10, 8))

    # plot data 
    ax[0].plot(ytrn_pred_df['actual'], marker='.', label='data(blue)',linewidth=0,alpha=1, markersize=3,color='royalblue')
    ax[0].plot(ytest_pred_df['actual'], marker='.', linewidth=0, alpha=1, markersize=3,color='royalblue')

    ax[0].plot(ytrn_pred_df['pred'], marker='.', label='train(green)',linewidth=0,alpha=0.3, markersize=2,color='lime')
    ax[0].plot(ytest_pred_df['pred'], marker='.', label='test(red)',linewidth=0,alpha=0.3, markersize=2, color='red')

    ax[0].set_title(f'Model Perfromance on Predicting {dataset.monitor}')
    ax[1].set_title(f'Model Perfromance on Predicting Daily Avg of {dataset.monitor}')

    #if roll:
        # resample
    #    ytrn_pred_df_avg = ytrn_pred_df.rolling(24, min_periods=0).mean().dropna()
    #    ytest_pred_df_avg = ytest_pred_df.rolling(24, min_periods=0).mean().dropna()
    #else:
        # resample
    ytrn_pred_df_avg = ytrn_pred_df.resample('d').mean().dropna()
    ytest_pred_df_avg = ytest_pred_df.resample('d').mean().dropna()


    # plot data 
    ax[1].plot(ytrn_pred_df_avg['actual'], marker='.', label='data(blue)',linewidth=1,alpha=1, markersize=3,color='royalblue')
    ax[1].plot(ytest_pred_df_avg['actual'], marker='.', linewidth=1, alpha=1, markersize=3,color='royalblue')

    ax[1].plot(ytrn_pred_df_avg['pred'], marker='.', label='train(green)',linewidth=1,alpha=1, markersize=3,color='lime')
    ax[1].plot(ytest_pred_df_avg['pred'], marker='.', label='test(red)',linewidth=1,alpha=1, markersize=3, color='red')


    # add label 
    for a in ax:
        a.legend()
        a.set_xlabel('date')
        a.set_ylabel(dataset.monitor)
        if len(xlim)>0:
            a.set_xlim(xlim)

    plt.tight_layout()

    if to_save:
        poll_name = dataset.monitor.replace('.','')
        plt.savefig(dataset.report_folder + f'{poll_name}_model_perfomance.png')

    return ytrn_pred_df, ytest_pred_df

def plot_sea_error(trn_error, sea_error, filename=None):
    """Plase seasonal error the training data and the average value.

    Args:
        trn_error: raw training error 
        sea_error: seasonal error used in the inference
        filename: save filename
    """

    # plot seasonal error and save 
    _, ax = plt.subplots(figsize=(10,4))
    _ = plot_season_avg(trn_error, 'error', ax, plot_error=False, roll=True, agg='mean')
    ax.plot(sea_error['error'], linewidth=3)
    ax.set_title('error (by pollution season)')
    ax.legend(['raw training error', 'average error'])
    if filename:
        plt.savefig(filename)

def plt_infer_actual(ytest_pred_df_avg, band_df, filename=None):
    """Compare the actual and inference predicted data in timeseries format. 
    Args:
        ytest_pred_df_avg: actual data to plot
        band_df: inference data to plot with different quantiles 
        filename: save filename
    
    """
    plt.figure(figsize=(12,4))
    
    plt.plot(ytest_pred_df_avg['actual'],color='royalblue',label='avg data', marker='.', linewidth=1, alpha=0.8)
    plt.plot(band_df, linewidth=2)
    plt.xlabel('date')
    legend_list = band_df.columns.to_list()
    legend_list = ['samples '+ s for s in legend_list]
    plt.legend(['actual']+ legend_list,frameon=True)
    plt.title('Compare Actual Data and Inference Bands')

    if filename:
        plt.savefig(filename)


def plot_infer_season(poll_df, pollutant, sea_pred, color_zip, filename=None ):
    """Compare the actual and inference predicted data in seasonal format 

    Args:
        poll_df: pollution data 
        pollutant: pollutant to plot 
        sea_pred: inference data 
        color_zip: color and level indicating the AQI 
        filename: save filename 
        

    """

    _, ax = plt.subplots(1,1, figsize=(10,4),sharex=True)
    ax.plot(sea_pred)
    _ = plot_season_avg(poll_df, 'PM2.5', ax, plot_error=True, roll=False, agg='mean')
    legend_list = sea_pred.columns.to_list()
    legend_list = ['samples '+ s for s in legend_list]
    ax.legend(legend_list + ['actual'] )
    plt.title('Seasonal Pattern of Actual Data and Inference Bands')
    #ax.set_ylim([0, 170])

    for l, c in color_zip:
        ax.axhline(l, color=c)

    if filename:
        plt.savefig(filename)

