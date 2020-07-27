# -*- coding: utf-8 -*-
import sys
from pathlib import Path
import re
import os
from tqdm import tqdm, tqdm_notebook
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from textwrap import fill
from math import ceil
import seaborn as sns
from matplotlib import cm
from matplotlib.ticker import MaxNLocator
import matplotlib
from glob import glob
import math
from itertools import combinations, product

# webscraping
import requests
import wget
from bs4 import BeautifulSoup
from selenium import webdriver


import time
from datetime import datetime, date, timedelta


from bokeh.io import output_file, output_notebook, show, reset_output, export_png
from bokeh.models import (
    GMapPlot,
    GMapOptions,
    ColumnDataSource,
    Circle,
    LogColorMapper,
    BasicTicker,
    ColorBar,
    DataRange1d,
    PanTool,
    WheelZoomTool,
    BoxSelectTool,
    CategoricalColorMapper,
    Slider,
    DateRangeSlider,
    DateSlider,
    SingleIntervalTicker,
    LinearAxis,
    Legend,
    LegendItem,
    GeoJSONDataSource)
from bokeh.models.mappers import ColorMapper, LinearColorMapper
from bokeh.palettes import Viridis5
from bokeh.plotting import figure, show, output_file
from bokeh.layouts import widgetbox, row, column, gridplot
from mpl_toolkits.basemap import Basemap
from bokeh.tile_providers import get_provider, Vendors


# machine learning

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split, KFold, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet, Ridge, Lasso, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from scipy.cluster import hierarchy as hc
from tpot import TPOTRegressor


# NN model 

# from keras.callbacks import EarlyStopping
# from keras.models import Sequential
# from keras.layers import Dense, Conv1D, SeparableConv1D, Dropout,BatchNormalization
# import keras.backend as K
# import tensorflow as tf
# from keras.optimizers import Adam
# from keras.models import load_model

# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Flatten
# import keras
# from keras import backend as K
# from keras import optimizers

# from sklearn.preprocessing import MinMaxScaler

# optimization 
from skopt.plots import plot_objective
from skopt import gp_minimize
from skopt.utils import use_named_args
from skopt.space import Real, Categorical, Integer
import joblib
from joblib import Parallel, delayed
import pickle
from dask.distributed import Client

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import pacf
from scipy.stats import spearmanr

import warnings
warnings.filterwarnings("ignore")
plt.rcParams.update({'font.size': 16})
