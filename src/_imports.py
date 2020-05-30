# -*- coding: utf-8 -*
"""For both EBM Regressor and EBM Band, import external packages

"""
# For both EBM Regressor and EBM Band
import json
import logging
import os
from copy import deepcopy
from collections import defaultdict
import warnings
import numpy as np
import pandas as pd
# state package for health score
from scipy.stats.mstats import winsorize
from scipy.stats import lognorm
import joblib
from tqdm import tqdm
from itertools import product
from ast import literal_eval
from glob import glob

# for hyperparameter tune
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_squared_log_error, explained_variance_score

# For EBM Regressor only
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split, KFold
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
# knn only
from sklearn.neighbors import KNeighborsRegressor
# gauss only
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

# nn only
from keras.models import load_model, clone_model, Sequential
from keras.layers import Dense, Input, Dropout, BatchNormalization
from keras.optimizers import Adam
import keras.backend as K
import tensorflow as tf
from keras.callbacks import EarlyStopping

# For EBM Band only
from scipy.signal import resample
from sklearn.cluster import AffinityPropagation
from sklearn.metrics import pairwise_distances

# For EBM Curve Fitting only
from numba import njit
from scipy.signal import find_peaks, peak_widths
from scipy.optimize import curve_fit
