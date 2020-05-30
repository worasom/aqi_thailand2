# -*- coding: utf-8 -*-
from .imports import *

"""For Both EBM Regressor and EBM Band, general utility functions for calculate health score, error score, and select subcylinder.

"""


def smooth(x, window_len: int = 11, window: str = 'hanning') -> np.array:
    """Smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    The output signal shape is larger than the original signal by window_len-1;
    therefore it is trimmed at both end before returning.

    Args:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    Returns:
        np.array: the smoothed signal

    Examples:
        t=linspace(-2,2,0.1)
        x=sin(t)+randn(len(t))*0.1
        y=smooth(x)


    From: https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html

    """

    if x.ndim != 1:
        raise AssertionError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise AssertionError(
            "Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if window not in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise AssertionError(
            "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')
    return y[int(window_len / 2):-int(window_len / 2)]


def select_subcylinder(cylinder,
                       door_feature_col: str) -> pd.core.frame.DataFrame:
    """Select subcylinder data with a single door event type.

    Use door_feature_col to select the cylinder dataframe with a single door event type.

    Args:
        door_feature_col: cylinder event type 'door_open'

    Returns:
        pd.core.frame.DataFrame: subcylinder dataframe

    Examples:
        to select the cylinder event associated with FIMS door open,
        select_subcylinder('door_open')

    """

    if door_feature_col == 'door_open':
        # door open events
        subcylinder = cylinder[(cylinder['iD'].isin(
            [32, 91])) & (cylinder['direction'] == 0)]
    elif door_feature_col == 'door_close':
        # door close events
        subcylinder = cylinder[(cylinder['iD'].isin(
            [32, 91])) & (cylinder['direction'] == 1)]
    elif door_feature_col == 'latch_forward':
        # latch forward events
        subcylinder = cylinder[(cylinder['iD'].isin(
            [34, 93])) & (cylinder['direction'] == 1)]
        # latch backward events
    elif door_feature_col == 'latch_backward':
        subcylinder = cylinder[(cylinder['iD'].isin(
            [34, 93])) & (cylinder['direction'] == 0)]
        # latch forward and door open events
    elif door_feature_col == 'latchF_doorO':
        subcylinder = pd.concat([cylinder[(cylinder['iD'].isin([32, 91])) & (cylinder['direction'] == 0)],
                                 cylinder[(cylinder['iD'].isin([34, 93])) & (cylinder['direction'] == 1)]])
        # all door open/close events
    elif door_feature_col == 'door':
        subcylinder = cylinder[cylinder['iD'].isin([32, 91])]
        # all latch forward/backward event
    elif door_feature_col == 'latch':
        subcylinder = cylinder[cylinder['iD'].isin([34, 93])]
        # all door and latch events
    elif door_feature_col == 'latch_door':
        subcylinder = cylinder[cylinder['iD'].isin(
            [32, 91, 34, 93])]
    else:
        # if use unrecognize keywords, return an empty df.
        subcylinder = pd.DataFrame(columns=cylinder.columns)

    return subcylinder


def cal_scores(
        ytrue: np.array,
        ypred: np.array,
        score_list: list = [
            r2_score,
            mean_squared_error],
    header_str: str = 'test_',
        to_print=False):
    """Calculate the prediction score

    Inputs:
        ytrue: 2D numpy array of true sensors data
        ypred: 2D numpy array of predicted data
        score_list(optional): a list of function to calculate score [default: [r2_score,mean_squared_error]]
        header_str(optional): string to add to the result_dict key. Useful for separating test_ and training data [default='test_']
        to_print: print the result to the console or result the dictionary

    Returns: dict
        result_dict: dictionary of the scores

    """

    result_dict = {}

    for score_fun in score_list:
        try:
            result_dict.update(
                {header_str + score_fun.__name__: score_fun(ytrue, ypred)})
        except BaseException:
            result_dict.update(
                {header_str + score_fun.__name__: np.nan})
    if to_print:
        print(result_dict)
    else:
        return result_dict


def similarity_score(
        y_true: np.array,
        y_pred: np.array,
        threshold: float = 0.05) -> float:
    """Prediction Metric Function.  Measure the portion of time the absolute error is below a user defined threshold.

    Threshold default value = 0.05, which is 5*0.01(O2 resolution)
    The smaller the better

    Args:
        y_true: 2D np.array of transformed data
        y_pred: 2D np.array of predicted data
        threshold(optional): deviation threshold [default: 0.01*5]

    Returns: float

    """

    abs_error = np.abs(y_true - y_pred)

    error_logic = abs_error > threshold
    time_lenght = error_logic.shape[1]
    # count the time the error go beyond threshold
    error_logic = error_logic.sum(axis=1)
    # turn into percent
    return np.mean(error_logic / time_lenght)


def avg_corr(y_true: np.array, y_pred: np.array, cor_fun=pearsonr):
    """Prediction Metric Function. Calculate average correlation coefficient

    Args:
        y_true: 2D np.array of transformed data
        y_pred: 2D np.array of predicted data
        cor_fun(optional): correlation function [defaul:pearsonr]

    Returns: float

    """
    corr_list = []
    for x, y in zip(y_true, y_pred):
        co = cor_fun(x, y)[0]
        corr_list.append(co)
    return np.mean(corr_list)


def process_sensor_name(sensor_name: str) -> str:
    """Make sensor name more suitable for a filename

    Sensors name sometimes contains / or ., which is not suitable for a filename.
    #. Remove special charactor.
    #. join space with '-'
    #. keep only the first 5 characters
    #. make lower case

    Args:
        sensor_name: original sensors name

    Returns:
        str: sensor name for filename

    """
    # remove forwardslash
    sensor_name = sensor_name.replace('/', '')
    # remove dot
    sensor_name = sensor_name.replace('.', '')
    # remove space
    sensor_name = sensor_name.replace(' ', '')
    # keep the first 5 character
    sensor_name = sensor_name[:5]
    # make lower case
    return sensor_name.lower()


def load_meta(conf: dict, model_meta=None, check: str = None):
    """Load model meta filename

    Args:
        conf: conf dictionary from reg_trainer or reg_predictor
        check(optional): check if input choices are in meta file if does not raise error.
                        This option is true during training, but False during hyperparameter tuning

    Returns: dict
        dict: conf dictionary with all configuration parameters
        model_meta(optional): [default:None]
        check: check if the input choice keys are in model_meta. For EBM Regressor use 'reg'.
        Use 'band' for EBM Band. Use None hyperparameter tunring process.


    Raises:
        AssertionError: if model meta file does not exist.

    """
    if model_meta is None:
        # model_meta was not passed. Load model meta from file
        # check if model meta file exist, then open it
        model_meta_filename = conf['model_meta_filename']

        if os.path.exists(model_meta_filename):
            with open(model_meta_filename, 'r') as f:
                model_meta = json.load(f)
        else:
            raise AssertionError(f'no model meta file {model_meta_filename}')

    if check == 'reg':
        # check if user intent input keys are in model meta
        if 'in_fc' not in model_meta.keys():
            raise AssertionError(
                f'in_fc choice not in model meta')

        if 'in_latch' not in model_meta.keys():
            raise AssertionError(
                f'in_latch choice not in model meta')

        if 'in_door' not in model_meta.keys():
            raise AssertionError(
                f'in_latch choice not in model meta')

        if 'add2doors' not in model_meta.keys():
            raise AssertionError(
                f'add2doors choice not in model meta')

    elif check == 'band':
        # check if user intent input keys are in model meta
        if 'feature_choice' not in model_meta.keys():
            raise AssertionError(
                f'feature_choice not in model meta')

        if 'door_pulse' not in model_meta.keys():
            raise AssertionError(
                f'door_pulse choice not in model meta')

        if 'door_feature_col' not in model_meta.keys():
            raise AssertionError(
                f'door_feature_col choice not in model meta')

        if 'smooth_window' not in model_meta.keys():
            model_meta['smooth_window'] = None

    # integrate to conf
    conf.update(model_meta)

    return conf


def remove_files(files: list):
    """Remove file in the files list
    Args:
        files: a list of filename to remove

    """

    for filename in files:
        if os.path.exists(filename):
            os.remove(filename)


def cal_running_stat(
        old_mu: float,
        old_sigma: float,
        old_n: int,
        data: float,
        ddof=0):
    """Calculate running means and variance by update the old value with a new one(Welford's algorithm).

    For the first data point, use old_n=0. old_mu and old_s are None. data = new_mu and new_s=0.
    For old_n> 0, use Welford's online runing variance to calculate new_s.

    Args:
        old_mu: previous mean value
        old_sigma: previous standard deviation, which is the square root of variance
        old_n: previous number of data point. Use 0 for the first data point.
        data: a new data point
        ddof(optional): Means Delta Degrees of Freedom. Use 1 for unbiased estimator of the variance of the infinite population. [default:0]

    Returns: (float, float, int)
        new_mu: new mean value
        new_sigma: new standard deviation
        new_n: number of data point after updating the mean

    Raises:
        AssertionError: if old_n < 0

    """
    # cast to int
    old_n = int(old_n)
    # check if old_n > 0
    if old_n < 0:
        raise AssertionError('old_n must be greater than or equal to 0')
    elif old_n is None:
        new_n = 1
    else:
        new_n = old_n + 1

    if new_n == 1:
        # the first data point
        new_mu = data
        new_var = 0

    else:

        # turn sigma into variance
        old_var = old_sigma**2
        new_mu = old_mu + (data - old_mu) / new_n
        # if new_n==2:
        #   old_s = 0
        # else:
        old_s = old_var * (new_n - 1 - ddof)

        new_s = old_s + (data - old_mu) * (data - new_mu)
        new_var = new_s / (new_n - ddof)

    return new_mu, np.sqrt(new_var), new_n



