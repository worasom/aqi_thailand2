from .imports import *
""" Unit conversion function

"""


def merc_x(lon):
    # convert logitude in degree to mercadian in meter
    # Earth radius in meter
    # from https://wiki.openstreetmap.org/wiki/Mercator
    try:
        lon = float(lon)
    except BaseException:
        pass
    r_major = 6378137.000
    return r_major * np.radians(lon)


def merc_y(lat, shift=False):
    # convert latitude in degree to mercadian in meter
    try:
        lat = float(lat)
    except BaseException:
        pass

    if shift:
        # Add correction to latitude
        lat += 0.08

    if lat > 89.5:
        lat = 89.5
    if lat < -89.5:
        lat = -89.5

    r_major = 6378137.000
    r_minor = 6356752.3142
    temp = r_minor / r_major
    eccent = np.sqrt(1 - temp**2)
    phi = np.radians(lat)
    sinphi = np.sin(phi)
    con = eccent * sinphi
    com = eccent / 2
    con = ((1.0 - con) / (1.0 + con))**com
    ts = math.tan((math.pi / 2 - phi) / 2) / con
    y = 0 - r_major * np.log(ts)
    return y


def get_color(
        series: (
            np.array,
            pd.core.series.Series) = None,
    color_length: int = None,
        cmap=cm.Blues):
    """Create a list of hex colormap for a series or for a specified length """
    if series:
        # colormap from a series
        vmin = np.min(series)
        vmax = np.max(series)
    else:
        # colormap for specified lenght
        series = np.arange(color_length)
        vmin = 0
        vmax = np.max(series)
    # normalize item number values to colormap
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    # colormap values = viridis, jet, spectral
    color_list = [cmap(norm(value)) for value in series]
    color_list = [str(cm.colors.to_hex(color)) for color in color_list]
    return color_list


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


def add_season(df, start_month='-12-01', end_month='-04-30'):
    # add winter season column
    # df.index must be datetime format sorted in ascending order
    df = df.sort_index()
    df['year'] = df.index.year
    df['season'] = 'other'
    for year in df.year.unique():
        start_date = str(year) + start_month
        end_date = str(year + 1) + end_month
        label = 'winter_' + str(year)

        df.loc[start_date:end_date, 'season'] = label

    return df

def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

        t=linspace(-2,2,0.1)
        x=sin(t)+randn(len(t))*0.1
        y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.

    """

    if window_len<3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y[int(window_len/2-1):-int(window_len/2)-1] 


