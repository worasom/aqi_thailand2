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
    ts = np.tan((np.pi / 2 - phi) / 2) / con
    y = 0 - r_major * np.log(ts)
    return y


def to_latlon(xy):
    """ Convert x and y mercator coordinate to latitude and longtitude
    Args:
        xy

    Return (float,float)

    """
    try:
        y = float(y)
    except BaseException:
        pass

    try:
        x = float(x)
    except BaseException:
        pass

    inProj = Proj('epsg:3395')
    outProj = Proj('epsg:4326')
    return np.array(transform(inProj, outProj, *xy))


def merc_lon(x):
    """Convert x in meter to longitude

    """
    return (x / 6378137.000) / np.pi * 180


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

def get_gas_color_list(gas_list, gas_color_dict=None):
    """Map the gas name to preset color in gas_color_dict. 
    
    Use the same color for the gas to make it consistence
    
    Args:
        gas_list: a name of the gas
        gas_color_dict(optional): color dictionary to map the gas to
    
    Returns: list
        a lit of color value 
    
    """
    if gas_color_dict==None:
        gas_color_dict = {'PM2.5': '#0000ff',
                        'PM10': '#660099',
                         'O3': '#cc0033',
                         'CO': '#cc3300',
                         'NO2': '#669900',
                         'SO2': '#00ff00'}
        
    return [gas_color_dict[gas] if gas in gas_color_dict.keys() else 'royalblue' for gas in gas_list ]
    


def cal_scores(
        ytrue: np.array,
        ypred: np.array,
        score_list: list = [
            r2_score,
            mean_squared_error, mean_absolute_error],
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

    # convert year to seasona year
    df['year'] = df['season'].str.split(
        '_', expand=True)[1].fillna(
        df['year']).astype(int)

    return df


def smooth(x, window_len=11, window='hanning'):
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

    if window_len < 3:
        return x

    if window not in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError(
            "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')
    return y[int(window_len / 2 - 1):-int(window_len / 2) - 1]


def season_avg(df, cols=[], roll=True, agg='max', offset=182):
    """Calculate thea seasonal average.

    Args:
        df: dataframe to calculate the average of
        cols: columns to use for the means
        roll: if True, calculate the rolling average or use daily average
        agg: either 'max' or 'mean'
        offset: date of year offset

    Returns: pd.DataFrame(), dict
        df:  dataframe of the seasonal pattern
        winder_day_dict: dictionary that map dayof year to month-day

    """

    if len(cols) == 0:
        cols = df.columns

    if roll:
        df = df[cols].rolling(24, min_periods=0).agg('mean').copy().dropna()
    else:
        df = df[cols]

    # resample data
    df = df.resample('d').agg(agg).copy()
    df['dayofyear'] = df.index.dayofyear
    df['year'] = df.index.year

    # add winter day by substratcing the first day of july
    winterday = df['dayofyear'] - offset
    # get rid of the negative number
    winter_day_max = winterday.max()
    winterday[winterday < 0] = winterday[winterday < 0] + \
        offset + winter_day_max
    df['winter_day'] = winterday

    # add month-day
    df['month_day'] = df.index.strftime('%m-%d')
    temp = df[['winter_day', 'month_day']].set_index('winter_day')
    temp.index = temp.index.astype(str)
    winter_day_dict = temp.to_dict()['month_day']

    return df, winter_day_dict


def to_aqi(value, pollutant):
    """Convert pollution value to AQI

    Args:
        value: pollution reading
        pollutant: type of pollutant

    Returns: int
        AQI value of the pollutant

    """
    try:
        transition_dict = {
            'PM2.5': [
                0, 12.0, 35.4, 55.4, 150.4, 250.4, 350.4, 500, 1e3], 'PM10': [
                0, 155, 254, 354, 424, 504, 604, 1e3], 'O3': [
                0, 54, 70, 85, 105, 200, 1e3], 'SO2': [
                    0, 75, 185, 304, 504, 604, 1e3], 'NO2': [
                        0, 53, 100, 360, 649, 1249, 2049, 1e3], 'CO': [
                            0, 4.4, 9.4, 12.4, 15.4, 30.4, 40.4, 50.4, 1e3]}

        aqi_list = [0, 50, 100, 150, 200, 300, 400, 500, 999]
        tran = np.array(transition_dict[pollutant])
        idx = np.where(value >= tran)[0][-1]
        if idx == len(tran):
            aqi = aqi_list[-1]
        else:
            lower = tran[idx]
            upper = tran[idx + 1]
            lower_aqi = aqi_list[idx]
            upper_aqi = aqi_list[idx + 1]
            aqi = (upper_aqi - lower_aqi) / (upper - lower) * \
                (value - lower) + lower_aqi
            aqi = int(ceil(aqi))
    except BaseException:
        aqi = np.nan

    return aqi


def get_circle(x_cen, y_cen, r, num_data=100):
    """Create x,y coordinate to form a circle

    Args:
        x_cen
        y_cen
        r
        num_data
    """
    step = 2 * np.pi / num_data
    angle = np.arange(0, 2 * np.pi + step, step)
    x_arr = x_cen + r * np.cos(angle)
    y_arr = y_cen + r * np.sin(angle)
    return np.array([x_arr, y_arr])
