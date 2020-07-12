# -*- coding: utf-8 -*-
from ..imports import *


def add_is_holiday(
        df,
        holiday_file='C:/Users/Benny/Documents/Fern/aqi_thailand2/data/th_holiday.csv'):
    """ add is_holiday columns. df must have 'datetime' columns

    """
    # prepare datetime columns

    try:

        df['datetime'] = pd.to_datetime(df['datetime'])

    except BaseException:
        df.index.name = 'datetime'
        df = df.reset_index()
        df['datetime'] = pd.to_datetime(df['datetime'])

    # load holiday files
    if os.path.exists(holiday_file):
        holiday = pd.read_csv(holiday_file)
        holiday['date'] = pd.to_datetime(holiday['date'])
        # keep only national holiday
        holiday = holiday[holiday['type'].isin(
            ['National holiday', 'Joint Holiday', 'Public Holiday'])]

    else:
        raise AssertionError('the holiday file does not exist')

    df['date'] = pd.to_datetime(df['datetime'].dt.date)
    holiday['date'] = pd.to_datetime(holiday['date'])
    df = df.merge(holiday[['date', 'name']], on='date', how='left')
    df = df.drop('date', axis=1)
    df['is_holiday'] = ~df['name'].isna() * 1
    return df.drop('name', axis=1).set_index('datetime')


def add_calendar_info(
        df,
        holiday_file='C:/Users/Benny/Documents/Fern/aqi_thailand2/data/th_holiday.csv'):
    """ Add information related to calendar such as holiday, is_weekend, day of week and time of day.

    Args:
        df: data frame with datetime index,
    """
    # add datetime information
    df = add_is_holiday(df, holiday_file)
    # add weekend information
    df['is_weekend'] = df.index.dayofweek.isin([5, 6]) * 1
    # add day of week
    df['day_of_week'] = df.index.dayofweek
    df['time_of_day'] = df.index.hour

    return df


def wind_to_dummies(series):
    """One hot encode wind direction columns and group major wind direction
    Args:
        series: wind data series from weather['Wind']
    
    Raises:
        AssertionError: if passed an empty series
    """
    if len(series)==0:
        raise AssertionError('empty series')

    series = series.astype('category')
    dummies = pd.get_dummies(series)
    
    # group the wind direction into major wind direction
    direction_to_collpse = dummies.columns.to_list()
    # remove 'CALM'
    if 'CALM' in direction_to_collpse:
        direction_to_collpse.remove('CALM')

    if 'VAR' in direction_to_collpse:
        direction_to_collpse.remove('VAR')

    for direction in direction_to_collpse:
        if len(direction) > 1:
            for char in set(direction):
                dummies[char] = dummies[char] + dummies[direction]
            dummies.drop(direction, axis=1, inplace=True)

    # group the 'var' direction
    major_direction = ['E', 'N', 'S', 'W']
    for direction in major_direction:
        try:
            dummies[direction] = dummies[direction] + dummies['VAR']
        except:
            pass
    dummies.drop('VAR', axis=1, inplace=True)
    dummies.columns = ['wind_' + s for s in dummies.columns]
    return dummies


def add_is_rain(
    df,
    rain_list=[
        'Rain',
        'Shower',
        'Thunder',
        'Strom',
        'Drizzle']):
    df['is_rain'] = df['Condition'].str.contains('|'.join(rain_list)) * 1
    df = df.drop('Condition', axis=1)
    return df


def find_num_lag(poll_series, thres=0.5):
    """ Calculate the numbers of partial autocorrelation lag to add as feature to a time series.

    """
    pac = pacf(poll_series)
    # find the number of lag
    idxs = np.where(pac >= 0.5)[0]
    return idxs[1:]


def add_lags(data, pollutant, num_lags=None):
    """Add lags columns to x_data.

    """
    # calculate num lags
    if num_lags==None:
        num_lags = find_num_lag(data[pollutant])
    
    for idx in num_lags:
        lag_name = f'{pollutant}_lag_{idx}'
        lag_series = data[pollutant].shift(idx)
        lag_series.name = lag_name
        # add to data
        data = pd.concat([data, lag_series], axis=1)
    data = data.dropna()
    return data

# function for feature eng fire


def cal_power_damp(
        series: pd.core.series.Series,
        distance: pd.core.series.Series,
        surface='sphere'):
    """ Calculate the damped power based on the distance series.

    The damping factor maybe 1/distance or 1/distance**2.
    Args:
        series: series to recalculate
        distance: distance array. Must have the same lenght as the series
        surface(optional): either 'circle' or 'sphere'

    Returns:
        new_series

    Examples:
        cal_power_damp(fire['power'], fire['distance'],surface='sphere')

    """
    if surface == 'sphere':
        new_series = series / distance**2

    elif surface == 'circle':
        new_series = series / distance

    return new_series


def cal_arrival_time(
    detection_time: pd.core.series.Series,
    distance: pd.core.series.Series,
    wind_speed: (
        float,
        np.array) = 2):
    """ Calculate the approximate time that the pollution arrived at the city using the wind speed and distance from the hotspot.

    Round arrival time to hour

    Args:
        detection_time: datetime series
        distance: distance series in km
        wind_speed(optional): approximate wind speed, can be floar or array in km/hour

    Returns:
        arrival_time: datetime series of arrival time

    """
    arrival_time = detection_time + pd.to_timedelta(distance / wind_speed, 'h')
    return arrival_time.dt.round('H')


def shift_fire(
    fire_df: pd.core.frame.DataFrame,
    fire_col: str = 'power',
    damp_surface: str = 'sphere',
    shift: int = 0,
    roll: int = 48,
    w_speed: (
        float,
        int) = 8):
    """ Feature engineer fire data. Account of the distance from the source and time lag using wind speed.

    Args:
        fire_df:
        fire_col
        damp_surface
        shift
        roll

    """
    require_cols = ['distance', fire_col]
    if fire_df.columns.isin(require_cols).sum() > len(require_cols):
        raise AssertionError(
            'missing required columns for feature engineering fire data')

    # calculate the damping factors
    fire_df['damp_' + fire_col] = cal_power_damp(
        fire_df[fire_col], fire_df['distance'], surface=damp_surface)
    # calculate particle arrival time
    fire_df['arrival_time'] = cal_arrival_time(
        detection_time=fire_df.index,
        distance=fire_df['distance'],
        wind_speed=w_speed)
    fire_df = fire_df.set_index('arrival_time')
    fire_df = fire_df.resample('h').sum()['damp_' + fire_col]
    fire_df = fire_df.rolling(roll).sum()
    fire_df = fire_df.shift(shift)
    fire_df.index.name = 'datetime'
    return fire_df


def get_fire_feature(
    fire,
    zone_list=[
        0,
        100,
        200,
        400,
        800,
        1000],
        fire_col: str = 'power',
        damp_surface: str = 'sphere',
        shift: int = 0,
        roll: int = 48,
        w_speed: (
            float,
        int) = 8):
    """ Separate fire from different distance

    """
    fire_col_list = []
    new_fire = pd.DataFrame()
    for start, stop in zip(zone_list, zone_list[1:]):
        col_name = f'fire_{start}_{stop}'

        fire_col_list.append(col_name)
        # select sub-data baseline the distance
        fire_s = fire[(fire['distance'] < stop) & (
            fire['distance'] >= start)][[fire_col, 'distance']].copy()
        fire_s = shift_fire(
            fire_s,
            fire_col=fire_col,
            damp_surface=damp_surface,
            shift=shift,
            roll=roll,
            w_speed=w_speed)
        fire_s.name = col_name
        new_fire = pd.concat([new_fire, fire_s], axis=1, ignore_index=False)

    new_fire = new_fire.fillna(0)
    new_fire.index.name = 'datetime'
    return new_fire, fire_col_list


def sep_fire_zone(fire, fire_col, zone_list=[0, 100, 200, 400, 800, 1000]):
    """ Separate fire data into zone mark by a distance in the zone_list without perform feature enginering.
    Use for data visualization

    Args:
        fire: fire dataframe
        fire_col: 'power' or 'count'
        zone_list:

    Return:
        new_fire: a dataframe with each column, a fire data in that zone
        fire_col_list: a list of column name

    """
    fire_col_list = []
    new_fire = pd.DataFrame()
    for start, stop in zip(zone_list, zone_list[1:]):
        col_name = f'fire_{start}_{stop}'
        fire_col_list.append(col_name)
        # select sub-data baseline the distance
        fire_s = fire[(fire['distance'] < stop) & (
            fire['distance'] >= start)][[fire_col]].copy()
        fire_s.columns = [col_name]
        fire_s = fire_s.resample('h').sum()
        new_fire = pd.concat([new_fire, fire_s], axis=1, ignore_index=False)

    return new_fire, fire_col_list
