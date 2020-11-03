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
    if len(series) == 0:
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
        except BaseException:
            pass
    try:
        dummies.drop('VAR', axis=1, inplace=True)
    except BaseException:
        pass
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
    if num_lags is None:
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
        surface:(str, float)='sphere'):
    """ Calculate the damped power based on the distance series.

    The damping factor maybe 100/distance or 100/distance**2.

    The hundred is add to increase the magnitude of the columns 

    Args:
        series: series to recalculate
        distance: distance array. Must have the same lenght as the series
        surface(optional): either 'circle' or 'sphere' or power factor 

    Returns:
        new_series

    Examples:
        cal_power_damp(fire['power'], fire['distance'],surface='sphere')

    """
    if surface == 'sphere':
        new_series = series*100 / distance**2 

    elif surface == 'circle':
        new_series = series*100 / distance

    else:
        new_series = series*100 / distance**surface

    return new_series


def cal_arrival_time(
    detection_time: pd.core.series.Series,
    distance: pd.core.series.Series,
    wind_speed: (
        float,
        np.array, pd.core.series.Series) = 2):
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
        int) = 1):
    """ Feature engineer fire data. Account of the distance from the source and time lag using wind speed.
    This function use average wind speed.

    Args:
        fire_df: fire df
        fire_col: fire column to use, either 'power' or 'count'
        damp_surface: damping surface, either 'sphere', or 'cicle'
        shift: row to lag the fire data
        roll: rolling sum factor  

    Return: pd.DataFrame 

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
    fire_df = fire_df.rolling(roll, min_periods=1).sum()
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
        int) = 1):
    """ Separate fire from different distance and take the average. This function use average wind speed. 
    
    Args:
        fire: fire dataframe
        zone_list: a list of distance in km to separate fire feature
        fire_col: fire column to use, either 'power' or 'count'
        damp_surface: damping surface, either 'sphere', or 'cicle'
        shift: row to lag the fire data
        roll: rolling sum factor  
        w_speed: average wind speed in km per hour
    
    Returns: (pd.DataFrame, list)
        new_fire: new fire feature ready to merge with the weather data
        fire_col_list: a list of fire columns 

    """
    fire_col_list = []
    # look for the hotspots very close to the city center, which can create infinity-like value. 
    idxs = fire[fire['distance'] <1].index
    fire.loc[idxs, 'distance'] = 1

    new_fire = pd.DataFrame()
    # weight the fire columns by confidence 
    #fire[fire_col] *= fire['confidence']
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


def dummy_time_of_day(df, col='time_of_day', group_hour=3):
    """One hot encode time_of_day columns. df must have datetime index.

    Args:
        df: dataframe to add the encoded
        col: column name
        group_hour: integer of hour to group the data into this help reduce the number of variables

    Returns: processed dataframe

    """
    # add time of day column
    df[col] = df.index.hour

    if group_hour == 1:
        df[col] = df[col].astype("category")
    else:
        # turn the columns into catergory. Use group_hour to reduce the number
        df[col] = pd.cut(
            df[col],
            bins=np.arange(
                0,
                24 + group_hour,
                group_hour),
            right=False)

    temp = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df, temp], axis=1)
    # drop old columns
    return df.drop(col, axis=1)

def dummy_day_of_week(df, col='day_of_week'):
    """One hot encode day_of_week columns. df must have datetime index.

    Args:
        df: dataframe to add the encoded
        col: column name
         

    Returns: processed dataframe

    """
    # add time of day column
    df[col] = df.index.dayofweek
    temp = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df, temp], axis=1)
    # drop old columns
    return  df.drop(col, axis=1)

def fill_missing_poll(df, limit: int = 6):
    """Fill missing pollution data. Only Work with one columns dataframe

    Args:
        df: hourly pollution data with missing value
        limit(default): number of forward and backward filling

    Returns: new dataframe

    """
    min_date = df.index.min()
    max_date = df.index.max()
    new_index = pd.date_range(start=min_date, end=max_date, freq='h')
    new_poll_df = df.merge(
        pd.DataFrame(
            index=new_index),
        right_index=True,
        left_index=True,
        how='right')
    new_poll_df.index.name = 'datetime'

    new_poll_df_f = new_poll_df.fillna(method='ffill', limit=limit).copy()
    new_poll_df_b = new_poll_df.fillna(method='bfill', limit=limit).copy()

    return pd.concat([new_poll_df_f, new_poll_df_b],
                     axis=0).groupby(level=0).mean()


def add_wea_vec(wea_df:pd.DataFrame, roll_win=6, daily_avg=True)-> pd.DataFrame:
    """Add wind direction vector columns. This is to prepare the weather data for fire feature engineering.
    
    
    Args:
        wea_df: weather dataframe with datetime index 'Wind' and 'Wind Speed(kmph)' columns
        roll_win: rolling average windows 
        daily_avg: if True, calculate the daily average value. 
    Returns: process weather dataframe
    

    """
    # dictionay to map the string direction 
    wind_vec_x_dict = {'N': 0.0, 'NNE': 0.38, 'NE': 0.71, 'ENE': 0.92, 'E': 1.0, 'ESE': 0.92, 'SE': 0.71, 'SSE': 0.38,
                       'S': 0.0, 'SSW': -0.38, 'SW': -0.71, 'WSW': -0.92, 'W': -1.0, 'WNW': -0.92, 'NW': -0.71, 'NNW': -0.38}
    wind_vec_y_dict = {'N': -1.0, 'NNE': -0.92, 'NE': -0.71, 'ENE': -0.38, 'E': 0.0, 'ESE': 0.38, 'SE': 0.71,
                       'SSE': 0.92, 'S': 1.0, 'SSW': 0.92, 'SW': 0.71, 'WSW': 0.38, 'W': 0.0, 'WNW': -0.38, 'NW': -0.71, 'NNW': -0.92}
    
    # keep only wind direction and wind speed columns
    wea_proc = wea_df[['Wind', 'Wind Speed(kmph)']].copy()
    # create a winvector columns
    wea_proc['wind_vec_x'] =  wea_proc['Wind'].map(wind_vec_x_dict)
    wea_proc['wind_vec_y'] =  wea_proc['Wind'].map(wind_vec_y_dict)
    # rolling average smooth abrupt change. 
    wea_proc = wea_proc.rolling(roll_win, min_periods=1).mean()
    if daily_avg:
        # reample to daily average because the fire data is a daily data 
        wea_proc = wea_proc.resample('d').mean().round()
    # normalize wind vector 
    norm_vec = np.linalg.norm(wea_proc[['wind_vec_x', 'wind_vec_y']].values, axis=1)
    wea_proc['wind_vec_x'] = wea_proc['wind_vec_x']/norm_vec
    wea_proc['wind_vec_y'] = wea_proc['wind_vec_y']/norm_vec
    
    # drop the 'Wind' direction columns
    return wea_proc


def cal_wind_damp_row(row, city_x, city_y):
    """Calculate damping factor for each hotspot. This function should be applied to Panda DataFrame.
    
    Round to negative damping factor to zero.
    
    Args:
        row: panda row
        city_x: longitude in km in Mercator coordinate
        city_y: latitude in km in Mercator coordinate
        
    Returns: float 
        a damping factor for that row.
        
    """
    # forming a vector in km unit 
    hot_vec = [(city_x - row['long_km']), (city_y - row['lat_km'])]
    # normalize this vector
    hot_vec = hot_vec/np.linalg.norm(hot_vec)
 
    wea_vec = [row['wind_vec_x'], row['wind_vec_y']]
     
    # round to zero if negative 
    return np.maximum(round(np.dot(hot_vec, wea_vec), 4), 0)

def cal_wind_damp(fire_df, wea_df, city_x, city_y):
    """Calculate the damping due to the wind direction. The new column is named 'winddamp' columns
    
    Args:
        fire_df: hotspots information
        wea_df: weather dataframe 
        city_x: longitude in km in Mercator coordinate
        city_y: latitude in km in Mercator coordinate
        
    Returns:   pd.DataFrame
    
    """
    fire_df['round_time'] = fire_df.index.round('D')
    # obtain process weather dataframe
    wea_proc = add_wea_vec(wea_df)
    # add windspeed and direction to the fire data
    fire_df = fire_df.merge(wea_proc, left_on='round_time', right_index=True, how='left')    
    # calculate the damping factors due to win direction
    fire_df['winddamp'] = fire_df.apply(cal_wind_damp_row, axis=1, args=(city_x, city_y))
    
    #remove unuse columns
    return fire_df.drop(['wind_vec_x', 'wind_vec_y', 'round_time'], axis=1)