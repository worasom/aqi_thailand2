# -*- coding: utf-8 -*-
from ..imports import *

def add_is_holiday(df,holiday_file='C:/Users/Benny/Documents/Fern/aqi_thailand2/data/th_holiday.csv'):
    """ add is_holiday columns. df must have 'datetime' columns
    
    """
    # prepare datetime columns

    try: 
        
        df['datetime'] = pd.to_datetime(df['datetime'])
    
    except: 
        df.index.name = 'datetime'
        df = df.reset_index() 
        df['datetime'] = pd.to_datetime(df['datetime'])
        
    # load holiday files
    if os.path.exists(holiday_file):
        holiday = pd.read_csv(holiday_file)
        holiday['date'] = pd.to_datetime(holiday['date'])
        # keep only national holiday
        holiday = holiday[holiday['type'].isin(['National holiday','Joint Holiday','Public Holiday'])]
    
    else:
        raise AssertionError('the holiday file does not exist')

    df['date'] = pd.to_datetime(df['datetime'].dt.date)
    holiday['date'] = pd.to_datetime(holiday['date'])
    df = df.merge(holiday[['date','name']], on='date',how='left')
    df = df.drop('date',axis=1)
    df['is_holiday'] = ~df['name'].isna()*1
    return df.drop('name',axis=1).set_index('datetime')

def add_calendar_info(df, holiday_file='C:/Users/Benny/Documents/Fern/aqi_thailand2/data/th_holiday.csv'):
    """ Add information related to calendar such as holiday, is_weekend, day of week and time of day. 
    
    Args:
        df: data frame with datetime index, 
    """
    # add datetime information 
    df = add_is_holiday(df, holiday_file)
    # add weekend information 
    df['is_weekend'] = df.index.dayofweek.isin([5,6])*1
    # add day of week
    df['day_of_week'] = df.index.dayofweek
    df['time_of_day'] = df.index.hour
    
    return df
    
def wind_to_dummies(series):
    """One hot encode wind direction columns and group major wind direction
    
    """
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
        if len(direction)>1:
            for char in set(direction):
                dummies[char] = dummies[char] + dummies[direction]
            dummies.drop(direction, axis=1,inplace=True)

    # group the 'var' direction
    major_direction = ['E','N','S','W']
    for direction in major_direction:
        dummies[direction] = dummies[direction] + dummies['VAR']
    dummies.drop('VAR',axis=1,inplace=True)
    dummies.columns = ['wind_' + s for s in dummies.columns ]
    return dummies

def add_is_rain(df,rain_list=['Rain','Shower','Thunder','Strom','Drizzle']):
    df['is_rain'] = df['Condition'].str.contains('|'.join(rain_list))*1
    df = df.drop('Condition', axis=1)
    return df


def find_num_lag(poll_series, thres=0.5):
    """ Calculate the numbers of partial autocorrelation lag to add as feature to a time series. 
    
    """

    pac = pacf(poll_series)
    # find the number of lag 
    idxs = np.where(pac >= 0.5)[0]
    return idxs[1:]

def add_lags(data, pollutant):
    """Add lags columns to x_data.

    """
    # calculate num lags
    num_lags = find_num_lag(data[pollutant])
    for idx in num_lags:
        lag_name = f'{pollutant}_lag_{idx}'
        lag_series = data[pollutant].shift(idx) 
        lag_series.name = lag_name
        # add to data 
        data = pd.concat([data, lag_series], axis=1) 
    data = data.dropna()
    return data


