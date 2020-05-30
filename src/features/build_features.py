# -*- coding: utf-8 -*-
from ..imports import *

def add_is_holiday(df,holiday_file='C:/Users/Benny/Documents/Fern/aqi_thailand2/data/th_holiday.csv'):
    """ add is_holiday columns. df must have 'datetime' columns
    
    """
    # prepare datetime columns

    try: 
        
        df['datetime'] = pd.to_datetime(df['datetime'])
    
    except: 
        df = df.reset_index() 
        df['datetime'] = pd.to_datetime(df['datetime'])
        
    # load holiday files
    if os.path.exists(holiday_file):
        holiday = pd.read_csv(holiday_file)
        holiday['date'] = pd.to_datetime(holiday['date'])
        # keep only national holiday
        holiday = holiday[holiday['type']=='National holiday']
    
    else:
        raise AssertionError('the holiday file does not exist')

    df['date'] = pd.to_datetime(df['datetime'].dt.date)
    holiday['date'] = pd.to_datetime(holiday['date'])
    df = df.merge(holiday[['date','name']], on='date',how='left')
    df['is_holiday'] = ~df['name'].isna()*1
    return df.drop('name',axis=1).set_index('datetime')