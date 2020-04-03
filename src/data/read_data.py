# -*- coding: utf-8 -*-
from ..imports import *

def read_b_data(filename):
    """Read Berkeley earth .txt file data. Return a dataframe. Create a datetime column with local timezone.
    
    """
    #read data file
    data_df = pd.read_csv(filename, sep='\t', 
                   header=None, skiprows=10)
    #assemble datetime column 
    data_df['datetime'] = pd.to_datetime({'year': data_df[0], 'month': data_df[1], 'day':data_df[2],'hour': data_df[3]})
    #convert to Bangkok time zone and remove the time zone information
    data_df['datetime'] = data_df['datetime'].dt.tz_localize('UTC').dt.tz_convert('Asia/Bangkok')
    data_df['datetime'] = data_df['datetime'].dt.tz_localize(None)
    # drop Year, month, day, UTC hours, PM10_mask columns
    data_df=data_df.drop([0,1,2,3,5,6],axis=1)
    data_df.columns = ['pm25', 'datetime']
    return data_df 