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


def read_his_xl(filename):
    # read air4thai historical data
    xl = pd.ExcelFile(filename)
    station_data = pd.DataFrame()
    
    for sheet_name in xl.sheet_names:
        data = xl.parse(sheet_name,skiprows=[1])
     
        if len(data)> 0:
            data = parse_1xl_sheet(data)
            station_data = pd.concat([station_data,data],ignore_index=True)
            station_data = convert_pollution_2_number(station_data)
        
    return station_data

def isnumber(x):
    # if the data is number
    try:
        float(x)
        return True
    except:
        return False
    
def convert_to_float(s):
    """Convert the data in a series to float
    
    """
    # remove non-numeric data
    s = s[s.apply(isnumber)]
    return s.astype(float)

def convert_to_int(s):
    """Convert the data in a series to int
    
    """
    # remove non-numeric data
    s = s[s.apply(isnumber)]
    return s.astype(int)

def convert_pollution_2_number(data_df):
    
    # convert all pollution data to int or float
    pollution_cols = data_df.columns.to_list() 
    pollution_cols.remove('datetime')
    # convert data for all pollution column
    for col in pollution_cols:
        s = data_df[col].copy()
        data_df[col] = convert_to_float(s)
                
    return data_df

def convert_year(data_point):
    # apply to the date column in the data to prepare for making datetime column
    # convert datatype to string
    data_point = str(data_point)
    
    if len(data_point) == 3:
        data_point = '2000'+'0'+ data_point
        
    elif len(data_point) == 4:
        data_point = '2000'+ data_point
    
    elif len(data_point) == 5:
        data_point = '200'+ data_point
        
    elif len(data_point) == 6:
        data_point = '20'+ data_point
    
    return data_point


def convert_hour(data_point):
    # apply to the hour column in the data to prepare for making datetime column
    # convert datatype to string
    data_point = str(data_point)
    
    if len(data_point) == 3:
        data_point = '0'+ data_point
        
    data_point = data_point[:2]
    
    if data_point=='24':
        data_point ='00'
    
    return data_point

def make_datetime_from_xl(data_df):
    # preprocess date and hour columns
    data_df['date'] = data_df['date'].apply(convert_year)
    data_df['hour'] = data_df['hour'].apply(convert_hour)
    data_df['datetime'] = data_df['date'] + '-' + data_df['hour']
    data_df['datetime'] = pd.to_datetime(data_df['datetime'], format='%Y%m%d-%H')
    
    # drop old columns
    data_df.drop('date',axis=1,inplace=True)
    data_df.drop('hour',axis=1,inplace=True)
    return data_df

def parse_1xl_sheet(data_df):
    
    # change column name
    data_df.columns = data_df.columns.str.rstrip()
    data_df.columns = data_df.columns.str.lstrip()
    data_df.columns = data_df.columns.str.replace('ปี/เดือน/วัน','date')
    data_df.columns = data_df.columns.str.replace('ชั่วโมง','hour')
    to_drops = data_df.columns[data_df.columns.str.contains('Unnamed')]
    # preprocess date and hour columns to create datetime columns
    data_df = make_datetime_from_xl(data_df)
    for col in to_drops:
        data_df.drop(col,axis=1, inplace=True)
        
    return data_df