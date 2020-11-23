# -*- coding: utf-8 -*-
import sys
import os
import json
import numpy as np
import pandas as pd
from glob import glob

def read_b_data(filename):
    """Read Berkeley earth .txt file data. Return a dataframe and city information.

    Create a datetime column with local timezone.

    """
    # read data file
    data_df = pd.read_csv(filename, sep='\t',
                          header=None, skiprows=10)

    # inspecting the top of the files to get the timezone
    with open(filename, 'r') as f:
        city_info = {}
        for i in range(9):
            line = f.readline()
            # remove %
            line = line.replace('% ', '')
            line = line.replace('\n', '')
            k, v = line.split(': ')
            city_info[k] = v
    time_zone = city_info['Time Zone']
    # assemble datetime column
    data_df['datetime'] = pd.to_datetime(
        {'year': data_df[0], 'month': data_df[1], 'day': data_df[2], 'hour': data_df[3]})
    # convert to Bangkok time zone and remove the time zone information
    data_df['datetime'] = data_df['datetime'].dt.tz_localize(
        'UTC').dt.tz_convert(time_zone)
    data_df['datetime'] = data_df['datetime'].dt.tz_localize(None)
    # drop Year, month, day, UTC hours, PM10_mask columns
    data_df = data_df.drop([0, 1, 2, 3, 5, 6], axis=1)
    data_df.columns = ['PM2.5', 'datetime']

    # inspecting the top of the files
    with open(filename, 'r') as f:
        city_info = {}
        for i in range(9):
            line = f.readline()
            # remove %
            line = line.replace('% ', '')
            line = line.replace('\n', '')
            k, v = line.split(': ')
            city_info[k] = v

    return data_df, city_info


def build_us_em_data(city_name: str, data_folder: str = '../data/us_emb/'):
    """Combine the pollution data from US Embassy monitoring station for the city. Return a list of pollution dataframe.

    """
     
    if city_name == 'Jakarta':
        name_list = ['JakartaCentral', 'JakartaSouth']
    elif city_name == 'Rangoon':
        name_list = ['Rangoon']
    elif city_name == 'Vientiane':
        name_list = ['Vientiane']
    elif city_name =='Ho Chi Minh City':
        name_list = ['HoChiMinhCity']
    elif city_name =='JakartaCentral':
        name_list = ['JakartaCentral']
    elif city_name == 'JakartaSouth':
        name_list = ['JakartaSouth']
    elif city_name =='Hanoi':
        name_list = ['Hanoi']
    else:
        raise AssertionError(f'no data for {city_name}')

    data_list = []

    for name in name_list:
        files = glob(f'{data_folder}{name}_*.csv')
    
        data = pd.DataFrame()
        # concatenate all data
        for file in files:
            df = pd.read_csv(file, na_values=[-999])
            df.columns = df.columns.str.replace('Raw Conc.', 'Value')
            #remove negative number 
            idxs = df[df['Value'] < 0].index
            df.loc[idxs, 'Value'] = np.nan
            data = pd.concat([data, df])
        # format the data
        data['Parameter'] = data['Parameter'].str.split(' - ', expand=True)[0]
        data['datetime'] = pd.to_datetime(data['Date (LT)'])
        data = data.sort_values('datetime')
        data = data.drop_duplicates(['datetime','Parameter'])
        data = data.pivot(
            columns='Parameter',
            values='Value',
            index='datetime').reset_index()
        data = data.dropna(how='all')

        # remove outlier from the data 
        data = data.set_index('datetime')
        for col in data.columns:
            q_high = data[col].quantile(0.9999)
            if city_name == 'Jakarta':
                q_high = 300
            idxs = data[data[col] > q_high].index
            data.loc[idxs, col] = np.nan
        data = data.reset_index()
        
        data_list.append(data)

    return data_list


def read_his_xl(filename):
    # read air4thai historical data
    xl = pd.ExcelFile(filename)
    station_data = pd.DataFrame()

    for sheet_name in xl.sheet_names:
        data = xl.parse(sheet_name, skiprows=[1])

        if len(data) > 0:
            data = parse_1xl_sheet(data)
            station_data = pd.concat([station_data, data], ignore_index=True)
            station_data = convert_pollution_2_number(station_data)

    return station_data.set_index('datetime').dropna(axis=0, how='all')


def isnumber(x):
    # if the data is number
    try:
        float(x)
        return True
    except BaseException:
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
        data_point = '2000' + '0' + data_point

    elif len(data_point) == 4:
        data_point = '2000' + data_point

    elif len(data_point) == 5:
        data_point = '200' + data_point

    elif len(data_point) == 6:
        if '9' == data_point[0]:
            data_point = '19' + data_point
        else:
            data_point = '20' + data_point

    return data_point


def convert_hour(data_point):
    # apply to the hour column in the data to prepare for making datetime column
    # shift by 1 hour to get rid of 2400
    data_point = int(data_point - 100)
    # convert datatype to string
    data_point = str(data_point)

    if len(data_point) == 3:
        data_point = '0' + data_point

    data_point = data_point[:2]

    # if data_point=='24':
    #    data_point ='00'

    return data_point


def make_datetime_from_xl(data_df):
    # drop nan value
    data_df = data_df[~data_df[['date', 'hour']].isna().any(axis=1)].copy()
    data_df['date'] = data_df['date'].astype(int)
    data_df['hour'] = data_df['hour'].astype(int)
    # preprocess date and hour columns
    data_df['date'] = data_df['date'].apply(convert_year)
    data_df['hour'] = data_df['hour'].apply(convert_hour)
    data_df['datetime'] = data_df['date'] + '-' + data_df['hour']
    data_df['datetime'] = pd.to_datetime(
        data_df['datetime'], format='%Y%m%d-%H')

    # drop old columns
    data_df.drop('date', axis=1, inplace=True)
    data_df.drop('hour', axis=1, inplace=True)

    return data_df


def parse_1xl_sheet(data_df):

    # change column name
    data_df.columns = data_df.columns.str.rstrip()
    data_df.columns = data_df.columns.str.lstrip()
    data_df.columns = data_df.columns.str.replace('ปี/เดือน/วัน', 'date')
    data_df.columns = data_df.columns.str.replace('วัน/เดือน/ปี', 'date')
    if 'date.1' in data_df.columns:
        data_df['date'] = data_df['date'].fillna(data_df['date.1'])
        data_df = data_df.drop('date.1', axis=1)
    data_df.columns = data_df.columns.str.replace('ชั่วโมง', 'hour')
    to_drops = data_df.columns[data_df.columns.str.contains('Unnamed')]

    # drop nan value
    data_df = data_df[~data_df[['date', 'hour']].isna().any(axis=1)].copy()
    data_df[['date', 'hour']] = data_df[['date', 'hour']].astype(int)
    if len(data_df) > 0:
        # preprocess date and hour columns to create datetime columns
        data_df = make_datetime_from_xl(data_df)
        data_df.drop(to_drops, axis=1, inplace=True)
    else:
        data_df = pd.DataFrame()

    return data_df

def get_th_stations(city_name: str, data_folder:str='../data/aqm_hourly2/'):
    """Look for all polltuions station in the city by Thailand PCD.
    
    Arg:
        city_name: name of the city
        data_folder: location of the staion information json file
    
    Returns: (list, list)
        station_ids: a list of station ids
        station_info_list: a list of station information dictionary 

    """
    # load stations information for air4 Thai
    station_info_file = data_folder + \
        'stations_locations.json'
    with open(station_info_file, 'r', encoding="utf8") as f:
        station_info = json.load(f)

    station_info = station_info['stations']

    # find stations in that city and add to a list
    station_ids = []
    station_info_list = []

    for i, stations in enumerate(station_info):
        if city_name in stations['areaEN']:
            station_ids.append(stations['stationID'])
            station_info_list.append(stations)

    return station_ids, station_info_list

def read_cmucdc(filename:str)->pd.DataFrame:
    """Read Chiang Mai University pollution data. Rename the columns and rop other columns. 
    
    Args:
        filename: filename string
    
    Returns: pd.DataFrame
        data dataframe
        
    """
    try:
        # read data
        df = pd.read_csv(filename)
        # replace the pollution name with our convention
        col_dict = {'log_datetime':'datetime',
            'pm10':'PM10',
            'pm25':'PM2.5'}
        # rename the columns
        df = df.rename(columns=col_dict)
        df['datetime'] = pd.to_datetime(df['datetime'])
        
    except:
        # exmpty data return an empty dataframe 
        df = pd.DataFrame(columns = ['datetime','PM10', 'PM2.5'])

    # drop temperature and humidity columns 
    cols = [ col for col in df.columns if col in ['datetime','PM10', 'PM2.5']]
    
    return df[cols]


def collect_vt_station(data_folder, city_name):
    """Extract all data from Vietnamese EPA stations for the specified city_name and average the data for that city  

    Note that each pollutant is recorded in AQI unit and has to be convert to the raw concentration first 
    Args:
        data_folder: folder with all scraped datafile 
        city_name: city_name to choose

    Return: a dataframe
        a average avalue for each pollutant   

    """
    # extract all data from Vietnamese EPA 
    dfs = glob('../data/vn_epa/*.csv')
    city_df = []
    for df in dfs:
        df = pd.read_csv(df, na_values='-')
        # keep only hanoi data
        df = df[df['city'] == city_name]
        df['datetime'] = pd.to_datetime(df['datetime'])
        city_df.append(df)

    city_df = pd.concat(city_df, ignore_index=True)
    # drop duplicates data
    city_df = city_df.drop_duplicates()
    city_df = hanoi_df.drop(['VN_AQI', 'city','station'], axis=1)
    city_df = city_df.sort_values('datetime')
    # taking average of all the station in hanoi
    return city_df.groupby('datetime', as_index=False).mean().round()