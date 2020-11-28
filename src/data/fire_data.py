# -*- coding: utf-8 -*-
import os
import sys


if __package__: 
    from ..imports import *
    from ..gen_functions import *
else:
    # import anything in the upper directory 
    _i = os.path.dirname(os.path.dirname(os.path.abspath("..")))
    if _i not in sys.path:
        sys.path.insert(0, _i)
    from imports import *
    from gen_functions import *

""" Functions for proess hospots data.

"""
 
def add_merc_to_fire(fire_folder='../data/fire_map/world_2000-2020/', instr='MODIS', chunk=1E6, long_range=[81,150], lat_range=[-19, 98]):
    """Add mercator coordinate to all the fire files in the folder and save as another folder

    """

    if instr =="MODIS":
        from_folder = fire_folder + 'M6/'
        to_folder = fire_folder + 'M6_proc/'
    elif instr == "VIIRS":
        from_folder = fire_folder + 'V1/'
        to_folder = fire_folder + 'V1_proc/'
    else:
        raise AssertionError(
                'instrument name can be either MODIS or VIIRS')

    if not os.path.exists(to_folder):
        os.mkdir(to_folder)
         
    files = glob(from_folder + '*.csv')
    
    for file in tqdm(files):
        #print(f'working with {file}')
        file = file.replace('\\', '/')
        new_filename = to_folder + file.split('/')[-1] 
         
        if not os.path.exists(new_filename):
            for df in pd.read_csv(file, chunksize=chunk):
                # keep only the data in range
                df = df[(df['longitude'] <= long_range[1]) & (df['longitude'] >= long_range[0])]
                df = df[(df['latitude'] <= lat_range[1]) & (df['latitude'] >= lat_range[0])]
                df = add_merc_col(df, lat_col='latitude', long_col='longitude', unit='km')
                if os.path.exists(new_filename):
                    df.to_csv(new_filename, mode='a', header=None, index=False)
                else:
                    df.to_csv(new_filename, index=False)
                
            #print(f'save as {new_filename}')

        #else:
            
            #print(f'skip file {file}')
        

def read_fire(
    file: str,
    lat_km: float,
    long_km: float,
    distance: (
        int,
        float) = 1000) -> pd.core.frame.DataFrame:
    """Read fire data and keep the data within distance
    Args:
        file: fire data csv filename
        lat_km: latitude of the city center in km
        long_km: longitude of the city center in km
        distance(optional): distance in km from the city center for keeping the data

    Returns: dataframe of fire data

    """

    f = pd.read_csv(file)

    # convert lat
    #f['lat_km'] = (f['latitude'].apply(merc_y) / 1E3).round().astype(int)
    #f['long_km'] = (merc_x(f['longitude']) / 1E3).round().astype(int)

    # if 'lat_km' not in f.columns: 
    #     f = add_merc_col(f, lat_col='latitude', long_col='longitude', unit='km')
    #     f.to_csv(file, index=False)
        
    #f['lat_km'] = (f['lat_m']/1000).astype(int)
    #f['long_km'] = (f['long_m']/1000).astype(int)

    # add distance columns
    f['distance'] = np.sqrt((f['lat_km'] - lat_km)** 2 + ((f['long_km'] -long_km)**2))
    # remove by distance 
    f = f[f['distance'] <= distance]

    return f


def add_datetime_fire(fire):
    # add datetime conlumns to hotspot data
    # assemble datetime column \
    fire['datetime'] = fire['acq_date'] + ' ' + fire['acq_time'].astype(str)
    fire['datetime'] = pd.to_datetime(
        fire['datetime'], format='%Y-%m-%d %H%M', utc=True)

    # convert to Bangkok time zone and remove time zone information
    fire['datetime'] = fire['datetime'].dt.tz_convert('Asia/Bangkok')
    fire['datetime'] = fire['datetime'].dt.tz_localize(None)
    fire = fire.sort_values('datetime')

    return fire


def process_fire_data(filename=None, fire=None, and_save=False):
    """ Add datetime,  drop duplicate data and remove uncessary columns.

    """
    if filename:
        fire = pd.read_csv(filename)

    # add datetime
    fire = add_datetime_fire(fire)

    # drop duplicate data
    print('before drop', fire.shape)
    # sort values by brightness
    try:
        # for MODIS file
        fire = fire.sort_values(
            ['datetime', 'lat_km', 'long_km', 'brightness'], ascending=False)
    except BaseException:
        # for VIIRS
        fire = fire.sort_values(
            ['datetime', 'lat_km', 'long_km', 'bright_ti4'], ascending=False)

    fire = fire.drop_duplicates(['datetime', 'lat_km', 'long_km'])

    # drop unncessary columns
    try:
        columns_to_drop = [
            'acq_date',
            'satellite',
            'instrument',
            'version',
            'daynight',
            'bright_t31',
            'type']
        fire = fire.drop(columns_to_drop, axis=1)
    except BaseException:
        columns_to_drop = [
            'acq_date',
            'satellite',
            'instrument',
            'version',
            'daynight',
            'bright_ti5',
            'type']
        fire = fire.drop(columns_to_drop, axis=1)

    fire = fire.sort_values('datetime')
    fire = fire.set_index('datetime')
    # remove the data before '2002-07-04' because there is only one satellite
    fire = fire.loc['2002-07-04':]

    print('after drop', fire.shape)

    if and_save:
        fire.to_csv(filename, index=False)
    else:
        return fire


def cal_repeat_spots(
        df,
        repeat_list=[
            2,
            3,
            5,
            10],
    group_list=[
            'year',
            'lat_km',
            'long_km'],
        accum=False,
        start_month='-12-01',
        end_month='-04-30'):
    """Calculate % of hotspots repeat withint the same year or different year
    Args:
        df: fire data series with datetime index
        repeat_list: number of repetition
        group_list: data to be group by if use ['year','lat_km','long_km'] mean looking at the data withint the same year

    """
    df = add_season(df, start_month, end_month)
    # remove the data from other season
    df = df[df['season'] != 'other']

    # total unique hotspot
    total_loc = df['coor'].nunique()

    year_rep = df.groupby(group_list, as_index=False).sum()['count']

    summary = {}
    for rep in repeat_list:
        if accum:
            repeat_per = int(len(year_rep[year_rep >= rep]) / total_loc * 100)
        else:
            repeat_per = int(len(year_rep[year_rep == rep]) / total_loc * 100)
        summary[rep] = repeat_per

    return pd.Series(summary)
