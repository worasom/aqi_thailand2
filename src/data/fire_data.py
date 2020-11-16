# -*- coding: utf-8 -*-
from ..imports import *
from ..gen_functions import *
""" Functions for proess hospots data.

"""


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
    f = add_merc_col(f, lat_col='latitude', long_col='longitude', unit='m')
    # remove by lat
    f = f[(f['lat_km'] <= (lat_km + distance)) &
          (f['lat_km'] >= (lat_km - distance))]
    # remove by long
    f = f[(f['long_km'] <= (long_km + distance)) &
          (f['long_km'] >= (long_km - distance))]
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
