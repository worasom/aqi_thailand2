# -*- coding: utf-8 -*-
from ..imports import *
""" Proess hospot data

"""


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


def process_fire_data(filename=None,fire=None,and_save=False):
    """ Add datetime and drop duplicate data

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
        fire = fire.sort_values(['datetime', 'lat_km', 'long_km', 'brightness'], ascending=False)
    except:
        # for VIIRS
        fire = fire.sort_values(['datetime','lat_km','long_km','bright_ti4'], ascending=False)

    fire = fire.drop_duplicates(['datetime', 'lat_km', 'long_km'])
    print('after drop', fire.shape)

    if and_save:
        fire.to_csv(filename, index=False)
    else:
        return fire
