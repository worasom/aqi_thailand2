# -*- coding: utf-8 -*-
import os
import sys
import pandas as pd
import numpy as np
import json
from glob import glob
import logging
from joblib import Parallel

if __package__: 
    from ..imports import *
    from ..gen_functions import *
    from ..data.read_data import *
    from ..data.fire_data import *
    from ..data.weather_data import *
    #from ..visualization.mapper import Mapper
    from .build_features import *
    from .config import set_config
    from .dataset import Dataset

else:
    # import anything in the upper directory 
    _i = os.path.dirname(os.path.dirname(os.path.abspath("..")))
    if _i not in sys.path:
        sys.path.insert(0, _i)
    from imports import *
    from gen_functions import *
    from data.read_data import *
    from data.fire_data import *
    from data.weather_data import *
    #from visualization.mapper import Mapper
    # import files in the same directory
    _i = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _i not in sys.path:
        sys.path.insert(0, _i)

    from build_feaures import *
    from config import set_config
    from dataset import Dataset

"""Pollution Dataset Object of a particular city. This dataset is for imputing PM2.5, so will load all pollutants.

"""
class ProvDataset(Dataset):
    """Pollution Dataset Object of a particular city. This dataset is for imputing PM2.5, so will load all pollutants.

    Only work for Thailand. 

    Args:
        Dataset ([type]): [description]
    """

    
    def __init__(
            self,
            city_name: str,
            main_data_folder: str = '../data/',
            model_folder='../models/', report_folder='../reports/'):
        """Initialize

        #. Check if the city name exist in the database
        #. Setup main, data, and model folders. Add as attribute
        #. Check if the folders exists, if not create the folders
        #. Load city information and add as atribute

        """
        super().__init__(city_name, main_data_folder, model_folder , report_folder)


    # def get_pcd_station(self)->list:
    #     """"Find a list of pcd station id 

    #     Returns:
    #         list: a list of pcd station id
            
    #     """
    #     # init mapper object to take advantage of the station function 
    #     # this is so that we don't write the same function twice 

    #     mapper = Mapper(main_folder=self.main_folder)
    #     pcd_stations = mapper.build_pcd_station()
    #     # obtain a list of PCD station_id
        

    #     return pcd_stations[pcd_stations['City']==self.city_name]['id'].to_list()

    
    def build_prov_pollution(self):
        """Collect all Pollution data from a different sources and take the average. 
        Override the original function in Dataset class.

        Since each city have different data sources. It has to be treat differently. 
        The stations choices is specified by the config.json

        Returns: a list of dataframe each dataframe is the data from all station.

        """   

        # data list contain the dataframe of all pollution data before merging
        # all of this data has 'datetime' as a columns
        data_list = []

        # load data from Berkeley Earth Projects This is the same for all
        # cities
        filename = self.main_folder + 'pm25/' + self.city_name.replace(' ', '_') + '.txt'
        if os.path.exists(filename):
            b_data, _ = read_b_data(filename)
            data_list.append(b_data)

        pcd_stations = self.get_pcd_station() 
        pcd_stations = pcd_stations[pcd_stations['City']==self.city_name] 
        station_ids = pcd_stations['id'].to_list()
        print('station_ids ', station_ids)
        self.build_pollution( station_ids = station_ids)

    def build_all_data(self):

        # self.build_pollution()
        self.build_prov_pollution()
        self.build_weather()
        self.save_()


    def build_feature(self, pollutant: str = 'PM2.5', rolling_win=24, fill_missing=False, dropna=True):
        """Assemble pollution data, datetime and weather data. Omit the fire data for later step.

        #. Call self.load_() to load processed data
        #. Add pollutant as pollutant attribute

        Args:
            pollutant(optional): name of the pollutant [default:'PM2.5']
            rolling_win(optional): rolling windows size [defaul:24]. This does not do anything.
            fill_missing(optional): if True, fill the missing pollution data
            cat_hour(optional): if true, one hot encode the time_of_day column
            group_hour(optiona): hour to reduce the catergory of the time_of_day. This is needed if cat_hour==True
            cat_dayofweek(optional): if true, one hot encode the day_of_week column
            cat_month(optional): if true, add one hot encode the month columns 

        Raises:
            AssertionError: if pollutant not in self.poll_df

        """
        #logger = logging.getLogger(__name__)

        # check if pollutant data exist
        if not hasattr(self, 'poll_df') or not hasattr(self, 'wea'):
            self.load_()

        # check if pollutant data exist
        if pollutant not in self.poll_df.columns:
            raise AssertionError(f'No {pollutant} data')
         
        self.pollutant = pollutant

        if fill_missing:
            self.poll_df = fill_missing_poll(self.poll_df, limit=6)

        # merge pollution and wind data

        if 'datetime' in self.wea.columns:
            self.wea.set_index('datetime', inplace=True)
        
        # select weather cols 
        wea_cols = set(['Temperature(C)', 'Humidity(%)','Dew_Point(C)', 'Wind_Speed(kmph)', 'Wind_Gust(kmph)',
        'Pressure(hPa)', 'Precip.(mm)'])
        wea_cols = wea_cols.intersection(self.wea.columns.to_list())
        wea_cols = list(wea_cols)

        data = self.poll_df.merge(
            self.wea[wea_cols],
            left_index=True,
            right_index=True,
            how='inner')

        # some data need to be crop. This is specified by the crop_dict in the config.py 
         
        if self.city_name in self.crop_dict.keys():
            crop_dict_city = self.crop_dict[self.city_name]
            for pollutant in crop_dict_city.keys():
                start_date = crop_dict_city[pollutant]
                print(f'drop {pollutant} data before {start_date}'  )
                data.loc[:start_date, pollutant] = np.nan
            
        # drop columns with all nan
        data = data.dropna(axis=1, how='all')

        #logger.info('data no fire has shape  {data.shape}')
        if dropna:
            self.data = data.dropna()
        else:
            self.data = data

    def save_(self):
        """Save the process data for fast loading without build.

        Save if the attribute exist.
        - save pollution data
        - save weather data
        - save data

        """

        if hasattr(self, 'poll_df'):
            # save pollution data
            if 'datetime' in self.poll_df.columns:
                # save without index
                self.poll_df.to_csv(self.data_folder + 'poll.csv', index=False)

            else:
                # save with index
                self.poll_df.to_csv(self.data_folder + 'poll.csv')

        if hasattr(self, 'wea'):
            # save data
            if 'datetime' in self.wea.columns:
                # save without index
                self.wea.to_csv(self.data_folder + 'weather.csv', index=False)

            else:
                # save with index
                self.wea.to_csv(self.data_folder + 'weather.csv')


        if hasattr(self, 'data'):
            # save data
            if 'datetime' in self.data.columns:
                # save without index
                self.data.to_csv(
                    self.data_folder + 'imp_data.csv', index=False)
            else:
                # save with index
                self.data.to_csv(self.data_folder + 'imp_data.csv')

        if hasattr(self, 'impute_pm25'):
            # save  data
            if 'datetime' in self.impute_pm25.columns:
                # save without index
                self.impute_pm25.to_csv(
                    self.data_folder + 'imp_pm25.csv', index=False)
            else:
                # save with index
                self.impute_pm25.to_csv(self.data_folder + 'imp_pm25.csv')

    
    def load_(self):
        """Load the process pollution data from the disk without the build

        Load if the file exist for.
        - pollution data
        - weather data
        - data no fire
        - data_org
        - data

        Args:
            fire: name of the hotspot instrument. Either 'MODIS' or 'VIIRS' 

        """

        if os.path.exists(self.data_folder + 'poll.csv'):
            self.poll_df = pd.read_csv(self.data_folder + 'poll.csv')
            self.poll_df['datetime'] = pd.to_datetime(self.poll_df['datetime'])
            self.poll_df.set_index('datetime', inplace=True)
            # add pollution list
            self.gas_list = self.poll_df.columns.to_list()

            if (self.city_name == 'Chiang Mai') :
                # for Thailand, delete all PM2.5 record before 2010
                self.poll_df.loc[:'2010', 'PM2.5'] = np.nan

            elif (self.city_name == 'Bangkok'):
                # for Thailand, delete all PM2.5 record before 2014
                self.poll_df.loc[:'2013', 'PM2.5'] = np.nan
            
            elif (self.city_name == 'Bangkok'):
                # for Thailand, delete all PM2.5 record before 2014
                self.poll_df.loc[:'2012', 'NO2'] = np.nan

            elif (self.city_name == 'Hat Yai'):
            # for Thailand, delete all PM2.5 record before 2014
                self.poll_df.loc[:'2015', 'PM2.5'] = np.nan
                #pass

        else:
            print('no pollution data. Call self.build_pollution first')


        if os.path.exists(self.data_folder + 'weather.csv'):
            self.wea = pd.read_csv(self.data_folder + 'weather.csv')
            try:
                self.wea.drop(['Time'],
                              axis=1,
                              inplace=True)
            except BaseException:
                pass
            self.wea['datetime'] = pd.to_datetime(self.wea['datetime'])
            self.wea.set_index('datetime', inplace=True)
        else:
            print('no weather data. Call self.build_weather first')

        if os.path.exists(self.data_folder + 'imp_data.csv'):
            self.data = pd.read_csv(
                self.data_folder + 'imp_data.csv')
            self.data['datetime'] = pd.to_datetime(
                self.data['datetime'])
            self.data.set_index('datetime', inplace=True)


         



