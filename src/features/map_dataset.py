# -*- coding: utf-8 -*-
import os
import sys
import pyproj
from pyproj import Transformer
import gdal
import geopandas as gpd
from shapely.geometry import Polygon, MultiPoint, Point, MultiPolygon
import fiona


if __package__: 
    from ..imports import *
    from ..gen_functions import *
    from .dataset import Dataset
    from .build_features import add_wea_vec
    from ..data.read_data import *
    from ..visualization.mapper import Mapper

else:
    # import anything in the upper directory 
    _i = os.path.dirname(os.path.dirname(os.path.abspath("..")))
    if _i not in sys.path:
        sys.path.insert(0, _i)
    from imports import *
    from gen_functions import *
    from data.read_data import *
    from visualization.mapper import Mapper
    # import files in the same directory
    _i = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _i not in sys.path:
        sys.path.insert(0, _i)
    from dataset import Dataset
    from build_features import add_wea_vec


class MapDataset():
    """MapDataset is used for easily process shape files and sattelite data. It is different from Mapper object, because MapDataset focus on one country. 

    """
    

    def __init__(self, country:str, main_folder: str = '../data/', report_folder:str='../reports/', n_jobs:int=-2):
        """
        Description of __init__

        Args:
            country (str): country to generate map
            main_folder (str='../data/'): main data folder for initializing Dataset object
            report_folder='../reports/' (str):
            n_jobs=-2 (int):

        """
        self.country = country
        self.main_folder = os.path.abspath(main_folder) + '/'
        #  folder to save the process data 
        self.map_folder = main_folder + 'poll_map/'
        # folder containing the shape file
        self.shapefile_folder  = main_folder + 'world_maps/'
        self.health_folder = main_folder + 'nso_data/'
        # folder to save the images 
        self.report_folder = os.path.abspath(report_folder ) + '/' + country + '/'
        if not os.path.exists(self.report_folder):
            os.mkdir(self.report_folder)

    def load_prov_df(self):

        # add region information 
        provinces = pd.read_csv(self.health_folder + 'income_prov/income_98_19.csv')[['province']]

        regions = ['Whole Kingdom', 'Greater Bangkok', 'Northeastern Region', 'Northern Region', 'Central Region',
               'Southern Region']

        provinces = provinces[~provinces['province'].isin(regions)]

        self.prov_map = provinces


    def load_shapefile(self, layer=2):
        """Load shape file for that country 

        """
        if self.country=='Thailand':
            filename = self.shapefile_folder + 'THA.gdb'
            # select province level
            prov_map = gpd.read_file(filename, driver='FileGDB', layer=layer)
            
            # overide old crs and convert
            crs = pyproj.CRS('EPSG:4326')
            prov_map['geometry'] = prov_map['geometry'].set_crs(crs, allow_override=True)
            prov_map['province'] = prov_map['admin1Name_en']

            # add region information 
            provinces = pd.read_csv(self.health_folder + 'income_prov/income_98_19.csv')[['province']]

            regions = ['Whole Kingdom', 'Greater Bangkok', 'Northeastern Region', 'Northern Region', 'Central Region',
                   'Southern Region']
            provinces = pd.concat([provinces,  provinces[provinces.isin(regions)]['province']], axis=1)
            provinces = provinces.fillna(method='ffill')
            provinces.columns = ['province', 'region']
            
            # fix province name 
            provinces['province'] = provinces['province'].str.replace('Bungkan', 'Bueng Kan')
            provinces['province'] = provinces['province'].str.replace('Phattalung', 'Phatthalung')
           
            prov_map = prov_map.merge(provinces, on='province', how='left')

            self.prov_map = prov_map


    def build_station_info(self):
        """Build station information using mapper object's build function

        """

        print(f'all build stations information using mapper object')
        mapper = Mapper(main_folder=self.main_folder)
        # because station order may shift as new data is download, so we have to build the station information before loading station information. 
        mapper.build_station_info()
        
        # if os.path.exists(self.map_folder + 'all_station_info.csv'):
        #     all_station_info = pd.read_csv(self.map_folder + 'all_station_info.csv')
        #     #print('number of stations =', self.all_station_info.shape)
        # else:
        #     raise AssertionError('no station information file')

        all_station_info = pd.read_csv(self.map_folder + 'all_station_info.csv')

        print(f'build stations information for {self.country}')

        # add city and country based in location of the stations
        city_list = []
        country_list = []
        for i, row in all_station_info.iterrows():
            p = Point(row['Longitude'], row['Latitude'])
            city = self.prov_map[self.prov_map.contains(p)]
            if len(city)> 0:
                city = city.iloc[0]['province']
                country_list.append(self.country)
            else:
                # not in this country 
                city = np.nan
                country_list.append(np.nan)
            city_list.append(city)
            
        all_station_info['province'] = city_list
        all_station_info['Country'] = country_list

        # keep stations in the country
        self.stations_info = all_station_info[all_station_info['Country']== self.country]
        self.stations_info = self.stations_info.dropna(axis=1, how='all')

        self.stations_info.to_csv(self.map_folder + f'{self.country}_stations_info.csv', index=False)

        
    def load_(self, layer=2):
        """Load compiled station information and add as attribute. 
    
        Args:
            layer: layer for loading the sub-region. [Default:2 means loading province level]
    
        Raises:
            AssertionError if station information file doesn't exist. 
    
        """
    
        self.load_shapefile(layer=layer)
    
        # station info filename 
        filename = self.map_folder + f'{self.country}_stations_info.csv'
        if os.path.exists(filename):
            self.stations_info = pd.read_csv(filename)
        else:
            self.build_station_info()

    def build_poll_prov(self, pollutant:str, build=False):
        """Compile pollution data for each province. 

        If there are more than one stations in the province, take the average of the value.
        Save the file for later use.

        Args:
            pollutant (str): pollutant to load the pollution 
            build: if True, build the data using mapper object
        
        To-dos:
            - load from the raw data directly to speed up the process

        """
        if build:
            # use mapper object to compile the latest pollution data.
            mapper = Mapper(main_folder=self.main_folder)
            mapper.load_()
            if self.country=='Thailand':
                data_columns =  mapper.build_pcd_data()
                mapper.build_cmu_data(data_columns)
                mapper.build_bkp_data(data_columns)
                if pollutant == 'PM2.5':
                    mapper.build_b_data(data_columns)
            else:
                data_columns =  mapper.build_pcd_data()
                mapper.build_b_data(data_columns)
                mapper.build_usemb_data(data_columns)

        prov_list = self.prov_map['province'].unique() 
        poll_filename = self.map_folder + 'data.csv'
        chunksize = 10000
        
        prov_poll = []
        
        
        for i, province in tqdm(enumerate(prov_list)):
            # search for station id
            temp = self.stations_info[self.stations_info['province'] == province]
            if len(temp)> 0:
                df = []
                for chunk in pd.read_csv(poll_filename, chunksize=chunksize):
                    # keep only data with stations id 
                    df.append(chunk[chunk['stationid'].isin(temp['id'])][['datetime', pollutant]])
                
                df = pd.concat(df)   
                df['datetime'] = pd.to_datetime(df['datetime'])
                df = df.sort_values('datetime')
                df = df.groupby('datetime').mean().dropna()
                df.columns = [province]
                prov_poll.append(df)
            else:
                prov_poll.append(pd.DataFrame())
                
        self.poll = pd.concat(prov_poll, axis=1)  
        self.poll = self.poll.dropna(how='all')
        pollutant = pollutant.replace('.','')
        load_filename = self.map_folder + f'{self.country}_{pollutant}_data.csv'
        self.poll.to_csv(load_filename, index=True)

    def load_poll_prov(self, pollutant:str):
        """Load pollution by province if the file already exist or call build_poll_province to generate one.

        Args:
            pollutant: pollutant to load the pollution 

        """

        temp = pollutant.replace('.','')
        load_filename = self.map_folder + f'{self.country}_{temp}_data.csv'
        if os.path.exists(load_filename):
            self.poll = pd.read_csv(load_filename)
            self.poll['datetime'] = pd.to_datetime(self.poll['datetime'])
            self.poll = self.poll.set_index('datetime')
        else:
            'Build pollution data by province'
            self.build_poll_prov(pollutant=pollutant)

