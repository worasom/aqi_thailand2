# -*- coding: utf-8 -*-
from ..imports import *
from ..data.read_data import *
"""Pollution Dataset Object of a particular city 

"""

class Dataset():
    """Dataset object contains pollution, weather and fire data for a city. In charge of feature engineering for ML input.

    Args:
        city_name: lower case of city name 
        main_folder(optional): main data folder [default:'../data/]

    Attributes:
        gas_list:
        city_name:
        main_folder:
        data_folder:
        model_folder: 
        city_info:

    Raises:
        AssertionError: if the city_name is not in city_names list 
    
    """
    
    #list of pollutants
    gas_list = ['PM2.5', 'PM10', 'O3', 'CO', 'NO2', 'SO2']
    # mapping city name to weather city name 
    city_wea_dict = {'Chiang Mai': 'Mueang Chiang Mai',
                 'Bangkok':'Bangkok',
                'Hanoi':'Soc Soc'}

    def __init__(self, city_name:str,main_data_folder:str='../data/', model_folder='../models/'):

        city_names =['Chiang Mai','Bangkok','Hanoi','Jakarta']

        if city_name not in city_names:
            raise AssertionError('city name not in the city_names list. No data for this city')
        else:
            # the city exist in the database set folders attributes 
            self.city_name = city_name
            city_name = city_name.lower().replace(' ','_')
            self.main_folder = main_data_folder
            self.data_folder = main_data_folder+city_name +'/'
            self.model_folder = model_folder+city_name +'/'
            self.wea_name = self.city_wea_dict[self.city_name]

        if not os.path.exists(self.data_folder):
            os.mkdir(self.data_folder)

        if not os.path.exists(self.model_folder):
            os.mkdir(self.model_folder)

        self.load_city_info()

    
    def load_city_info(self):
        # load city information 
        with open(self.main_folder+'pm25/cities_info.json','r') as f:
            city_info_list = json.load(f)

        for city_json in city_info_list:
            if self.city_name == city_json['City']:
                self.city_info = city_json
                break

    def get_th_stations(self):
        """Look for all polltuions station by Thailand PCD.

        """
        # load stations information for air4 Thai
        station_info_file = f'{self.main_folder}aqm_hourly2/' + 'stations_locations.json'
        with open(station_info_file, 'r',encoding="utf8") as f:
            station_info = json.load(f)
        
        
        station_info = station_info['stations']

        # find stations in Chiangmai and parase that files
        station_ids = []
        station_info_list = []
         
        for i, stations in enumerate(station_info):
            if self.city_name in stations['areaEN']:
                station_ids.append(stations['stationID'])
                station_info_list.append(stations)

        return station_ids, station_info_list

    def merge_new_old_pollution(self, station_ids):
        """Merge Thai pollution data from the station in station_ids list from two folders.
        
        """
        for station_id in station_ids:
            # load old data if exist 
            try: 
                old_data = pd.read_csv(f'{self.main_folder}/aqm_hourly2/' + 'process/'+station_id + '.csv')
            except:
                old_data = pd.DataFrame()
            else:
                old_data['datetime'] = pd.to_datetime(old_data['datetime'])
                old_data = old_data.set_index('datetime')
                # keep only the gass columns
                old_data = old_data[self.gas_list]
    
            new_data = pd.read_csv(f'{self.main_folder}/air4thai_hourly/' + station_id + '.csv',na_values='-')
            new_data = new_data.set_index('datetime')
            new_data.columns = [s.split(' (')[0] for s in new_data.columns]
            # keep only the gass columns
            new_data = new_data[self.gas_list]
            # concatinate data and save
            data = pd.concat([old_data,new_data])
            filename = self.data_folder+station_id + '.csv'
            print('save file', filename)
            data.to_csv(filename)

    def build_pollution(self):
        """Load Pollution data from a different sources and take the average. 

        Since each city have different data sources. It has to be treat differently 

        """
        # data list contain the dataframe of all pollution data before merging
        # all of this data has 'datetime' as a columns
        data_list = []
        # load data from Berekely Earth Projects This is the same for all cities
        b_data, _ = read_b_data(self.main_folder+'pm25/'+self.city_name.replace(' ','_') + '.txt')
        data_list.append(b_data)

        if self.city_name =='Chiang Mai':
            # Chiang Mai has two stations, which are stored into folder (historical data and newly scrape data)
            station_ids, _ =  self.get_th_stations()
            # for Chiang mai keep only the first two stations 
            station_ids = station_ids[:2]
            # update the file
            self.merge_new_old_pollution(station_ids)
            # load the file 
            for station_id in station_ids:
                filename = self.data_folder+station_id + '.csv'
                data = pd.read_csv(filename)
                data['datetime'] = pd.to_datetime(data['datetime'])
                data_list.append(data)

        elif self.city_name =='Hanoi':
            # for Hanoi Data, also load Ha Dong Data 
            b_data, _ = read_b_data(self.main_folder+'pm25/'+ 'Ha_Dong.txt')
            data_list.append(b_data)


        return data_list
         
        


