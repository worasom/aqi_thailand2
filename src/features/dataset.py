# -*- coding: utf-8 -*-
from ..imports import *
from ..data.read_data import *
from ..gen_functions import *
from ..data.fire_data import *

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
        poll_df

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

        # add lattitude and longtitude in km
        self.city_info['lat_km'] = (merc_y(self.city_info['Latitude'])/1000).round()
        self.city_info['long_km'] = (merc_y(self.city_info['Longitude'])/1000).round()

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

    def collect_stations_data(self):
        """Collect all Pollution data from a different sources and take the average. 

        Since each city have different data sources. It has to be treat differently 

        Returns: a list of dataframe each dataframe is the data from all station.

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

        elif self.city_name =='Bangkok':
            # List of Bangkok stations that has been processed 
            station_ids = ['02t', '03t', '05t', '11t', '12t', '50t', '52t', '53t', '59t', '61t']
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
            data_list += build_us_em_data(city_name=self.city_name, data_folder=f'{self.main_folder}us_emb/')

        elif self.city_name =='Jakarta':
            data_list.append(b_data)
            data_list += build_us_em_data(city_name=self.city_name, data_folder=f'{self.main_folder}us_emb/')

        return data_list

    def build_pollution(self):
        """Collect all Pollution data from a different sources and take the average. 

        Use self.collect_stations_data to get a list of pollution dataframe.
        Add pollution data as attribute self.poll_df

        """

        data_list = self.collect_stations_data()
        print(f'Averaging data from {len(data_list)} stations')
        data = pd.DataFrame()
        for df in data_list:
            df = df.sort_values(['datetime','PM2.5'])
            df = df.drop_duplicates('datetime')
            data = pd.concat([data, df], axis=0,ignore_index=True)

        # take the average of all the data
        data = data.groupby('datetime').mean()
        data = data.dropna(how='all')

        self.poll_df = data

    def build_fire(self,instr:str='MODIS', distance=1000,fire_data_folder:str='fire_map/world_2000-2020/'):
        """Extract hotspots satellite data within distance from the city location.
        
        Args:
            instr: instrument name either MODIS or VIIRS 
            distance: distance in km from the city latitude and longtitude
            fire_data_folder: location of the hotspots data 
        
        Raises:
            AssertionError: if the instrument name does not exist
        
        """
        print('Building fire data. This might take sometimes')
        
        # the instrument is either MODIS or VIIRS 
        if instr =='MODIS':
            folder = self.main_folder + fire_data_folder + 'M6/*.csv'
        
        elif instr =='VIIRS':
            folder = self.main_folder + fire_data_folder + 'V1/*.csv'
        
        else:
            raise AssertionError('instrument name can be either MODIS or VIIRS')

        # keeping radius
        r_lat = self.city_info['lat_km'] + distance
        r_long = self.city_info['long_km'] + distance

        files = glob(folder)

        fire  = pd.DataFrame()
        for file in tqdm(files):
            df = pd.read_csv(file)
            # convert lat and long to km
            df['lat_km'] = (df['latitude'].apply(merc_y)/1E3).round().astype(int)
            df['long_km'] = (merc_x(df['longitude'])/1E3).round().astype(int)

            # remove by lat 
            df = df[df['lat_km'] <= (r_lat)]
            df = df[df['lat_km'] >= (r_lat)]

            # remove by long 
            df = df[df['long_km'] <= (r_long)]
            df = df[df['long_km'] >= (r_long)]

            fire = pd.concat([fire, df], ignore_index=True)

        fire = process_fire_data(filename=None,fire=fire,and_save=False)
        
        if instr =='MODIS':
            filename = self.data_folder + 'fire_m.csv'
        
        elif instr =='VIIRS':
            filename = self.data_folder + 'fire_v.csv'
        
        # save fire data
        fire.to_csv(filename, index=False)

    
    def save_(self):
        """Save the process data for fast loading without the build

        """
        if hasattr(self, 'poll_df'):
            # save pollution data 
            if 'datetime' in self.poll_df.columns:
            # save without index 
                self.poll_df.to_csv(self.data_folder +'poll.csv',index=False)
        
            else:
                # save with index
                self.poll_df.to_csv(self.data_folder +'poll.csv')
        
        if hasattr(self, 'fire'):
            # save fire data 
            if 'datetime' in self.fire.columns:
                # save without index 
                self.poll_df.to_csv(self.data_folder +'fire.csv',index=False)
        
            else:
                # save with index
                self.poll_df.to_csv(self.data_folder +'fire.csv')


    def load_(self):
        """Load the process pollution data from the disk without the build

        """
        
        if os.path.exists(self.data_folder +'poll.csv'):
            self.poll_df = pd.read_csv(self.data_folder +'poll.csv')
            self.poll_df['datetime'] = self.poll_df['datetime']
            self.poll_df.set_index('datetime',inplace=True)
        else:
            print('no pollution data. Call self.build_pollution first')

        if os.path.exists(self.data_folder +'fire.csv'):
            self.fire = pd.read_csv(self.data_folder +'fire.csv')
            self.fire['datetime'] = self.fire['datetime']
            self.fire.set_index('datetime',inplace=True)
        else:
            print('no pollution data. Call self.build_fire first')




         
        


