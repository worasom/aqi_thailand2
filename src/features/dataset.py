# -*- coding: utf-8 -*-
from ..imports import *
from ..data.read_data import *
from ..gen_functions import *
from ..data.fire_data import *
from ..data.weather_data import *
from .build_features import *

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
        poll_df:
        fire:
        wea:

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
        self.city_info['long_km'] = (merc_x(self.city_info['Longitude'])/1000).round()

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
        print('Loading all hotspots data. This might take sometimes')
        
        # the instrument is either MODIS or VIIRS 
        if instr =='MODIS':
            folder = self.main_folder + fire_data_folder + 'M6/*.csv'
        
        elif instr =='VIIRS':
            folder = self.main_folder + fire_data_folder + 'V1/*.csv'
        
        else:
            raise AssertionError('instrument name can be either MODIS or VIIRS')

        # keeping radius
        upper_lat = self.city_info['lat_km'] + distance
        lower_lat = self.city_info['lat_km'] - distance
        upper_long = self.city_info['long_km'] + distance
        lower_long = self.city_info['long_km'] - distance

        files = glob(folder)

        fire  = pd.DataFrame()
        for file in tqdm(files):
             
            df = pd.read_csv(file)
            # convert lat and long to km
            df['lat_km'] = (df['latitude'].apply(merc_y)/1E3).round().astype(int)
            df['long_km'] = (merc_x(df['longitude'])/1E3).round().astype(int)

            # remove by lat 
            df = df[df['lat_km'] <= (upper_lat)]
            df = df[df['lat_km'] >= (lower_lat)]

            # remove by long 
            df = df[df['long_km'] <= (upper_long)]
            df = df[df['long_km'] >= (lower_long)]

            fire = pd.concat([fire, df], ignore_index=True)

        fire = process_fire_data(filename=None,fire=fire,and_save=False)
        
        if instr =='MODIS':
            filename = self.data_folder + 'fire_m.csv'
        
        elif instr =='VIIRS':
            filename = self.data_folder + 'fire_v.csv'

        # add distance columns 
        fire['distance'] = np.sqrt((fire['lat_km'] - self.city_info['lat_km']/1000) **2 + ((fire['long_km'] - self.city_info['long_km']/1000)**2))
        # create power column and drop unncessary columns
        fire['power'] = fire['scan']*fire['track']*fire['frp']
        fire['count'] = 1
        print(fire.columns)
        fire = fire.drop(['latitude', 'longitude', 'brightness','acq_time','track','scan','frp'], axis=1)
        # save fire data
        fire.to_csv(filename, index=False)

    
    def build_weather(self,wea_data_folder:str='weather_cities/'):
        """Load weather data and feature engineer the weather data.

        """

        filename = self.city_wea_dict[self.city_name]
        filename = self.main_folder + wea_data_folder + filename.replace(' ','_') + '.csv'
        wea = pd.read_csv(filename)
        wea = fill_missing_weather(wea,limit=12)
        # round the weather data 
        wea[['Temperature(C)', 'Humidity(%)', 'Wind Speed(kmph)']] = wea[['Temperature(C)', 'Humidity(%)', 'Wind Speed(kmph)']].round()

        self.wea = wea

    def build_holiday(self):
        """Scrape holiday data from https://www.timeanddate.com/holidays/

        """
        if self.city_name=='Chiang Mai' or self.city_name=='Bangkok':
            head_url = 'https://www.timeanddate.com/holidays/thailand/'

        elif self.city_name=='Jakarta':
            head_url = 'https://www.timeanddate.com/holidays/indonesia/'
        
        elif self.city_name=='Hanoi':
            head_url = 'https://www.timeanddate.com/holidays/vietnam/'


        years = np.arange(2000,datetime.now().year+1) 
        
        holiday = pd.DataFrame()
        
        for year in years:
            url = head_url+str(year)
            df = pd.read_html(url)[0]
            df['year'] = year
            holiday = pd.concat([holiday,df],ignore_index=True)

        holiday.columns = ['Date', 'day_of_week','name','type','year']
        holiday = holiday[~holiday['Date'].isna()]

        holiday['date'] = holiday['Date'] + ', ' + holiday['year'].astype(str)
        holiday['date'] = pd.to_datetime(holiday['date'])
        holiday.to_csv(self.data_folder + 'holiday.csv', index=False)


    def build_all_data(self,build_fire:bool=False,build_holiday:bool=False):
        """Build all data from raw files. Use after just download more raw data.

        """
        self.build_pollution()
        self.build_weather()
        self.save_()

        if build_fire:
            self.build_fire()
        
        if build_holiday:
            self.build_holiday()

    def feature_no_fire(self,pollutant:str='PM2.5'):
        """Assemble pollution data, datetime and weather data. Omit the fire data for later step. 

        """
        if not hasattr(self, 'poll_df') or not hasattr(self, 'wea'):
            self.load_()

        if not os.path.exists(self.data_folder +'holiday.csv'):
            self.build_holiday()

        self.pollutant = pollutant
        cols = [pollutant, 'Temperature(C)', 'Humidity(%)', 'Wind', 'Wind Speed(kmph)', 'Condition']
        # merge pollution and wind data 

        data = self.poll_df.merge(self.wea, left_index=True, right_index =True,how='inner')

        # select data and drop null value
        data = data[cols]
        data = data.dropna()

        if (pollutant == 'PM2.5') and self.city_name=='Chiang Mai':
            data = data.loc['2010':]
        # add lag information
        data = add_lags(data, pollutant)
        # one hot encode wind data 
        dummies = wind_to_dummies(data['Wind'])
        data.drop('Wind',axis=1, inplace=True)
        data = pd.concat([data, dummies], axis=1)
        data = add_is_rain(data)
        data = add_calendar_info(data,holiday_file=self.data_folder +'holiday.csv')
        try:
            data = data.astype(float)
        except:
            raise AssertionError('some data cannot be convert to float')
        
        # find duplicate index and drop them
        data.sort_index(inplace=True)
        data = data.loc[~data.index.duplicated(keep='first')]
 
        print('data no fire has shape', data.shape)
        self.data_no_fire = data

    def merge_fire(self,fire_dict=None):
        """Process raw hotspot data into fire feature and merge with the rest of the data 

        """
        if not fire_dict:
            # did not pass fire_dict
            # use self.fire_dict attribute
            if not hasattr(self,'fire_dict'):
                # fire dict attribute does not exist. Load from model meta 
                with open(self.data_folder + model_meta.json) as f:
                    model_meta = json.load(f)
                try: 
                    self.fire_dict = model_meta['fire_dict']
                except: 
                    # use default value
                    self.fire_dict = {'fire_col': 'power', 'surface': 'sphere', 'w_speed': 4,'shift': -24, 'roll': 108}
            else:
                fire_dict = self.fire_dict
            
        if self.city_name=='Chiang Mai':
            zone_list = [0, 100, 400, 700, 1000]
        else:
            zone_list = [0,100,200,400,800,1000]

        fire_proc, _ = get_fire_feature(self.fire,zone_list=zone_list, 
                        fire_col=fire_dict['fire_col'],damp_surface=fire_dict['damp_surface'], 
                        shift=fire_dict['shift'], roll=fire_dict['roll'], w_speed=fire_dict['w_speed'])

        # merge with fire data 
        data = self.data_no_fire.merge(fire_proc, left_index=True, right_index=True, how='inner')
        data = data.dropna()
        data  = data.loc[~data.index.duplicated(keep='first')]
        self.data = data 

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
                self.fire.to_csv(self.data_folder +'fire.csv',index=False)
        
            else:
                # save with index
                self.fire.to_csv(self.data_folder +'fire.csv')

        if hasattr(self, 'wea'):
            # save fire data 
            if 'datetime' in self.wea.columns:
                # save without index 
                self.wea.to_csv(self.data_folder +'weather.csv',index=False)
        
            else:
                # save with index
                self.wea.to_csv(self.data_folder +'weather.csv')

        if hasattr(self, 'data_no_fire'):
            # save fire data 
            if 'datetime' in self.data_no_fire.columns:
                # save without index 
                self.data_no_fire.to_csv(self.data_folder +'data_no_fire.csv',index=False)
        
            else:
                # save with index
                self.data_no_fire.to_csv(self.data_folder +'data_no_fire.csv')


    def load_(self,fire='MODIS'):
        """Load the process pollution data from the disk without the build

        """
        
        if os.path.exists(self.data_folder +'poll.csv'):
            self.poll_df = pd.read_csv(self.data_folder +'poll.csv')
            self.poll_df['datetime'] = pd.to_datetime(self.poll_df['datetime'])
            self.poll_df.set_index('datetime',inplace=True)
            # add pollution list 
            self.gas_list = self.poll_df.columns.to_list()
        else:
            print('no pollution data. Call self.build_pollution first')

        if fire=='MODIS':
            filename = self.data_folder +'fire_m.csv'
        else:
            filename = self.data_folder +'fire_v.csv'
        
        if os.path.exists(filename):
            self.fire = pd.read_csv(filename)
            self.fire['datetime'] = pd.to_datetime(self.fire['datetime'])
            self.fire.set_index('datetime',inplace=True)
        else:
            print('no fire data. Call self.build_fire first')

        if os.path.exists(self.data_folder +'weather.csv'):
            self.wea = pd.read_csv(self.data_folder +'weather.csv')
            self.wea.drop(['Time','Dew Point(C)','Wind Gust(kmph)','Pressure(in)','Precip.(in)'], axis=1, inplace=True)
            self.wea['datetime'] = pd.to_datetime(self.wea['datetime'])
            self.wea.set_index('datetime',inplace=True)
        else:
            print('no weather data. Call self.build_weather first')

        if os.path.exists(self.data_folder +'data_no_fire.csv'):
            self.data_no_fire = pd.read_csv(self.data_folder +'data_no_fire.csv')
            self.data_no_fire['datetime'] = pd.to_datetime(self.data_no_fire['datetime'])
            self.data_no_fire.set_index('datetime',inplace=True)




         
        


