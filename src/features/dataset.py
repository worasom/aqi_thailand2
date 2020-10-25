# -*- coding: utf-8 -*-
from ..imports import *
from ..gen_functions import *
from ..data.read_data import *
from ..data.fire_data import *
from ..data.weather_data import *
from .build_features import *
from .config import set_config



"""Pollution Dataset Object of a particular city. This object is contains the raw dataset
and processed data, and is in charge of splitting data for training.

"""


class Dataset():
    """Dataset object contains pollution, weather and fire data for a city.
    It is also in charge of feature engineering and splitting data for ML hyperparameter tuning.

    Args:
        city_name: lower case of city name
        main_folder(optional): main data folder [default:'../data/]
        model_folder(optional): model folder[default:'../models/']
        report_folder(optional): folder to save figure[default:'../reports/']

    Attributes:
        city_name: name of the city
        main_folder: main data folder [default:'../data/']. This folder contains raw data.
        data_folder: data folder spcified for this city for keeping processed data
        model_folder: model folder for this city for model meta and model files
        report_folder: report folder for saving figures
        city_info: dictionary contain city latitude and longtitude
        poll_df: raw pollution data
        fire: raw fire data
        wea: raw weather data
        data_no_fire: processed pollution data, and weather data
        data: processed pollution, weather and fire data
        x_cols:
        fire_dict:
        pollutant
        monitor

    Raises:
        AssertionError: if the city_name is not in city_names list

    """
    # a defaul list of pollutants
    #gas_list = ['PM2.5', 'PM10', 'O3', 'CO', 'NO2', 'SO2']

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
        # add city name as attribute
        self.city_name = city_name
        # add congiruation attribute
        set_config(self)

        if city_name not in self.city_wea_dict.keys():
            print('city name not in the city_names list. No weather data for this city')
        else:
            self.wea_name = self.city_wea_dict[self.city_name]
        # the city exist in the database set folders attributes    
        city_name = city_name.lower().replace(' ', '_')
        self.main_folder = main_data_folder
        self.data_folder = main_data_folder + city_name + '/'
        self.model_folder = model_folder + city_name + '/'
        self.report_folder = report_folder + city_name + '/'
            
        if not os.path.exists(self.data_folder):
            os.mkdir(self.data_folder)

        if not os.path.exists(self.model_folder):
            os.mkdir(self.model_folder)

        if not os.path.exists(self.report_folder):
            os.mkdir(self.report_folder)

        self.load_city_info()

    def load_city_info(self):
        """Load city information add as city_info dictionary.

        Add latitude and longtitude information in mercadian coordinate (km). Add as 'lat_km' and 'long_km' keys

        """
        with open(self.main_folder + 'pm25/cities_info.json', 'r') as f:
            city_info_list = json.load(f)

        for city_json in city_info_list:
            if self.city_name == city_json['City']:
                self.city_info = city_json
                break

        # add lattitude and longtitude in km
        self.city_info['lat_km'] = (
            merc_y(
                self.city_info['Latitude']) /
            1000).round()
        self.city_info['long_km'] = (
            merc_x(
                self.city_info['Longitude']) /
            1000).round()

        if self.city_info['Country'] == 'Viet Nam':
            # fix the name of Vietnam
            self.city_info['Country']= 'Vietnam'


    @staticmethod
    def parse_th_station(folder, station_id):
        """Parse raw station data into .csv file in the form, which is ready to merge with the newly parse data.

        Look for all the files containing station_id in the filename.
        Create the folder to save the file if doesn't exist.

        Args:
            folder: folder where the raw data file is saved
            station_id: station_id
            save_filename: filename to save the data as

        """
        # look for all files containing station_id in the filename
        p = Path(folder)
        filenames = []
        for i in p.glob('**/*.xlsx'):
            if station_id in i.name:
                filenames.append(str(i))

        # if filename exist load that file
        if len(filenames) > 0:
            save_filename = folder + 'process/' + station_id + '.csv'

            if not os.path.exists(folder + 'process/'):
                os.mkdir(folder + 'process/')
        
            station_data = []
            for f in filenames:
                try:
                    station_data.append(read_his_xl(f))
                except BaseException:
                    pass
            if len(station_data) > 0:
                station_data = pd.concat(station_data).drop_duplicates()
            else:
                print('cannot parse data from station', station_id)
            # save the data if the dataframe is not empty
            if len(station_data) > 0:
                print('save data as', save_filename)
                station_data.to_csv(save_filename, index=True)     

        else:
            print( 'not data file for station', station_id)           

    def merge_new_old_pollution(
            self,
            station_ids: list,
            hist_folders: str = ['aqm_hourly2/', 'aqm_hourly3/'],
            new_folder='air4thai_hourly/',
            parse=False):
        """Merge Thai pollution data from the station in station_ids list from two folders: the historical data folder and new data folder.

        Save the data for each station as data_folder/station_id.csv

        Args:
            station_ids: a list of pollution station for the city.
            his_folders(optional): a list of folder containg the historcal data folder default:['aqm_hourly2/', 'aqm_hourly3/']
            new_folder(optional): name of the new data folder(which is update constantly)[default:'air4thai_hourly/']
            parse(optional): if True, also parse the stations data from the excel files
        """
        for station_id in station_ids:
            

            old_data_list = []
            # extract data from both folder
            for hist_folder in hist_folders:
                # load old data if exist
                old_filename = f'{self.main_folder}{hist_folder}' + \
                    'process/' + station_id + '.csv'
                if not os.path.exists(old_filename):
                    # data not exists parse data from raw excel file
                    self.parse_th_station(
                        f'{self.main_folder}{hist_folder}', station_id)

                try:
                    old_data = pd.read_csv(old_filename)
                except BaseException:
                    old_data = pd.DataFrame()
                else:
                    old_data['datetime'] = pd.to_datetime(old_data['datetime'])
                    old_data = old_data.set_index('datetime')
                    # keep only the gass columns
                    gas_list = [
                        s for s in self.gas_list if s in old_data.columns]
                    old_data = old_data[gas_list]
                    old_data_list.append(old_data)

            # combine old data from two folders
            if len(old_data_list) > 0:
                old_data = pd.concat(old_data_list)
                old_data.index = pd.to_datetime(old_data.index)
                old_data = old_data.sort_index()
                old_data = old_data[~old_data.index.duplicated(keep='first')]
            else:
                # no data 
                old_data = pd.DataFrame()

            # read data parse from the website
            try:
                new_data = pd.read_csv(
                    f'{self.main_folder}{new_folder}' +
                    station_id +'.csv',
                    na_values='-')
            except:
                new_data = pd.DataFrame()
            
            else:
                new_data['datetime'] = pd.to_datetime(new_data['datetime'])
                new_data = new_data.set_index('datetime')
                new_data.columns = [s.split(' (')[0] for s in new_data.columns]
                # keep only the gas columns
                new_data = new_data[self.gas_list]

            # concatinate data and save
            data = pd.concat([old_data, new_data])
            data = data.sort_index()
            data = data[~data.index.duplicated(keep='first')]
            filename = self.data_folder + station_id + '.csv'
            
            if len(data) > 0:
                print('save file', filename)
                data.to_csv(filename)
            else:
                print( 'no old and new data for station', station_id)

    def collect_stations_data(self):
        """Collect all Pollution data from a different sources and take the average.

        Since each city have different data sources. It has to be treat differently. 
        The stations choices is specified by the config.json

        Returns: a list of dataframe each dataframe is the data from all station.

        """
        # data list contain the dataframe of all pollution data before merging
        # all of this data has 'datetime' as a columns
        data_list = []

        # load data from Berkeley Earth Projects This is the same for all
        # cities
        b_data, _ = read_b_data(self.main_folder + 'pm25/' + self.city_name.replace(' ', '_') + '.txt')
        data_list.append(b_data)

        config_dict = self.config_dict
        
        # load thailand stations if applicable 
        if 'th_stations' in config_dict.keys():
            station_ids = config_dict['th_stations']
            print('th_stations', station_ids)
            self.merge_new_old_pollution(station_ids)
            # load the file
            for station_id in station_ids:
                filename = self.data_folder + station_id + '.csv'
                data = pd.read_csv(filename)
                data['datetime'] = pd.to_datetime(data['datetime'])
                data_list.append(data)
        # load the Thailand stations maintained by cmucdc project 
        if 'cmu_stations' in config_dict.keys():
            station_ids = config_dict['cmu_stations']
            print('cmu_stations', station_ids)
            for station_id in station_ids:
                filename = self.main_folder + 'cdc_data/' + str(station_id) + '.csv' 
                data_list.append(read_cmucdc(filename))
        
        if 'b_stations' in config_dict.keys():
            # add Berkeley stations in near by provinces 
            station_ids = config_dict['b_stations']
            print('add Berkerley stations', station_ids)
            for station_id in station_ids:
                b_data, _ = read_b_data(self.main_folder + 'pm25/' + f'{station_id}.txt')
                data_list.append(b_data)

        if 'us_emb' in config_dict.keys():
            # add the data from US embassy 
            print('add US embassy data')
            data_list += build_us_em_data(city_name=self.city_name,
                                                    data_folder=f'{self.main_folder}us_emb/')

        return data_list 

    def build_pollution(self):
        """Collect all Pollution data from a different sources and take the average.

        Use self.collect_stations_data to get a list of pollution dataframe.
        Add the average pollution data as attribute self.poll_df

        """

        data_list = self.collect_stations_data()
        print(f'Averaging data from {len(data_list)} stations')
        data = pd.DataFrame()
        for df in data_list:
            df = df.sort_values(['datetime', 'PM2.5'])
            df = df.drop_duplicates('datetime')
            data = pd.concat([data, df], axis=0, ignore_index=True)

        # take the average of all the data
        data = data.groupby('datetime').mean()
        data = data.dropna(how='all')

        self.poll_df = data.round()

    def build_fire(self, instr: str = 'MODIS', distance=1200,
                   fire_data_folder: str = 'fire_map/world_2000-2020/'):
        """Extract hotspots satellite data within distance from the city location.

        #. Loop through the fire data in the folder to extract the hotspots within the distance from the city
        #. Call process fire data to add datetime information
        #. Calculate the distance from the hotspot to the city location.
        #. Add fire power column  = scan * track*frp. This account of the size and the temperature of the fire
        #. Add the count column, which is 1
        #. Remove unncessary columns
        #. Save the data as data_folder/fire_m.csv if instr is "MODIS'. Use fire_v.csv if instr is 'VIIR'.

        Args:
            instr(optional): instrument name either MODIS or VIIRS[default:'MODIS']
            distance(optional): distance in km from the city latitude and longtitude[default:1000]
            fire_data_folder(optional): location of the hotspots data[default:'fire_map/world_2000-2020/']

        Raises:
            AssertionError: if the instrument name does not exist

        """
        print('Loading all hotspots data. This might take sometimes')

        #if self.city_name == 'Hanoi':
        #    distance = 1200

        # else self.city_name == 'Bangkok':
        #    distance = 600

        # the instrument is either MODIS or VIIRS
        if instr == 'MODIS':
            folder = self.main_folder + fire_data_folder + 'M6/*.csv'

        elif instr == 'VIIRS':
            folder = self.main_folder + fire_data_folder + 'V1/*.csv'

        else:
            raise AssertionError(
                'instrument name can be either MODIS or VIIRS')

        files = glob(folder)

        fire = pd.DataFrame()
 
        lat_km = self.city_info['lat_km']
        long_km = self.city_info['long_km']

        # use joblib to speed up the file reading process over 2 cpus
        fire = Parallel(
            n_jobs=2)(
            delayed(read_fire)(
                file,
                lat_km,
                long_km,
                distance) for file in files)
        fire = pd.concat(fire, ignore_index=True)
        fire = fire.drop_duplicates(ignore_index=True)

        fire = process_fire_data(filename=None, fire=fire, and_save=False)

        if instr == 'MODIS':
            filename = self.data_folder + 'fire_m.csv'

        elif instr == 'VIIRS':
            filename = self.data_folder + 'fire_v.csv'

        # add distance columns
        fire['distance'] = np.sqrt((fire['lat_km'] -
                                    self.city_info['lat_km']) ** 2 +
                                   ((fire['long_km'] -
                                     self.city_info['long_km'])**2))
        # create power column and drop unncessary columns
        fire['power'] = fire['scan'] * fire['track'] * fire['frp']
        fire['count'] = 1

        try:
            fire = fire.drop(['brightness',
                              'acq_time',
                              'track',
                              'scan',
                              'frp'],
                             axis=1)
        except BaseException:
            fire = fire.drop([
                'bright_ti4',
                'acq_time',
                'track',
                'scan',
                'frp'],
                axis=1)

        # save fire data
        fire.to_csv(filename)

    def build_weather(self, wea_data_folder: str = 'weather_cities/'):
        """Load weather data and fill the missing value. Add as wea attibute.

        Args:
            wea_data_folder(optional): weather data folder[default:'weather_cities/']

        """
        # check if there is a weather data
        if hasattr(self, 'wea_name'):
            filename = self.wea_name
            filename = self.main_folder + wea_data_folder + \
                filename.replace(' ', '_') + '.csv'
            wea = pd.read_csv(filename)
            wea = fill_missing_weather(wea, limit=12)
            # round the weather data
            wea[['Temperature(C)', 'Humidity(%)', 'Wind Speed(kmph)']] = wea[[
            'Temperature(C)', 'Humidity(%)', 'Wind Speed(kmph)']].round()

            self.wea = wea
        else:
            print('no weather data')

    def build_holiday(self):
        """Scrape holiday data from https://www.timeanddate.com/holidays/ since 2000 until current.

        Save the data as data_folder/holiday.csv

        """
     
        country = self.city_info['Country'].lower()
        print('Getting holiday for ', country)
        head_url = f'https://www.timeanddate.com/holidays/{country}/'

        years = np.arange(2000, datetime.now().year + 1)

        holiday = pd.DataFrame()

        for year in years:
            url = head_url + str(year)
            df = pd.read_html(url)[0]
            df['year'] = year
            holiday = pd.concat([holiday, df], ignore_index=True)

        holiday.columns = ['Date', 'day_of_week', 'name', 'type', 'year']
        holiday = holiday[~holiday['Date'].isna()]

        holiday['date'] = holiday['Date'] + ', ' + holiday['year'].astype(str)
        holiday['date'] = pd.to_datetime(holiday['date'])
        holiday.to_csv(self.data_folder + 'holiday.csv', index=False)

    def build_all_data(
            self,
            build_fire: bool = False,
            build_holiday: bool = False):
        """Build all data from raw files. Use after just download more raw data.

        The function build pollution and weather data by default, but fire and holiday data are optionals

        Args:
            build_fire(optional): if True, also build the fire data[default:False]
            build_holiday(optional): if True, also build the holiday data[default:False]

        """
        self.build_pollution()
        self.build_weather()
        self.save_()

        if build_fire:
            self.build_fire()

        if build_holiday:
            self.build_holiday()

        self.save_()

    def feature_no_fire(
            self,
            pollutant: str = 'PM2.5',
            rolling_win=24,
            fill_missing=False,
            cat_hour=False,
            group_hour=2, cat_dayofweek=False):
        """Assemble pollution data, datetime and weather data. Omit the fire data for later step.

        #. Call self.load_() to load processed data
        #. Build holiday data if not already exist
        #. Add pollutant as pollutant attribute

        Args:
            pollutant(optional): name of the pollutant [default:'PM2.5']
            rolling_win(optional): rolling windows size [defaul:24]. This does not do anything.
            fill_missing(optional): if True, fill the missing pollution data
            cat_hour(optional): if true, one hot encode the time_of_day column
            group_hour(optiona): hour to reduce the catergory of the time_of_day. This is needed if cat_hour==True
            cat_dayofweek(optional): if true, one hot encode the day_of_week column

        Raises:
            AssertionError: if pollutant not in self.poll_df

        """
        if not hasattr(self, 'poll_df') or not hasattr(self, 'wea'):
            self.load_()

        if not os.path.exists(self.data_folder + 'holiday.csv'):
            self.build_holiday()

        # check if pollutant data exist
        if pollutant not in self.poll_df.columns:
            raise AssertionError(f'No {pollutant} data')
        self.pollutant = pollutant

        if fill_missing:
            self.poll_df = fill_missing_poll(self.poll_df, limit=6)

        cols = [
            pollutant,
            'Temperature(C)',
            'Humidity(%)',
            'Wind',
            'Wind Speed(kmph)',
            'Condition']
        # merge pollution and wind data

        if 'datetime' in self.wea.columns:
            self.wea.set_index('datetime', inplace=True)

        data = self.poll_df.merge(
            self.wea,
            left_index=True,
            right_index=True,
            how='inner')

        # select data and drop null value
        data = data[cols]
        data[pollutant] = data[pollutant].rolling(
            rolling_win, min_periods=0).mean().round(1)
        data = data.dropna()

        # if (pollutant == 'PM2.5') and self.city_name == 'Chiang Mai':
        #     data = data.loc['2010':]
        # elif self.city_name == 'Hanoi':
        #     data = data.loc['2016-03-21':]

        # some data need to be crop. This is specified by the crop_dict in the config.py 
        try: 
            start_date = self.crop_dict[self.city_name][self.pollutant]
        except:
            pass
        else:
            data = data.loc[start_date:]

        # add lag information
        # data = add_lags(data, pollutant)
        # one hot encode wind data
        dummies = wind_to_dummies(data['Wind'])
        data.drop('Wind', axis=1, inplace=True)
        data = pd.concat([data, dummies], axis=1)
        data = add_is_rain(data)
        data = add_calendar_info(
            data, holiday_file=self.data_folder + 'holiday.csv')

        # fix COVID shutdown for Bangkok
        if self.city_name == 'Bangkok':
            data.loc['2020-03-21':'2020-04-16','is_holiday'] = 1

        if cat_hour:
            # one hot encode the time of day columns
            data = dummy_time_of_day(
                data, col='time_of_day', group_hour=group_hour)
        
        if cat_dayofweek:
            data = dummy_day_of_week(data)

        # include traffic data if exist 
        if hasattr(self,'traffic'):
            data = data.merge(self.traffic, left_index=True, right_index=True, how='inner')
        
        try:
            data = data.astype(float)
        except BaseException:
            raise AssertionError('some data cannot be convert to float')

        # find duplicate index and drop them
        data.sort_index(inplace=True)
        data = data.loc[~data.index.duplicated(keep='first')]

        print('data no fire has shape', data.shape)
        self.data_no_fire = data

    def get_wind_damp_fire(self, wind_lag):
        """Use self.fire and self.wea dataframes to calculate the self.damp_fire feature. 
        The self.damp_fire has a winddamp columns, which is the damping factor from a dot production between the location of the hotspot and  the wind direction.
        Any hotspots with the dot products less than 0 is removed. 

        This function also calculated the hotspot arrival time to the city using the dynamic wind speed (not the average wind speed). 

        Args:

            wind_damp: if True, use fire_damp attribute for fire feature calculation instead of fire. If fire_damp hasn't exsited, calculate it. 
            wind_lag: if True, use real wind speed for arrivial time 

        """
        
        # make a copy of the fire data and weather data 
        fire_df = cal_wind_damp(self.fire.copy(), self.wea, self.city_info['long_km'] , self.city_info['lat_km'] )
         
        if wind_lag:
            # overide the index of the fire with the new arrival time 
            fire_df.index = cal_arrival_time(detection_time=fire_df.index, distance=fire_df['distance'], wind_speed=fire_df['Wind Speed(kmph)'])
        # set at attribute
        self.damped_fire = fire_df

    def merge_fire(self, fire_dict=None, damp_surface='sphere', wind_damp =False, wind_lag=False):
        """Process raw hotspot data into fire feature and merge with the rest of the data
        If wind_damp is True, use self.damped_fire attribute for fire data, if False, use self.fire attribute.  

        Call src.features.build_features.get_fire_feature() to calcuate the daming due to distance and shift due to effective wind_speed. 

        The fire_proc dataframe is merged with the rest of the pollution, and weather dataframe. 

        Args:
            fire_dict(optional): fire dictionary containing wind_speed, shift and roll as keys [default:None] 
            damp_surface(optional): damping surface, either 'sphere', or 'cicle' 
            wind_damp(optional): if True, use fire_damp attribute for fire feature calculation instead of fire. If fire_damp hasn't exsited, calculate it. 
            wind_lag(optional): if True, use real wind speed for arrivial time 

        Returns: (list, list)
            fire_cols: list of the fire columns
            zone_list: a list of fire zone 
        
        """
        fire_col = 'power'
        if wind_damp or wind_lag:
            # use wind_damp fire default option
            # set self.fire_dict attribute 
            if fire_dict is None:
                print('use default fire feature')
                fire_dict = {'w_speed': 1, 'shift': -5, 'roll': 44, 'damp_surface':damp_surface, 'wind_damp': wind_damp, 'wind_lag': wind_lag }
                self.fire_dict = fire_dict

            # check if has damped_fire attribute
            if not hasattr(self, 'damped_fire'):
                #create the damped fire first 
                self.get_wind_damp_fire(wind_lag=wind_lag)
            # use damped fire attribute      
            fire_df = self.damped_fire
            if wind_damp:
                # keep only the columns with more than zero winddamp factor to reduce computation time
                fire_df = fire_df[fire_df['winddamp'] > 0]
                # damp the fire_col columns 
                fire_df[fire_col] = fire_df[fire_col]*fire_df['winddamp']
              
        else:

            # set self.fire_dict attribute
            if fire_dict is None:
                print('use default fire feature')
                fire_dict = {'w_speed': 7, 'shift': -5, 'roll': 44, 'damp_surface':damp_surface, 'wind_damp': wind_damp, 'wind_lag': wind_lag}
                self.fire_dict = fire_dict
            # use raw fire data        
            fire_df = self.fire

        # obtain processed fire dataframe and the fire columns     
        fire_proc, fire_cols = get_fire_feature(fire_df, zone_list=self.zone_list,
                                                fire_col=fire_col, damp_surface=damp_surface,
                                                shift=fire_dict['shift'], roll=fire_dict['roll'], w_speed=fire_dict['w_speed'])

        # merge with fire data
        data = self.data_no_fire.merge(
            fire_proc,
            left_index=True,
            right_index=True,
            how='inner')
        data = data.dropna()
        data = data.loc[~data.index.duplicated(keep='first')]
        self.data = data

        return fire_cols, self.zone_list

    def make_diff_col(self):
        """Add pollutant diff column for modeling the diff instead of the actual value.
        Drop all the columns with 'lag' name on it.

        The function update self.data_no_fire attribute, x_cols attribute, and monitor attribute.

        """
        # create diff columns
        new_col = self.pollutant + '_diff'
        self.data_no_fire[new_col] = self.data_no_fire[self.pollutant].diff()
        # add monitor col
        self.monitor = new_col

        # the first row of diff is nan. Add that value as attribute before
        # deleting the row.
        self.start_value = self.data_no_fire.iloc[0][self.pollutant]
        self.data_no_fire = self.data_no_fire.dropna()

        # find the columns to drop
        to_drop = self.data_no_fire.columns[self.data_no_fire.columns.str.contains(
            'lag_')]
        self.data_no_fire.drop(to_drop, axis=1, inplace=True)

    def split_data(
            self,
            split_ratio: list = [
                0.4,
                0.2,
                0.2,
                0.2],
            shuffle: bool = False):
        """Split the data datetime index into train, valiadation and test sets

        Add a list of the index in each set as atttibutes
        Args:
            split_ratio(optional): porportion of data in each set. Must add up to less than or equal to one.
            shuffle(optional): shuffle the data before splitting

        """
        # use the data after merge

        if np.sum(split_ratio) > 1:
            raise AssertionError(
                'The sum of the splitting ratios must not exceed 1')

        idxs = self.data.index
        if shuffle:
            numpy.random.shuffle(idxs)
        # find splitting indicies
        split_ratio = (np.array(split_ratio) * len(idxs)).astype(int)
        split_ratio = split_ratio.cumsum()
        self.split_list = np.split(idxs, split_ratio[:-1])

    def get_data_matrix(self, use_index: list, x_cols: list = []):
        """Extract data in data dataframe into x,y matricies using input index list.

        y is specified by self.monitor attribute. Use the data specified by x_cols.
        If x_cols is an empty list, use entire columns in self.data

        Args:
            use_index: a list of datetime index for the dataset
            x_cols(optional): a list of columns for x data [default:[]]

        Returns:
            x: x data matrix
            y: y data matrix
            x_cols: data columns

        """
        try:
            temp = self.data.loc[use_index]
        except BaseException:
            raise AssertionError(
                'no self.data attribute. Call self.merge_fire() first')

        y = temp[self.monitor].values

        if len(x_cols) == 0:
            x = temp.drop(self.monitor, axis=1)
        else:
            x = temp[x_cols]

        x_cols = x.columns.to_list()
        return x.values, y, x_cols

    def build_lag(self, lag_range: list, roll=True):
        """Build the lag data using number in lag_range.
        Add the new data as self.data attribute.

        Args:
            lag_range: list of lag value to add. Can be from np.arange(1,5) or [1,3, 10]
            roll(optional): if True, use the calculate the rolling average of previous values and shift 1

        """

        lag_list = [self.data_org]
        for n in lag_range:
            lag_df = self.data_org[self.x_cols_org].copy()
            lag_df.columns = [s + f'_lag_{n}' for s in lag_df.columns]
            if roll:
                # calculate the rolling average
                lag_df = lag_df.rolling(n, min_periods=None).mean()
                lag_df = lag_df.shift(1)
            else:
                lag_df = lag_df.shift(n)

            lag_list.append(lag_df)

        self.data = pd.concat(lag_list, axis=1, ignore_index=False)
        self.data = self.data.dropna()

    def save_(self):
        """Save the process data for fast loading without build.

        Save if the attribute exist.
        - save pollution data
        - save weather data
        - save data no fire
        - save data_org
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
            # save fire data
            if 'datetime' in self.wea.columns:
                # save without index
                self.wea.to_csv(self.data_folder + 'weather.csv', index=False)

            else:
                # save with index
                self.wea.to_csv(self.data_folder + 'weather.csv')

        if hasattr(self, 'data_no_fire'):
            # save fire data
            if 'datetime' in self.data_no_fire.columns:
                # save without index
                self.data_no_fire.to_csv(
                    self.data_folder + 'data_no_fire.csv', index=False)

            else:
                # save with index
                self.data_no_fire.to_csv(self.data_folder + 'data_no_fire.csv')

        if hasattr(self, 'data_org'):
            # save fire data
            if 'datetime' in self.data_no_fire.columns:
                # save without index
                self.data.to_csv(
                    self.data_folder + 'data_org.csv', index=False)

            else:
                # save with index
                self.data.to_csv(self.data_folder + 'data_org.csv')

        if hasattr(self, 'data'):
            # save fire data
            if 'datetime' in self.data_no_fire.columns:
                # save without index
                self.data.to_csv(
                    self.data_folder + 'data.csv', index=False)

            else:
                # save with index
                self.data.to_csv(self.data_folder + 'data.csv')

    def load_(self, fire='MODIS'):
        """Load the process pollution data from the disk without the build

        Load if the file exist for.
        - pollution data
        - weather data
        - data no fire
        - data_org
        - data

        """

        if os.path.exists(self.data_folder + 'poll.csv'):
            self.poll_df = pd.read_csv(self.data_folder + 'poll.csv')
            self.poll_df['datetime'] = pd.to_datetime(self.poll_df['datetime'])
            self.poll_df.set_index('datetime', inplace=True)
            # add pollution list
            self.gas_list = self.poll_df.columns.to_list()

            if (self.city_name == 'Chiang Mai') or (
                    self.city_name == 'Bangkok'):
                # for Thailand, delete all PM2.5 record before 2010
                self.poll_df.loc[:'2010', 'PM2.5'] = np.nan

        else:
            print('no pollution data. Call self.build_pollution first')

        if fire == 'MODIS':
            filename = self.data_folder + 'fire_m.csv'
        else:
            filename = self.data_folder + 'fire_v.csv'

        if os.path.exists(filename):
            self.fire = pd.read_csv(filename)
            self.fire['datetime'] = pd.to_datetime(self.fire['datetime'])
            self.fire.set_index('datetime', inplace=True)
        else:
            print('no fire data. Call self.build_fire first')

        if os.path.exists(self.data_folder + 'weather.csv'):
            self.wea = pd.read_csv(self.data_folder + 'weather.csv')
            try:
                self.wea.drop(['Time',
                               'Dew Point(C)',
                               'Wind Gust(kmph)',
                               'Pressure(in)',
                               'Precip.(in)'],
                              axis=1,
                              inplace=True)
            except BaseException:
                pass
            self.wea['datetime'] = pd.to_datetime(self.wea['datetime'])
            self.wea.set_index('datetime', inplace=True)
        else:
            print('no weather data. Call self.build_weather first')

        if os.path.exists(self.data_folder + 'data_no_fire.csv'):
            self.data_no_fire = pd.read_csv(
                self.data_folder + 'data_no_fire.csv')
            self.data_no_fire['datetime'] = pd.to_datetime(
                self.data_no_fire['datetime'])
            self.data_no_fire.set_index('datetime', inplace=True)

        if os.path.exists(self.data_folder + 'data_org.csv'):
            self.data = pd.read_csv(self.data_folder + 'data_org.csv')
            self.data['datetime'] = pd.to_datetime(
                self.data['datetime'])
            self.data.set_index('datetime', inplace=True)

        if os.path.exists(self.data_folder + 'traffic.csv'):
            self.traffic = pd.read_csv(self.data_folder + 'traffic.csv')
            self.traffic['datetime'] = pd.to_datetime(
                self.traffic['datetime'])
            self.traffic.set_index('datetime', inplace=True)

        if os.path.exists(self.data_folder + 'data.csv'):
            self.data = pd.read_csv(
                self.data_folder + 'data.csv')
            self.data['datetime'] = pd.to_datetime(
                self.data['datetime'])
            self.data.set_index('datetime', inplace=True)

