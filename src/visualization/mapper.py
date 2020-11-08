# -*- coding: utf-8 -*-
from ..imports import *
from ..gen_functions import *
import imageio
from ..features.dataset import Dataset
from ..features.build_features import add_wea_vec
from ..data.read_data import *


class Mapper():
    """Mapper object is in charge of visualizing pollution hotspots. 


    Args:
        main_folder(optional): main data folder for initializing Dataset object [default:'../data/]
        report_folder(optional): folder to save figure for initializing Dataset object [default:'../reports/']
        n_jobs(optional): number of CPUs to use during optimization
    
    Attributes:
        city: define the center city on the map
        Dataset: 
    
    
    """
    # a defaul list of pollutants
    gas_list = ['PM2.5', 'PM10', 'O3', 'CO', 'NO2', 'SO2']
    source_list = ['TH_PCD', 'TH_CMU', 'Berkeley', 'US_emb']
    # add color label attribute
    poll_colors = [ 'goldenrod', 'orange', 'red', 'purple', 'purple']
    plt_color = plt.rcParams['axes.prop_cycle'].by_key()['color']


    transition_dict =  {
        'PM2.5': [ 0, 12.0, 35.4, 55.4, 150.4, 250.4, 350.4, 500, 1e3], 
        'PM10': [0, 155, 254, 354, 424, 504, 604, 1e3], 
        'O3': [0, 54, 70, 85, 105, 200, 1e3], 
        'SO2': [0, 75, 185, 304, 504, 604, 1e3], 
        'NO2': [0, 53, 100, 360, 649, 1249, 2049, 1e4], 
        'CO': [0, 4.4, 9.4, 12.4, 15.4, 30.4, 40.4, 50.4, 1e3]} 

    def __init__(self,main_folder: str = '../data/', report_folder='../reports/', n_jobs=-2):

        self.main_folder = main_folder
        #  folder to save the process data 
        self.map_folder = main_folder + 'poll_map/'
        # folder to save the images 
        self.report_folder = report_folder + 'ani_plot/'
        self.image_folder = self.report_folder + 'map_images/'
        if not os.path.exists(self.image_folder):
            os.mkdir(self.image_folder)

    def build_b_station(self, folder='pm25/', prefix='berk', label='Berkeley'):
        """Compile Berkekely Earth data station and add nessary information.

        Args:
            folder(optional): name of the data folder
            prefix(optional): name of the stationid prefix 
            label(optional): source label for separating data source

        Returns: pd.DataFrame
            dataframe of the station information 

        """
        folder = self.main_folder + folder 

        # process load station info
        with open(folder + 'cities_info.json') as f:
            b_stations = json.load(f)

        b_stations = pd.DataFrame(b_stations)

        # obtain Berkeley data location information 
        # add mercadian coordinates in meter 
        b_stations['long_m'] = b_stations['Longitude'].apply(merc_x)
        b_stations['lat_m'] = b_stations['Latitude'].apply(merc_y, shift=True)
        # add station id (preparing to merge with other station jsons)
        b_stations['id'] = [f'{prefix}{num}' for num in np.arange(len(b_stations))]
        # add data source 
        b_stations['source'] = label
        b_stations['City'] = b_stations['City (ASCII)']

        return b_stations

    def build_pcd_station(self, folder='air4thai_hourly/', label='TH_PCD'):
        """Compile Thailand PCD monitoring stations and add nessary information.

        Args:
            folder(optional): name of the data folder
            label(optional): source label for separating data source

        Returns: pd.DataFrame
            dataframe of the station information 

        """
        folder =  self.main_folder + folder 
        
        # pcd stations information
        with open(folder + 'station_info.json', encoding="utf8") as f:
            station_infos = json.load(f)
    
        
        # get the station id_list
        pcd_stations = []
        for station in station_infos['stations']:
            if 'b' in station['stationID']:
                #remove stations run by Bangkok city 
                pass
            else:
                pcd_stations.append(station)
                
        pcd_stations = pd.DataFrame(pcd_stations)
        # change column name 
        pcd_stations.columns = pcd_stations.columns.str.replace( 'lat', 'Latitude')
        pcd_stations.columns = pcd_stations.columns.str.replace( 'long', 'Longitude')
        pcd_stations.columns = pcd_stations.columns.str.replace( 'stationID', 'id')
        
        # add mercadian coordinates
        pcd_stations['long_m'] = pcd_stations['Longitude'].apply(merc_x)
        pcd_stations['lat_m'] = pcd_stations['Latitude'].apply(merc_y,shift=True)
        
        # add city/country info (preparing to merge with other station jsons)
        pcd_stations['Country'] = 'Thailand'
        temp  = pcd_stations['areaEN'].str.split(',', expand=True)
        pcd_stations['City'] = temp[2].fillna(temp[1])
        # add data source 
        pcd_stations['source'] = label
        return pcd_stations

    def build_cmu_station(self,folder='cdc_data/', label='TH_CMU'):
        """Compile Chiang Mai University pollution project's stations  and add nessary information.

        Args:
            folder(optional): name of the data folder
            label(optional): source label for separating data source

        Returns: pd.DataFrame
            dataframe of the station information 

        """
        folder =  self.main_folder + folder

        # load Chiang Mai University pollution project's stations 
        # pcd stations information
        with open(folder + 'station_info.json') as f:
            cmu_stations = json.load(f)
            
        cmu_stations = pd.DataFrame(cmu_stations)
        cmu_stations['dustboy_id'] = cmu_stations['dustboy_id'].astype(int)
        
        # change column name 
        cmu_stations.columns = cmu_stations.columns.str.replace( 'dustboy_lat', 'Latitude')
        cmu_stations.columns = cmu_stations.columns.str.replace( 'dustboy_lng', 'Longitude')
        cmu_stations.columns = cmu_stations.columns.str.replace( 'dustboy_id', 'id')
        
        # add mercadian coordinates
        cmu_stations['long_m'] = cmu_stations['Longitude'].apply(merc_x)
        cmu_stations['lat_m'] = cmu_stations['Latitude'].apply(merc_y, shift=True)
        
        # add city/country info (preparing to merge with other station jsons)
        cmu_stations['Country'] = 'Thailand'
        temp  = cmu_stations['dustboy_name_en'].str.split(',', expand=True)
        cmu_stations['City'] = temp[2].fillna(temp[1])
        cmu_stations['City'] = cmu_stations['City'].fillna(temp[0])
        # add data source 
        cmu_stations['source'] = label 
        return cmu_stations

    def build_usemb_station(self, folder='us_emb/', label='US_emb'):
        """Compile location of US Embassy with pollution data in ASEAN and add nessary information.

        Args:
            folder(optional): name of the data folder
            label(optional): source label for separating data source
    
        Returns: pd.DataFrame
            dataframe of the station information 
    
        """
        folder =  self.main_folder + folder
        usemb_stations = pd.read_csv(folder+'station_info.csv')
        # add mercadian coordinates
        usemb_stations['long_m'] = usemb_stations['Longitude'].apply(merc_x)
        usemb_stations['lat_m'] = usemb_stations['Latitude'].apply(merc_y, shift=True)

        # add data source 
        usemb_stations['source'] = label 
        return usemb_stations

    def build_station_info(self):
        """Compile pollution monitoring station informations from all 4 data sources

        1. Berkekely Earth database 
        2. air4thai database http://air4thai.pcd.go.th/webV2/
        3. CMU project databse cmuccdc.org
        4. US Embassy database https://www.airnow.gov/international/us-embassies-and-consulates/

        """
        
        all_station_info = []
        all_station_info.append(self.build_pcd_station())
        all_station_info.append(self.build_cmu_station())
        all_station_info.append(self.build_b_station())
        all_station_info.append(self.build_usemb_station())

        # concadinate all station_info
        all_station_info = pd.concat(all_station_info, ignore_index=True)
        print('number of stations =', all_station_info.shape)
        all_station_info.to_csv(self.map_folder + 'all_station_info.csv', index=False)
        self.all_station_info = all_station_info

    def build_pcd_data(self):
        """Concatenate all PCD data and save 

        Returns: list
            columns of the saved dataframe to meke sure other save are of this order 

        """
        temp_stations = self.all_station_info[self.all_station_info['source'] =='TH_PCD']

        all_station_data = []
        # compile all PCD data 
        for i, row in tqdm(temp_stations.iterrows()):
            stationid = row['id']
            
            filename = self.main_folder + f'aqm_hourly_final/{stationid}.csv'
            if os.path.exists(filename): 
                df = pd.read_csv(filename)
            else:
                print(filename, 'does not exist')
                df = pd.DataFrame()

            df['stationid'] = stationid    
            all_station_data.append(df)

        all_station_data = pd.concat(all_station_data)
        print('pcd data shape', all_station_data.shape)
        all_station_data.to_csv(self.map_folder + 'data.csv', index=False)

        return all_station_data.columns

    def build_cmu_data(self, data_columns):
        """Concatenate all PCD data and append to exisiting data file
        
        Args:
            data_columns: data columns name to guide the concatenation 

        """
        temp_stations = self.all_station_info[self.all_station_info['source'] =='TH_CMU']

        all_station_data = [pd.DataFrame(columns=data_columns)]
        # compile all PCD data 
        for i, row in tqdm(temp_stations.iterrows()):
            stationid = row['id']
             
            filename = self.main_folder + f'cdc_data/{stationid}.csv'
            #print(filename)
            df = read_cmucdc(filename)
            if len(df)> 0:
                df['stationid'] = stationid    
                all_station_data.append(df)

        all_station_data = pd.concat(all_station_data)
        print('cmu data shape', all_station_data.shape)
        all_station_data.to_csv(self.map_folder + 'data.csv', mode ='a', index=False, header=False)
         
    def build_b_data(self, data_columns):
        """Concatenate all Berkeley data and append to exisiting data file 
        
        Args:
            data_columns: data columns name to guide the concatenation 

        """
        temp_stations = self.all_station_info[self.all_station_info['source'] =='Berkeley']

        all_station_data = [pd.DataFrame(columns=data_columns)]
        # compile all PCD data 
        for i, row in tqdm(temp_stations.iterrows()):
            stationid = row['id']
             
            filename = self.main_folder + f'pm25/' + row['City'].replace(' ', '_') + '.txt'
            #print(filename)
            df, _ = read_b_data(filename)
            if len(df)> 0:
                df['stationid'] = stationid    
                all_station_data.append(df)

        all_station_data = pd.concat(all_station_data)
        print('Berkeley data shape', all_station_data.shape)
        all_station_data.to_csv(self.map_folder + 'data.csv', mode ='a', index=False, header=False)
 
    def build_usemb_data(self, data_columns):
        """Concatenate all US Embassy data and append to exisiting data file 
        
        Args:
            data_columns: data columns name to guide the concatenation 

        """
        temp_stations = self.all_station_info[self.all_station_info['source'] =='US_emb']

        all_station_data = [pd.DataFrame(columns=data_columns)]
        # compile all PCD data 
        for i, row in tqdm(temp_stations.iterrows()):
            stationid = row['id']
             
            df = build_us_em_data(city_name=row['City'], data_folder=f'{self.main_folder}us_emb/')
            df = df[0]
            if len(df)> 0:
                df['stationid'] = stationid    
                all_station_data.append(df)

        all_station_data = pd.concat(all_station_data)
        print('US Embassy data shape', all_station_data.shape)
        all_station_data.to_csv(self.map_folder + 'data.csv', mode ='a', index=False, header=False)
 
    def build_pollution_all(self):
        """compile all pollution data into a single file and save the file. 
        
        The pollution data from all stations are too large to fit in a memory, so I have to build in a separate section and append to the old file. 
        I have to make sure that the columns order are the same to avoid putting pollution data into a wrong column.
        
        """
        # check if self.all_station_info attribute exist
        if not hasattr(self, 'all_station_info'):
            # no station information. Call load_()
            self.load_()

        # update pcd data, which exists in three different folder. 
        # making use of the parsing function in the dataset object 
        dataset = Dataset('Bangkok')
        temp_stations = self.all_station_info[self.all_station_info['source'] =='TH_PCD']
        dataset.merge_new_old_pollution(temp_stations['id'].to_list())

        # build PCD data first because it has all the pollutants
        # columns order is returned and used for building other data sources
        data_columns =  self.build_pcd_data()
        print('data_columns', data_columns)
        self.build_cmu_data(data_columns)
        self.build_b_data(data_columns)
        self.build_usemb_data(data_columns)
        
    def set_map_params(self, center_city:str, **kwargs):
        """Build map setting and store as dictionary 
        
        Args:
            center_city: name of the city at the center of the map. Must have station in the Berkeley database
            kwargs: optional keyword argument to overide the default setting 
                Possible keywords are 
                city_x: Longitude of the map center in mercator coordinate (meter)
                city_y: Latitude of the map center in mercator coordinate (meter)
                stepx and xmap_range: define the width of the map
                stepy and ymap_range: define the width of the map
                xmap_range: is a list xmap_range[0]*stepx defines the distance from the map center to the left and 
                            xmap_range[1]*stepx defines the distance from the map center to the left.
                inter_range: is a list which defines the pollution interpollation map 
                gridsize: meshgrid size in meter 
                plot_height: height of the line plot of the top of the map.



        """
        # obtain location of the city center
        row = self.all_station_info[(self.all_station_info['source']=='Berkeley') & (self.all_station_info['City']==center_city)]
        self.center_city = center_city
        map_dict = { 'city_x': row['long_m'].values[0].round(2),
                     'city_y': row['lat_m'].values[0].round(2),
                        'stepx': 5E5,
                        'stepy': 5E5,
                        'xmap_range': [0.8, 1.2],
                        'ymap_range': [0.3, 0.8],
                        'inter_range': [2, 2.5], 
                        'gridsize': 5E3,
                        'plot_height': 200,
                        'center_city': center_city}

        # update new setting 
        map_dict.update(kwargs)

        # calculate the xrange and update to mapset_dict
        map_range = [map_dict['city_x'] - map_dict['xmap_range'][0]*map_dict['stepx'], 
                    map_dict['city_x'] + map_dict['xmap_range'][1]*map_dict['stepx']]

        map_dict['xmap_range'] = map_range

        # calculate the yrange and update to mapset_dict
        map_range = [map_dict['city_y'] - map_dict['ymap_range'][0]*map_dict['stepy'], 
                    map_dict['city_y'] + map_dict['ymap_range'][1]*map_dict['stepy']]

        map_dict['ymap_range'] = map_range

        # calculate the xrange for interpolation map
        map_range = [map_dict['city_x'] - map_dict['inter_range'][0]*map_dict['stepx'], 
                    map_dict['city_x'] + map_dict['inter_range'][1]*map_dict['stepx']]

        map_dict['xinter_range'] = map_range

        # calculate the yrange and update to mapset_dict
        map_range = [map_dict['city_y'] - map_dict['inter_range'][0]*map_dict['stepy'], 
                    map_dict['city_y'] + map_dict['inter_range'][1]*map_dict['stepy']]

        map_dict['yinter_range'] = map_range
        
        # add as attribute 
        self.map_dict = map_dict
        
    def get_poll_by_daterange(self, start_date:str, end_date:str, chunksize=1E7):
        """Extract pollution data between start_date and end_date to construct pollution map
        
        Read the data in chunk to avoid memory error. Add coordinate information by mergging with self.all_station_info dataframe attribute.
        
        Args:
            start_date: first date of the desire data. Either string or datetime object for .loc command
            end_date: the last date of the desire data. Either string or datetime object for .loc command 
            chunksize(optional): chunksize to extract the data
            
        Returns:pd.DataFrame
        
        """ 
        self.start_date = start_date
        self.end_date = end_date
        filename = self.map_folder+ 'data.csv'
        polldata = []
        # read data in chunk because all data has large size 
        for df in pd.read_csv(filename, chunksize=chunksize ):
            df['datetime'] = pd.to_datetime(df['datetime']) 
            df = df.set_index('datetime')
            polldata.append(df.loc[start_date:end_date])
        
        polldata = pd.concat(polldata, ignore_index=False)
        polldata = polldata.reset_index()
        # add location information 
        polldata = polldata.merge(self.all_station_info[['id','Latitude', 'Longitude', 'long_m', 'lat_m']], left_on='stationid', right_on='id')
        # add date information 
        polldata['date'] = polldata['datetime'].dt.strftime('%Y-%m-%d')
        
        self.polldata = polldata

    def avg_city_poll(self):
        """Take the average of the pollution level of self.city_center 

        """
        # extract stationid
        stationid_list = self.all_station_info[(self.all_station_info['City'] == self.center_city) | (self.all_station_info['areaEN'].str.contains(self.center_city))]
        stationid_list = stationid_list['id'].to_list()
        # average pollution level among the those in stationid_list
        self.avg_polldata = self.polldata[self.polldata['stationid'].isin(stationid_list)]
        self.avg_polldata  = self.avg_polldata.groupby('datetime').mean()

    def inter_pollution(self, datetime ):
        """Interpolate pollution data form different station to form a continuous grid
        
        Use scipy.interpolate.Rbf for interpolation 
        
        Args: 
            datetime: datetime to generate the map for 
           
           
        Return pd.DataFrame
            dataframe with columns 'long_km', 'lat_km' and pollution value 
            
        """
        

        #select the data and pollutant on that date for grid interpolation  
        df = self.polldata [self.polldata ['datetime']==datetime ]
        # create interpolation x, y, z
        df = df.dropna()
        x_coor = df['long_m'].values
        y_coor = df['lat_m'].values
        values = df[self.pollutant].values

        gridsize = self.map_dict['gridsize'] 
        
        rbfi = Rbf(x_coor, y_coor, values , function='linear')
        x = np.arange(self.map_dict['xinter_range'][0], self.map_dict['xinter_range'][1], gridsize)
        y = np.arange(self.map_dict['yinter_range'][0], self.map_dict['yinter_range'][1], gridsize)
        xx, yy = np.meshgrid(x,y)
        xx  = xx.flatten()
        yy = yy.flatten()
    
        # calculate the interpoluation
        z =  rbfi(xx, yy)
    
        return pd.DataFrame({'long_m': xx, 
                      'lat_m':yy, self.pollutant : z})

    def set_pollutant_cbar(self, pollutant, cmap=cm.RdYlGn):
        """Add self.pollutant and self.colorbar attribute for building colormap. 

        Args:
            pollutant: name of the pollutant
            cmap: plt colormap 

        """

        self.pollutant = pollutant

        #poll = 'NO2'
        if pollutant == 'PM2.5':
            # determine the colormap maximum
            poll_max = (self.transition_dict[pollutant][3] + self.transition_dict[pollutant][4])*0.5
            
        elif pollutant =='NO2':
            # determine the colormap maximum
            poll_max = self.transition_dict[pollutant][1]  

        # Bokeh doesn't have its own gradient color maps supported but you can easily use on from matplotlib.
        # use the same colormap for all figure, so it is outsize the for loop
        colors = get_color(color_length=11, cmap = cmap) 
        colors.reverse()
        # remove some green colors 
        colors = [ c for c in colors if c not in colors[0:4]]
        #this mapper is what transposes a numerical value to a color. 
        self.mapper = LinearColorMapper(palette=colors, low=0, high=poll_max)

    def add_poll_colormap(self, datetime):
        """Add pollution color map to the plot

        Args: 
            p: bokeh figure object
            datetime: datetime to generate the map for 
            poll: pollution name 

        """
        # add basemap 
        p = plot_basemap(self.map_dict['xmap_range'], self.map_dict['ymap_range'])

        # get pollution interpoluation matrix 
        df = self.inter_pollution(datetime)
        dotsize = self.map_dict['gridsize']

        p.rect(x="long_m", y="lat_m", width=dotsize, height=dotsize,
        source=df, fill_color={'field': self.pollutant, 'transform': self.mapper}, line_color=None, alpha=0.6)
        
        # set colorbar property 
        color_bar = ColorBar(color_mapper=self.mapper, major_label_text_font_size="10pt",
                             ticker=BasicTicker(desired_num_ticks=8),
                             label_standoff=6, border_line_color=None, location=(0, 0))

        p.add_layout(color_bar, 'right')

        #text = Label(x=self.map_dict['xmap_range'][1] , y=self.map_dict['ymap_range'][0]-self.map_dict['stepy']*0.5, text=' Datetime: '+str(datetime), text_font_size='15pt', text_color='black', background_fill_color='white', text_align='right')
        #p.add_layout(text)
        text = Title(text=' Datetime: '+str(datetime), align="left", text_color='black', text_font_size='12px')
        p.add_layout(text, "above")

        p.grid.grid_line_color = None

        return p

    def add_line_plot(self, title=None, color='blue', legend_loc='bottom_right' ):
        """Make time series line plot of the pollution of the center city
        
        Args:
            poll_df: pollution dataframe with datetime index 
            poll: pollution name
        
        Returns: Bokeh figure object, pd.DataFrame of 24 hour moving average 
        
        """

        p = figure(plot_height=self.map_dict['plot_height'], x_axis_type='datetime', toolbar_location=None, title=title )    
        p.circle(self.avg_polldata.index, self.avg_polldata[self.pollutant],line_width=1, line_color=color, fill_color=color)
        
        moving_avg = self.avg_polldata[self.pollutant].rolling(24, min_periods=1 ).mean()
        
        p.line(moving_avg.index, moving_avg.values, line_width=3, line_color='dodgerblue', line_dash='dashed' )
    
        #p.xaxis.axis_label = 'date'
        p.yaxis.axis_label = get_unit(self.pollutant)
        p.legend.location = legend_loc
        
        return p

    def show_station_loc(self):
        """Show station location by station types 

        """
        
        # extract stationid
        stationid_list = self.polldata['stationid'].unique().tolist()
        
        current_stations = self.all_station_info[self.all_station_info['id'].isin(stationid_list)]
         
        # add basemap 
        p = plot_basemap(self.map_dict['xmap_range'], self.map_dict['ymap_range'])
        temp = current_stations[current_stations['source']=='Berkeley']
        # add Berkeley data stations information 
        p.square(temp['long_m'], temp['lat_m'], legend_label='Berkeley data', color=self.plt_colors[3], line_color='black',size=8)

        temp = current_stations[current_stations['source']=='TH_PCD']
        # add PCD data stations information 
        p.triangle(temp['long_m'], temp['lat_m'], legend_label='TH_PCD stations', color=self.plt_colors[0], line_color='black',size=8)

        temp = current_stations[current_stations['source']=='TH_CMU']
        # add PCD data stations information 
        p.circle(temp['long_m'], temp['lat_m'], legend_label='CMU stations', color=self.plt_colors[1], line_color='black',size=8)

        temp = current_stations[current_stations['source']=='US_emb']
        # add PCD data stations information 
        p.diamond(temp['long_m'], temp['lat_m'], legend_label='US Embassy', color=self.plt_colors[2], line_color='black',size=10)

        p.legend.location = "bottom_center"

        show(p)

    def add_fire(self, p, fire, datetime, duration=48, size=5):
        """Add burning hotspot on the basemap 

        Args:
            p: bokeh figure object
            fire: fire dataframe containing pollution location. Must have datetime index
            datetime: datetime of the last hotspot
            duration: amount of time in the past hours to include the hotspots
            size: size of each hotspot

        """
        # extract fire data
        start_datetime = pd.to_datetime(datetime) - pd.to_timedelta(duration,'hours')

        df = fire.loc[start_datetime:datetime]
        p.circle(df['long_km']*1000,df['lat_km']*1000,color='red',size=size, alpha=0.5, legend_label=f'hotspots {duration} hrs')

    def add_wind(self, p, wind, datetime, scale=10000):
        """Add wind vector on the map

        Args:
            p: bokeh figure object
            wind: wind direction dataframe, must have datetime index. 
            datetime: datetime of the last hotspot
            scale: vector scaling factor 

        """

        wind_row = wind.loc[datetime]
        # add vector 
        x_start = self.map_dict['city_x']
        y_start = self.map_dict['city_y']
        x_end = self.map_dict['city_x'] + wind_row['wind_vec_x']*scale
        y_end = self.map_dict['city_y'] + wind_row['wind_vec_y']*scale

        p.add_layout(Arrow(end=VeeHead(size=20, fill_color= 'lightgray'), line_color="lightgray", line_width= 4,
                           x_start=x_start, y_start=y_start, x_end=x_end, y_end=y_end))


    def get_datasamples(self, start_date, end_date, peak=True, freq:(str, int)='6H'):
        """Obtain the datetime sample, which are the datetime to plot the pollution map. 
        
        Plotting the pollution map every hour will be too much. 
        The hourly pollution fluctuation are highly influence by the effect of the wind. 
        I decided to pick a subset of hourly data. I choose to plot the data at the peak pollution to show.

        Two options:
            1. Use peak detection to plot the datetime at the pollution peak
            2. Plot every constant time interval specified by freq input

        Add data_samples as attribute
        
        Args:
            start_date: start date of the data  
            end_date: last date of the data 
            peak: if True, use peak detection to get the time sample 
            freq: if peak=True, freq is interpret as distance between two peaks. If peak==False,  freq is a string and determine a constant interval between two samples.

        Raises:
            AssertionError if self.avg_polldata doesn't exist. 

        """
        if not hasattr(self, 'avg_polldata'):
            raise AssertionError('need avg_polldata attribute. Call self.avg_city_poll() first')
        
        if peak:
            # extract pollution arr
            poll_arr = self.avg_polldata[self.pollutant].values
            peaks, peak_dict = find_peaks(poll_arr, distance=freq, prominence=2)
            self.data_samples = self.avg_polldata.iloc[peaks]
        else:
            idxs = pd.date_range(start_date, end_date, freq=freq) 
            self.data_samples = self.avg_polldata.loc[idxs]

        # check the lenght of the sample to prevent too many datapoint 

        if len(self.data_samples) > 200:
            print('WARNING: too many plots will make the gif file too large')

        # add pollution level label and colors
        self.data_samples['level'] = pd.cut(self.data_samples[self.pollutant], bins=self.transition_dict[self.pollutant]).cat.codes
        self.data_samples['color'] = [self.poll_colors[i] for i in self.data_samples['level']]

        self.data_samples = self.data_samples[[self.pollutant, 'level','color']]

    def plot_datasample(self, p):
        """Plot the self.data_samples attribute on the line plot. The plot is color code. 

        Args:
            p: bokeh figure object

        """
        for color in self.data_samples['color'].unique():
            temp = self.data_samples[self.data_samples['color']==color]
            p.circle(temp.index, temp[self.pollutant],line_width=1, line_color=color, fill_color=color)

    def delete_old_images(self):
        """Delete old images file in the folder 

        """
        # delete old png files
        files = glob(self.image_folder + '*.png')
        for file in files:
            os.remove(file)

    def build_ani_images(self, start_date, end_date, pollutant, add_title='', fire=[], wind=[], delete=False):
        """Make and save animation images for building gif file 
        
        Args:
            start_date: start date of the data  
            end_date: last date of the data 
            pollution: pollution name
            add_title: string to append to the plot title
            fire: fire dataframe
            wind: wind direction and speed 
            delete: if True, delete old png file before building a new one

        Return: list
            a list of filename to build the gif 

        """
        if delete:
            self.delete_old_images()

        title =  f'{self.pollutant} map from {start_date} to {end_date}' + add_title  
        # set colorbar
        self.set_pollutant_cbar(pollutant=pollutant)
        
        filenames = []
        for i, datetime in enumerate(self.data_samples.index):
            
            # add line plot 
            p1 = self.add_line_plot(title=f'{self.pollutant} level in {self.center_city}')
            self.plot_datasample(p1)

            # add vertical line 
            color = self.data_samples.loc[datetime, 'color']
            vline = Span(location=datetime, dimension='height', line_color=color, line_dash='dashed', line_width=2)
            p1.add_layout(vline)

            # add colormap
            p2 = self.add_poll_colormap(datetime)
            
            if len(fire) !=0:
                self.add_fire(p2, fire, datetime)

            if len(wind) != 0:
                self.add_wind(p2, wind, datetime)

            p = column(Div(text=f'<h3>{title}</h3>'), p1, p2)

            filename = self.image_folder + f"{self.pollutant}_{start_date}_{end_date}_{i}.png"
            filenames.append(filename)
            export_png(p, filename=filename)

        return filenames

    def make_ani(self, start_date, end_date, pollutant:str, add_title:str='', fire=[], wind=[], delete=False, peak=True, freq=4, duration=1):
        """ Create pollution map for each datasample date, save and used it to construct gif animation.

        Args:
            start_date: start date of the data  
            end_date: last date of the data 
            pollution: pollution name
            add_title: string to append to the plot title
            fire: fire dataframe
            wind: wind direction and speed 
            delete: if True, delete old png file before building a new one
            peak: if True, use peak detection to get the time samples
            freq: if peak=True, freq is interpret as distance between two peaks. If peak==False,  freq is a string and determine a constant interval between two samples.
            duration: duration of each image in second 

        Return: str
            filename of the gif image 

        """
        if not hasattr(self, 'polldata'):
            # preparing the data 
            self.get_poll_by_daterange(start_date, end_date)

        if not hasattr(self, 'avg_polldata'):
            # preparing the data 
            self.avg_city_poll()
        
        self.set_pollutant_cbar(pollutant=pollutant)
        # get datasamples
        self.get_datasamples(start_date, end_date, peak=peak, freq=freq) 
        if len(wind):
            wind = wind.loc[start_date:end_date]
            wind = add_wea_vec(wind, daily_avg=False,roll_win=6)
            #multiply wind vector with the wind speed 
            wind['wind_vec_x'] *= wind['Wind_Speed(kmph)']
            wind['wind_vec_y'] *= wind['Wind_Speed(kmph)']
            wind = wind.fillna(0)

        filenames = self.build_ani_images(start_date, end_date, pollutant, add_title=add_title, fire=fire, wind=wind, delete=delete)
        
        # create a gif showing pollution level for each month
        images = []
        for filename in filenames:
            images.append(imageio.imread(filename))

        ani_filename = self.report_folder + f'{self.pollutant}_{start_date}_{end_date}.gif'
        imageio.mimsave(ani_filename, images, duration=duration)

        return ani_filename


    def load_(self):
        """Load compiled station information and add as attribute. 

        Raises:
            AssertionError if station information file doesn't exist. 

        """
        if os.path.exists(self.map_folder + 'all_station_info.csv'):
            self.all_station_info = pd.read_csv(self.map_folder + 'all_station_info.csv')
            #print('number of stations =', self.all_station_info.shape)
        else:
            raise AssertionError('no station information file')


def plot_basemap(xmap_range, ymap_range, title=None, toolbar_location=None):
    """Add country map 


    Args:
        xrange: width of the map in mercator coordinate.
        yrange: height of the map in mercator coordinate. 
        title: title of the plot
        toolbar_location: location of the toolbar

    Returns: Bokeh plot object
    
    """
    
    
    p = figure(x_range=xmap_range, y_range=ymap_range,
           x_axis_type="mercator", y_axis_type="mercator",  toolbar_location=toolbar_location, 
          title = title)

    p.add_tile(get_provider(Vendors.STAMEN_TERRAIN_RETINA))
    
    wmark_bokeh(p)

    return p