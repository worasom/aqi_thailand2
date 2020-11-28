# -*- coding: utf-8 -*-
import os
import sys
from pyproj import CRS
from pyproj import Transformer
import gdal
import re
from math import ceil, floor
import swifter
import geopandas as gpd
from shapely.geometry import Polygon, MultiPoint, Point, MultiPolygon



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

"""Function to analyse landuse data 

"""

def load_gl(filename):
    """load gl data from a filename.
    
    Returns a tuple of (gl, lc_dict, label_dict)
        gl: gl object
        lc_dict: dictionary mapping the name to the gl index 
        label_list: a list of land label_dict. Only exist for LC_type1, LC_type5, LC_Prop2
        
    """
    gl = gdal.Open(filename)
    
    # define my own labels for lc_prop2
    lc2label =  {3: 'water',
           255: 'unclassified',
           9: 'urban',
           25:'crop',
           35: 'crop',
            36: 'crop',
           10:'forest',
           20:'forest',
           40: 'shrubland'}
    
    # define my own labels for lc_type5
    lc5label = {0: 'water',
           -1: 'unclassified',
           255: 'unclassified',
           9: 'urban',
           8:'crop',
           7: 'crop',
           4:'forest',
           3: 'forest',
           2: 'forest',
           1: 'forest',
           6: 'shrubland',
           11:'shrubland',
           5: 'shrubland'}
    # define my own labels for lc_type1
    lc1label = {17: 'water',
           -1: 'unclassified',
           255: 'unclassified',
           13: 'urban',
           12:'crop',
           14: 'crop',
           5:'forest',
           4:'forest',
           3: 'forest',
           2: 'forest',
           1: 'forest',
            8: 'forest',
           10: 'shrubland',
            9: 'shrubland',
           16: 'shrubland',
           6: 'shrubland',
           7: 'shrubland'}
    
    # build lc_dict to link the name to the index 
    lc_dict = {}
    label_list = []
    for i, item in enumerate(gl.GetSubDatasets()):
        k = item[0].split(':')[-1]
        lc_dict[k] = i
        
        if i == 2:
            label_list.append(lc2label)
        elif i == 6:  
            label_list.append(lc1label)
        elif i == 10:  
            label_list.append(lc5label) 
        else:
            label_list.append({})
            
    return (gl, lc_dict, label_list)
    


def get_lc_data(lc_name, year, gl, lc_dict, label_list):
    """obtain the data object for a specified lc_name and year, and label_dict 
    
    Args:
        lc_name:
        year: 
        gl:
        lc_dict
        label_list
        
    Returns 
        lc  
        band_index
        label_dict 
    
    """
    lc_index = lc_dict[lc_name]

    if lc_index not in [2,6, 10]:
        raise AssertionError('the specified lc does not have a label_dic')

    # obtain the filename
    lc_file = gl.GetSubDatasets()[lc_index][0]
    # open the dataobject 
    lc =  gdal.Open(lc_file)
    label_dict = label_list[lc_index]
    label_dict = pd.Series(label_dict, name= lc_name + '_label')
    label_dict = pd.DataFrame(label_dict)
    size_str = gl.GetSubDatasets()[lc_index][1]
    size_str = re.findall( '(\d+x\d+x\d+)', size_str)[0]
    num_band, ypixs, xpixs = size_str.split('x')
    
    # extract the band 
    # minium is 1
    band_index = max(1, int(year - 2001))
    # maximum is equal to num_band 
    band_index = min(band_index, int(num_band))
    
    
    return lc, band_index, label_dict 
    
def add_sinu_col(df, lat_col='latitude', long_col='longitude', unit='m'):
    """Add MODIS sinusocial coodinate column to a dataframe given latitude and longitude columns 
    
    Args:
        df: dataframe with latitude and longitud
        lat_col: name of the latitude columns
        long_col: name of the longitude columns
        unit: unit of the new column can be meter or km 
        
    Return: pd.DataFrame

    """
    crs_9112 = CRS.from_proj4("+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +a=6371007.181 +b=6371007.181 +units=m +no_defs")

    transformer = Transformer.from_crs("EPSG:4326", crs_9112) 
    
    return df.swifter.apply(to_merc_row, axis=1, transformer=transformer, lat_col=lat_col, long_col=long_col)

def get_label_row(row, lc_name, data_arr, buffer_pix=5):
    """Look up landuse data around the buffer pixels(buffer_pix). 
    
    Args:
        row: a single row with xpix and ypix 
        lc_name: lc_name to condition for crop, forrest 
        data_arr: landuse data array
        buffer_pix: buffer area (depends on the resolution of the data) I use 5 because the fire data has resolution of 1km, which about 2 time the landuse resolution.
    
    Return  a single label biased toward crop, forrest, and scrupland. 
    
    """
    
    x = np.arange(row['xpix'], row['xpix']+buffer_pix)
    x = x[x < data_arr.shape[1]]
    y = np.arange(row['ypix'], row['ypix']+buffer_pix)
    y = y[y < data_arr.shape[0]]
    xx, yy = np.meshgrid(x, y)
    xx = xx.flatten()
    yy = yy.flatten()
    labels = data_arr.take(yy, axis=0)
    labels = labels.take(xx, axis=1)
    labels = np.diagonal(labels)
     
    
    if lc_name == 'LC_Prop2':
        if (25 in labels) or (35 in labels) or (36 in labels):
            label = 25
        elif 40 in labels:
            label = 40

        else:
            label = np.bincount(labels).argmax()
    
    elif lc_name == 'LC_Type1':
        
        if (12 in labels) or (14 in labels):
            label = 12
        elif (6 in labels) or (7 in labels) or (9 in labels) or (10 in labels) or (16 in labels):
            label = 6
       
        else:
            label = np.bincount(labels).argmax()
    
    elif lc_name == 'LC_Type5':
        if (8 in labels) or (7 in labels):
            label = 8
        elif (6 in labels) or (11 in labels) or (5 in labels) :
            label = 6
       
        else:
            label = np.bincount(labels).argmax()
            
    else: 
        label = np.bincount(labels).argmax()
        
    return label 

    
def get_label(df, lc, lc_name, band_index, chunk=1000, buffer_pix=5):
    """Add lc land label to the dataframe using the lat_km and long_km columns. 

    Args:
        df: subset of fire dataframe 
        lc: landuse dataset object
        lc_name: name for the column
        band_index: band_index from the year information
        chunk: chunk size of the sattelite array 
        buffer_pix: number of pixel around area. Should be odd number. 

    Return a series of labeled data. The colum name is the lc name  

    """

    # extract geometrical property and image size

    GT = lc.GetGeoTransform()
    xsize = lc.RasterXSize
    ysize = lc.RasterYSize 

    # obtain the pixel and add buffer_pix offset 
    df['xpix'] = round((df['long_m'] - GT[0])/GT[1]).astype(int) - floor(buffer_pix/2)
    # divide the size of the grid into four section 
    df['ypix'] = round((df['lat_m'] - GT[3])/GT[5]).astype(int) - floor(buffer_pix/2)

    # cannot load everything into the memory, so only load the land label in chunk 
    max_iter_y = ceil(ysize/chunk)

    label_df = []

    for j in np.arange(max_iter_y):
        xoff = 0
        yoff = int(chunk*j)
        win_xsize = xsize
        win_ysize = chunk

        # correct the size of the last chunk

        if yoff + win_ysize > ysize:
            win_ysize = ysize - yoff

         
        # extract subdata for that chunk 
        mask = ((df['ypix'] >= yoff) &  (df['ypix'] < yoff+win_ysize))
        # keep only nessary info 
        sub_df = df.loc[mask, ['ypix', 'xpix']]

        if len(sub_df) > 0:
            sub_df['ypix'] -= yoff
            data = lc.GetRasterBand(band_index) 
            data_arr = data.ReadAsArray(xoff=xoff, yoff=yoff, win_xsize=win_xsize, win_ysize=win_ysize)
            labels = sub_df.swifter.apply(get_label_row, axis=1, lc_name=lc_name,data_arr=data_arr)
            sub_df[lc_name] = labels
            # obtain the labels
            label_df.append(sub_df[[lc_name]])

    label_df = pd.concat(label_df, ignore_index=False)

    return label_df


def label_landuse_fire(data_folder, landuse_file='../data/landuse_asean/MCD12Q1.006_500m_aid0001.nc', instr='MODIS', fire_chunk=1E5, lc_list= ['LC_Prop2', 'LC_Type1', 'LC_Type5']):
    """Load fire data and satellite data in chunk to prevent out of memory error, add different label types and save as original filename + label.csv

    """
    if instr == 'MODIS':
        filename = data_folder + 'fire_m.csv'
        save_filename = data_folder + 'fire_m_label.csv'

    elif instr == 'VIIRS':
        filename = data_folder + 'fire_v.csv'
        save_filename = data_folder + 'fire_v_label.csv'

    else:
        raise AssertionError('no fire data')

    # remove old file

    if os.path.exists(save_filename):
        os.remove(save_filename)

    gl_prop = load_gl(landuse_file)
    # keep some columns
    cols = ['datetime', 'latitude', 'longitude', 'distance']
    lc_list = ['LC_Prop2', 'LC_Type1', 'LC_Type5']


    for fire in tqdm(pd.read_csv(filename, chunksize=fire_chunk)):

        fire['datetime'] = pd.to_datetime(fire['datetime'] )
        fire = fire[cols]

        # add sinusodal coordinate 
        fire = add_sinu_col(fire)
        years = fire['datetime'].dt.year.unique() 

        for lc_name in lc_list:

            label_all = []
            for year in years:
                # extract fire for that year 
                sub_fire = fire[fire['datetime'].dt.year==year]
                # obtain lc, band_index for that year, label, and coordinate information 
                lc, band_index, label_dict = get_lc_data(lc_name, year, *gl_prop)
                # obtain the label number
                labels_df = get_label(sub_fire, lc, lc_name, band_index)
                # add the label text 
                labels_df = labels_df.merge(label_dict, left_on=lc_name, right_index=True, how='left')
                label_all.append(labels_df)
                
            label_all = pd.concat(label_all, ignore_index=False)
            fire = fire.merge(label_all, left_index=True, right_index=True, how='left')

        fire = fire.drop(['long_m', 'lat_m'], axis=1)
        
        # save the file 
        if os.path.exists(save_filename):
            fire.to_csv(save_filename, mode='a', header=None, index=False)
        else:
            fire.to_csv(save_filename, index=False)

def locate_country(p, gdf):
    """Find a country hosting the hotspot.

    Args:
        p: Point object
        gdf: geopandas dataframe with albel 
    
    Returns: str 
        name of the country 
    """
    try: 
        country = gdf[gdf['geometry'].contains(p)]['NAME'].values[0]
    except: 
        country = np.nan
        
    return country
                
def add_countries(df, city_xy_m =[], max_distance=1000, map_file = '../data/world_maps/map3/', country_list = ['Thailand', 'China', 'Vietnam', 'Myanmar (Burma)','Cambodia', 'Laos' ]):
    """Add country label of the hotspot 

    Args:
        df:  dataframe with longitude and latitude
        city_xy_km: [city_x_km, city_y_km] of the city center in mercator coordinate
        max_distance 
        map_file
        country_list

    Returns: (pd.DataFrame, geopanda.DataFrame)
        df: df dataframe with country label
        geodataframe used to labeling the country 

    """
    # prepare geopandas file 
    # loading world map
    gdf =  gpd.read_file(map_file)
    gdf.columns = ['OBJECTID', 'NAME', 'geometry']
    gdf = gdf[gdf['NAME'].isin(country_list)].reset_index(drop=True)

    # remove burma from the name 
    name_list = gdf['NAME'].to_list()
    name_list = [s.replace('Myanmar (Burma)', 'Myanmar') for s in name_list]
    gdf['NAME'] = name_list 
    gdf = gdf.sort_values('NAME')

    df['geometry'] = [Point(x,y) for x, y in zip(df['longitude'], df['latitude'])]
    df['country'] = df['geometry'].swifter.apply(locate_country, gdf=gdf)
    df.drop('geometry', axis=1)

    # sometimes additional information is need such as the area of each country within maximum distance from df 
    if len(city_xy_m) > 0:
         
        # create a 1000 km polygon in mercator in meter
        circle_gons = get_circle(x_cen=city_xy_m[0], y_cen=city_xy_m[1], r=max_distance*1000, num_data=1E4)
        # convert into a tuple 
        circle_gons = {'geometry': Polygon(list(map(tuple, circle_gons.transpose())))}
        circle_gons = gpd.GeoDataFrame(circle_gons, crs="EPSG:3857",index=[0])

        # convert to 6933 to get he correct area 
        circle_6933 = circle_gons['geometry'].to_crs(epsg=6933)
        circle_6933 = circle_6933.values[0] 

        temp = gdf.copy()
        # convert to mercator
        temp['geometry'] = gdf['geometry'].to_crs(epsg=6933)

        inter_area = []
        for i, row in temp.iterrows():
            polygon1 = row['geometry'].intersection(circle_6933) 
            inter_area.append(int(polygon1.area/10**6))

        gdf['inter_area(km2)'] = inter_area
            

    return df, gdf

        





    






