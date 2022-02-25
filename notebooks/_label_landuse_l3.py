
# -*- coding: utf-8 -*-
import os
import sys
import geopandas as gpd
from geopandas.tools import sjoin
from shapely.geometry import Point
from joblib import Parallel
import pyproj

if __package__: 

    
    from ..src.imports import *
    from ..src.gen_functions import *
    from ..src.features.map_dataset import MapDataset

else:
    # run as a script, use absolute import
    _i = os.path.abspath('..')
   
    if _i not in sys.path:
        sys.path.insert(0, _i)
    # -*- coding: utf-8 -*-
    from src.imports import *
    from src.gen_functions import *
    from src.features.map_dataset import MapDataset

def get_fire_gdf(fire, buffer=250):
    # add geometrical point and turn into area
    fire_gdf = gpd.GeoDataFrame(fire, geometry=gpd.points_from_xy(fire.longitude, fire.latitude))
    fire_gdf = fire_gdf.set_crs("EPSG:4326")
    fire_gdf = fire_gdf.to_crs("EPSG:3857")
    fire_gdf['geometry'] = fire_gdf['geometry'].buffer(buffer)
    return fire_gdf

def extract_lucode(idxs, sub_fire, sub_landuse ):
    return sjoin(sub_fire.loc[idxs][['geometry']], sub_landuse, how='left')[['lucode']]

def label_province(prov_name, fire_gdf, landuse, chunk=50):
    # select data by province 
    sub_fire = fire_gdf[fire_gdf['province'] == prov_name]
    if len(sub_fire) > 0:
        sub_landuse = landuse[landuse['province'] == prov_name][['geometry', 'lucode']]

        # calculate number of splits
        n_splits = ceil(len(sub_fire)/chunk)
        idx_splits = np.array_split(sub_fire.index, n_splits)

        labeled_all = Parallel(n_jobs=-2)(delayed(extract_lucode)(idxs, sub_fire, sub_landuse) for idxs in idx_splits)
        return pd.concat(labeled_all, ignore_index=False)
    else:
        return pd.DataFrame()

def label_province_v2(prov_name, fire_gdf, land_filename_header, chunk=50):
    # select data by province 
    sub_fire = fire_gdf[fire_gdf['province'] == prov_name]
    prov_filename = land_filename_header + '_' + prov_name + '.shp'
    if len(sub_fire) > 0:
        #print('province filename', prov_filename)
        sub_landuse = gpd.read_file(prov_filename, encoding='iso_8859_11')[['geometry', 'lucode']]

        # calculate number of splits
        n_splits = ceil(len(sub_fire)/chunk)
        idx_splits = np.array_split(sub_fire.index, n_splits)

        labeled_all = Parallel(n_jobs=-2)(delayed(extract_lucode)(idxs, sub_fire, sub_landuse) for idxs in idx_splits)
        del sub_landuse
        return pd.concat(labeled_all, ignore_index=False)
    else:
        return pd.DataFrame()

def add_fire_year(year, mfire_folder, label_folder, save_folder, instr='MODIS'):

    print('working with year ', year)
    if instr== 'MODIS':
        filename = mfire_folder + f'th_fire_m_{year}.csv'
        save_filename = save_folder + f'th_fire_m_{year}.csv'
        buffer = 500
    else:
        
        filename = mfire_folder + f'th_fire_v_{year}.csv'
        save_filename = save_folder + f'th_fire_v_{year}.csv'
        buffer = 200
    print('load fire ' + filename)
    fire = pd.read_csv(filename)
    fire = add_merc_col(fire, lat_col='latitude', long_col='longitude', unit='m')
    fire_gdf = get_fire_gdf(fire, buffer=buffer)
    print('process fire will be saved to ' + save_filename)

    if int(year) == 2014:
        land_filename = label_folder + str(year-1) + '/' + str(year-1) + '.shp'
    else:
        land_filename = label_folder + str(year) + '/' + str(year) + '.shp'

    print('load landuse ' + land_filename)
    landuse = gpd.read_file(land_filename, encoding='iso_8859_11')

    use_backup_songkla = 'Songkhla' not in landuse['province'].unique()
    #use_backup_songkla = True

    if (use_backup_songkla):
        print('load Songkhla landuse')
        songkhla_filename = label_folder  + '2012_songkhla/2012_songkhla.shp'
        print(songkhla_filename)
        songkhla_landuse = gpd.read_file(songkhla_filename)
    else:
        print('Songkhla already in the landuse')
        songkhla_landuse = pd.DataFrame()

    print('num province in fire data ', fire['province'].nunique())
    print('num province in landuse data ', landuse['province'].nunique())

    year_label = []
    for prov_name in tqdm(fire['province'].dropna().unique()):
        if (prov_name == 'Songkhla') & (use_backup_songkla):
            prov_label = label_province(prov_name, fire_gdf, songkhla_landuse, chunk=50)
        else:
            prov_label = label_province(prov_name, fire_gdf, landuse, chunk=50)

        year_label.append(prov_label)

    year_label = pd.concat(year_label)

    print('finish labeling, merging to fire')
    new_fire = fire.merge(year_label, left_index=True, right_index=True, how='outer')
    new_fire = new_fire.drop('geometry', axis=1)
    new_fire.to_csv(save_filename, index=False, encoding='iso_8859_11')

    print( 'num fire with no label ', new_fire[new_fire['lucode'].isna()].shape  )
    print( 'total shape ', new_fire.shape  )

    del landuse
    del songkhla_landuse

def add_fire_year_v2(year, mfire_folder, label_folder, save_folder, instr='MODIS'):
    # add fire year load the landuse in the province level to save memory 

    print('working with year ', year)
    if instr== 'MODIS':
        filename = mfire_folder + f'th_fire_m_{year}.csv'
        save_filename = save_folder + f'th_fire_m_{year}.csv'
    else:
        
        filename = mfire_folder + f'th_fire_v_{year}.csv'
        save_filename = save_folder + f'th_fire_v_{year}.csv'
    
    print('load fire ' + filename)
    fire = pd.read_csv(filename)
    fire = add_merc_col(fire, lat_col='latitude', long_col='longitude', unit='m')
    fire_gdf = get_fire_gdf(fire)
    print('process fire will be saved to ' + save_filename)

    if int(year) == 2014:
        land_filename_header = label_folder + str(year-1) + '_prov/' + str(year-1)  
    else:
        land_filename_header = label_folder + str(year) + '_prov/' + str(year)  
     

    print('num province in fire data ', fire['province'].nunique())


    year_label = []
    for prov_name in tqdm(fire['province'].dropna().unique()):
        prov_label = label_province_v2(prov_name, fire_gdf, land_filename_header, chunk=50)
        year_label.append(prov_label)

    year_label = pd.concat(year_label)

    print('finish labeling, merging to fire')
    new_fire = fire.merge(year_label, left_index=True, right_index=True, how='outer')
    new_fire = new_fire.drop('geometry', axis=1)
    new_fire.to_csv(save_filename, index=False, encoding='iso_8859_11')

    print( 'num fire with no label ', new_fire[new_fire['lucode'].isna()].shape  )
    print( 'total shape ', new_fire.shape  )

    

def add_fire_year_v3(year, mfire_folder, label_folder, save_folder, instr='MODIS', chunksize=1000):
    # add fire year load the landuse in the province level to save memory 
    # load data in block to save memory

    print('working with year ', year)
    if instr== 'MODIS':
        filename = mfire_folder + f'th_fire_m_{year}.csv'
        save_filename = save_folder + f'th_fire_m_{year}.csv'
    else:
        
        filename = mfire_folder + f'th_fire_v_{year}.csv'
        save_filename = save_folder + f'th_fire_v_{year}.csv'
    
    print('process fire will be saved to ' + save_filename)

    if int(year) == 2014:
        land_filename_header = label_folder + str(year-1) + '_prov/' + str(year-1)  
    else:
        land_filename_header = label_folder + str(year) + '_prov/' + str(year)  
     

    print('load fire ' + filename)
    fire = pd.read_csv(filename)
    fire = fire.sort_values('province')
    fire.to_csv(filename, index=False)

    print('reload fire '+ filename)
    new_fire_shape = 0
    for i, fire in tqdm(enumerate(pd.read_csv(filename, chunksize=chunksize))):

        fire = add_merc_col(fire, lat_col='latitude', long_col='longitude', unit='m')
        fire_gdf = get_fire_gdf(fire)
        year_label = []
        for prov_name in fire['province'].dropna().unique():
            prov_label = label_province_v2(prov_name, fire_gdf, land_filename_header, chunk=50)
            year_label.append(prov_label)

        year_label = pd.concat(year_label)

        print('finish labeling, merging to fire')
        new_fire = fire.merge(year_label, left_index=True, right_index=True, how='outer')
        new_fire = new_fire.drop('geometry', axis=1)
        new_fire_shape += len(new_fire)
        if i ==0:
            # first file create new file
            new_fire.to_csv(save_filename, index=False, encoding='iso_8859_11')
        else:
            new_fire.to_csv(save_filename, index=False, encoding='iso_8859_11', header=False, mode='a')

     
    print( 'total shape ', new_fire_shape  )

 

def add_fire_year_v4(year, fire_folder, label_folder, save_folder, instr='VIIRS'):
    # add fire year load the landuse in the province level to save memory 

    print('working with year ', year)
    filenames = glob(fire_folder + '*.csv')
    # select year for the fires
    filenames = [s for s in filenames if str(year) in s]
    print('number of files ', len(filenames))

    for filename in tqdm(filenames):
    
        print('load fire ' + filename)
        save_filename = filename.replace('v_prov', 'v_prov_proc')

        if not os.path.exists(save_filename):

            fire = pd.read_csv(filename)
            fire = add_merc_col(fire, lat_col='latitude', long_col='longitude', unit='m')
            fire_gdf = get_fire_gdf(fire)
            print('process fire will be saved to ' + save_filename)
    
            if int(year) == 2014:
                land_filename_header = label_folder + str(year-1) + '_prov/' + str(year-1)  
            else:
                land_filename_header = label_folder + str(year) + '_prov/' + str(year)  
         
            print('num province in fire data ', fire['province'].nunique())
    
    
            year_label = []
            for prov_name in  fire['province'].dropna().unique():
                prov_label = label_province_v2(prov_name, fire_gdf, land_filename_header, chunk=50)
                year_label.append(prov_label)
    
            year_label = pd.concat(year_label)
    
            print('finish labeling, merging to fire')
            new_fire = fire.merge(year_label, left_index=True, right_index=True, how='outer')
            new_fire = new_fire.drop('geometry', axis=1)
            new_fire.to_csv(save_filename, index=False, encoding='iso_8859_11')
    
            #print( 'num fire with no label ', new_fire[new_fire['lucode'].isna()].shape  )
            print( 'total shape ', new_fire.shape  )
        else:
            print(save_filename, ' already exist')


def main():

    label_folder = os.path.abspath('../data/landuse_l3/') + '/'
    poll_folder = os.path.abspath('../data/poll_map/') + '/'
    # modis
    #mfire_folder = poll_folder + 'th_fire_years_m/'
    #save_folder = poll_folder + 'th_fire_years_m_proc/'

    # VIIRS
    #mfire_folder = poll_folder + 'th_fire_years_v/'
    #save_folder = poll_folder + 'th_fire_years_v_proc/'
    
    # VIIRS
    mfire_folder = poll_folder + 'th_fire_years_v_prov/'
    save_folder = poll_folder + 'th_fire_years_v_prov_proc/'

    # year with landlabel 
    ##year_range = np.arange(2007, 2021)
    year_range = np.arange(2015, 2021)
    #year_range  = [2012]

    print('year range ', year_range)

    for year in year_range:
        add_fire_year_v4(year, fire_folder=mfire_folder, label_folder=label_folder, save_folder=save_folder, instr='VIIRS')

    

if __name__ == '__main__':

    main()