# -*- coding: utf-8 -*-
"""Set configuration dictionarys for the dataset objects

"""

def set_config(dataset):
    """Add a bunch of configuration dictonary as attribute to the dataset object
    
    Args:
        dataset: a dataset object

    """ 
    # a defaul list of pollutants
    dataset.gas_list = ['PM2.5', 'PM10', 'O3', 'CO', 'NO2', 'SO2']

     
    # pollution data configuration setting 
    config_dict = {'Chiang Mai': {'th_stations': ['35t', '36t']}, 
                    #'Bangkok': {'th_stations': ['02t', '03t', '05t', '11t', '12t', '50t', '52t', '53t', '54t', '59t', '61t']}, 
                    #'Bangkok': {'th_stations': ['53t', '54t', '59t', '61t']}, 
                    'Bangkok': {'th_stations': [ '10t', '11t', '50t', '53t', '54t', '59t',  '61t']}, 
                    'Nakhon Si Thammarat': {'th_stations': ['42t', 'm3'], 
                                            'cmu_stations': [118]}, 
                    'Nakhon Ratchasima': {'th_stations':['47t']},
                    'Hanoi': {'b_stations': ['Ha_Dong'], 'us_emb': True}, 
                    'Jakarta': {'us_emb': True}}

    # fire zone for each city
    zone_dict = {'Chiang Mai' : [0, 100, 200, 400, 700, 1000],
             'Bangkok':  [0, 100, 200, 400, 600, 800, 1200],
             'Nakhon Si Thammarat': [0, 200, 450,  1000],
             'Nakhon Ratchasima': [0, 130, 300, 600, 1200],
            'Da Nang': [0, 75, 300, 700,  1000],
             'Hanoi': [0, 120, 400, 700, 1200], 
            'default': [0, 100, 200, 400, 800, 1000]
                }

    
    # mapping city name to weather city name
    dataset.city_wea_dict = {'Chiang Mai': 'Mueang Chiang Mai',
                 'Bangkok': 'Bangkok',
                 'Hanoi': 'Soc Son',
                 'Jakarta': 'East Jakarta',
                 'Da Nang': 'Hai Chau',
                 'Nakhon Si Thammarat':'Mueang Nakhon Si Thammarat',
                 'Nakhon Ratchasima':'Chaloem Phra Kiat'}
      

    # cropping point to remove the earlier data 
    dataset.crop_dict = {'Chiang Mai':{'PM2.5': '2010'},
                    'Hanoi':{'PM2.5': '2016-03-21'}}

    # US AQI standard
    dataset.transition_dict =  {
                    'PM2.5': [ 0, 12.0, 35.4, 55.4, 150.4, 250.4, 350.4, 500, 1e3], 
                    'PM10': [0, 155, 254, 354, 424, 504, 604, 1e3], 
                    'O3': [0, 54, 70, 85, 105, 200, 1e3], 
                    'SO2': [0, 75, 185, 304, 504, 604, 1e3], 
                    'NO2': [0, 53, 100, 360, 649, 1249, 2049, 1e3], 
                    'CO': [0, 4.4, 9.4, 12.4, 15.4, 30.4, 40.4, 50.4, 1e3]}

    # rolling average for different pollutant
    dataset.roll_dict = {'PM2.5': 24,
             'PM10': 24,
             'O3': 8,
             'SO2': 1,
             'NO2': 1,
             'CO': 8}

    try:
        # load config_dict for the city 
        dataset.config_dict = config_dict[dataset.city_name]
    except:
        dataset.config_dict = {}
    # check if the preset fire zone exist for the city. Use default value otherwise.
    try:
        dataset.zone_list = zone_dict[dataset.city_name]
    except:
        dataset.zone_list = zone_dict['default']


