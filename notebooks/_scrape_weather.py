# -*- coding: utf-8 -*-
import os
import sys

if __package__: 

    
    from ..src.imports import *
    from ..src.gen_functions import *
    from ..src.data.weather_data import *
     
else:
    # run as a script, use absolute import
    _i = os.path.abspath('..')
   
    if _i not in sys.path:
        sys.path.insert(0, _i)
    # -*- coding: utf-8 -*-
    from src.imports import *
    from src.gen_functions import *
    from src.data.weather_data import *

def main():

    prov_weather_station_dict = { 'Rayong':'Ban Chang', 
    'Chiang Mai': 'Mueang Chiang Mai',
    'Songkhla':'Khlong Hoi Khong',
    'Saraburi':'Bangkok',
    'Bangkok':'Bangkok',
    'Samut Sakhon':'Bangkok',
    'Nan':'Mueang Nan',
    'Sa Kaeo':'Chaloem Phra Kiat',
    'Loei':'Mueang Loei',
    'Chon Buri':'Bangkok',
    'Khon Kaen':'Mueang Khon Kaen',
    'Ratchaburi':'Photharam','Phuket': 'Thalang',
    'Pathum Thani': 'Bangkok',
    'Samut Prakan': 'Bangkok',
    'Nonthaburi':' Bankgok',
    'Phra Nakhon Si Ayutthaya':'Bangkok',
    'Lampang': 'Mueang Lampang',
    'Narathiwat': 'Mueang Narathiwat',
    'Yala': 'Mueang Narathiwat',
    'Mae Hong Son': 'Mueang Mae Hong Son',
    'Chiang Rai': 'Mueang Chiang Rai',
    'Lamphun': 'Mueang Lamphun',
    'Phrae': 'Mueang Phrae',
    'Nakhon Ratchasima': 'Chaloem Phra Kiat',
    'Phayao': 'Mueang Chiang Rai',
    'Nakhon Sawan': 'Mueang Nakhon Sawan',
    'Surat Thani': 'Phunphin',
    'Chachoengsao':'Bangkok',
    'Prachin Buri':'Bangkok',
    'Tak' : 'Mueang Tak',
    'Kanchanaburi': 'Mueang Kanchanaburi',
    'Satun':'Langkawi',
    'Nakhon Pathom':'Bangkok',
    'Sukhothai':'Sawankhalok',
    'Si Sa Ket': 'Mueang Ubon Ratchathani',
    'Amnat Charoen':'Mueang Ubon Ratchathani',
    'Mukdahan':'Sikhottabong',
    'Bueng Kan':'Mueang Nakhon Phanom',
    'Nong Bua Lam Phu': 'Mueang Udon Thani',
    'Yasothon':'Thawat Buri',
    'Nakhon Nayok':'Bangkok',
    'Sakon Nakhon':'Mueang Sakon Nakhon',
    'Ubon Ratchathani': 'Mueang Ubon Ratchathani',
    'Nong Khai':'Sikhottabong',
    'Samut Songkhram':'Bangkok',
    'Chai Nat':' Bangkok',
    'Phetchaburi':'Bangkok',
    'Ang Thong':' Bangkok',
    'Nakhon Si Thammarat': 'Mueang Nakhon Si Thammarat',
    'Nakhon Phanom':'Mueang Nakhon Phanom',
    'Phitsanulok': 'Mueang Phitsanulok',
    'Trat': 'Khao Saming',
    'Suphan Buri': 'Bangkok'}

    # year using PM2.5
    prov_year_dict = {'Rayong': 2011, 'Chiang Mai': 2004, 'Songkhla': 2012, 'Saraburi': 2012, 'Bangkok': 2005, 'Samut Sakhon': 2013, 'Nan': 2015, 'Sa Kaeo': 2014, 'Loei': 2014, 'Chon Buri': 2015, 'Khon Kaen': 2014, 'Ratchaburi': 2014}    
    prov_year_dict = {'Phuket': 2016, 'Pathum Thani': 2016, 'Samut Prakan': 2016, 'Nonthaburi': 2016, 'Phra Nakhon Si Ayutthaya': 2016, 'Lampang': 2016, 'Narathiwat': 2017, 'Yala': 2016, 'Mae Hong Son': 2018, 'Chiang Rai': 2016, 'Lamphun': 2018, 'Phrae': 2018, 'Nakhon Ratchasima': 2016, 'Phayao': 2018, 'Nakhon Sawan': 2016, 'Surat Thani': 2016, 'Chachoengsao': 2019, 'Prachin Buri': 2016, 'Tak': 2016, 'Kanchanaburi': 2016, 'Satun': 2017, 'Nakhon Pathom': 2016, 'Sukhothai': 2020, 'Si Sa Ket': 2019, 'Amnat Charoen': 2019, 'Mukdahan': 2020, 'Bueng Kan': 2021, 'Nong Bua Lam Phu': 2019, 'Yasothon': 2019, 'Nakhon Nayok': 2019, 'Sakon Nakhon': 2016, 'Ubon Ratchathani': 2018, 'Nong Khai': 2016, 'Samut Songkhram': 2020, 'Chai Nat': 2020, 'Phetchaburi': 2020, 'Ang Thong': 2020, 'Nakhon Si Thammarat': 2016, 'Nakhon Phanom': 2021, 'Phitsanulok': 2016, 'Trat': 2021, 'Suphan Buri': 2016}
    # year using PM10
    prov_year_dict = {'Phuket': 1996, 'Pathum Thani': 1996, 'Samut Prakan': 1996, 'Nonthaburi': 1996, 'Phra Nakhon Si Ayutthaya': 1996, 'Lampang': 1998, 'Narathiwat': 2006, 'Yala': 2006, 'Mae Hong Son': 2008, 'Chiang Rai': 2008, 'Lamphun': 2009, 'Phrae': 2010, 'Nakhon Ratchasima': 2010, 'Phayao': 2010, 'Nakhon Sawan': 2013, 'Surat Thani': 2013, 'Chachoengsao': 2014, 'Prachin Buri': 2016, 'Tak': 2016, 'Kanchanaburi': 2017, 'Satun': 2017, 'Nakhon Pathom': 2019, 'Sukhothai': 2019, 'Si Sa Ket': 2019, 'Amnat Charoen': 2019, 'Mukdahan': 2019, 'Bueng Kan': 2019, 'Nong Bua Lam Phu': 2019, 'Yasothon': 2019, 'Nakhon Nayok': 2019, 'Sakon Nakhon': 2020, 'Ubon Ratchathani': 2020, 'Nong Khai': 2020, 'Samut Songkhram': 2020, 'Chai Nat': 2020, 'Phetchaburi': 2020, 'Ang Thong': 2020, 'Nakhon Si Thammarat': 2021, 'Nakhon Phanom': 2021, 'Phitsanulok': 2021, 'Trat': 2021, 'Suphan Buri': 2021,
                        'Rayong': 1995, 'Chiang Mai': 1995, 'Songkhla': 1996, 'Saraburi': 1996, 'Bangkok': 1996, 'Samut Sakhon': 1996, 'Nan': 2009, 'Sa Kaeo': 2011, 'Loei': 2011, 'Chon Buri': 2013, 'Khon Kaen': 2013, 'Ratchaburi': 2014}
    
    w_folder = '../data/weather_cities/'

    # prov_list = ['Rayong', 'Khon Kaen',
    # 'Songkhla',
    # 'Samut Sakhon',
    # 'Nan',
    # 'Sa Kaeo',
    # 'Loei',
    # 'Ratchaburi','Chiang Mai', 'Bangkok',
    # 'Chon Buri', 'Saraburi']

    prov_list = ['Tak',
    'Nakhon Sawan',
    'Samut Songkhram',
    'Samut Prakan',
    'Chiang Mai',
    'Nan',
    'Nakhon Pathom',
    'Lamphun',
    'Samut Sakhon',
    'Sa Kaeo',
    'Nakhon Ratchasima',
    'Nakhon Phanom',
    'Chiang Rai',
    'Ratchaburi',
    'Khon Kaen',
    'Lampang',
    'Loei',
    'Chon Buri',
    'Rayong',
    'Pathum Thani',
    'Phra Nakhon Si Ayutthaya',
    'Ubon Ratchathani',
    'Nonthaburi',
    'Bueng Kan'
    ]


    # create a unique city_names and min_year 
    city_names = [prov_weather_station_dict[prov] for prov in prov_list]
    year_list = [prov_year_dict[prov] for prov in prov_list]
    df = pd.DataFrame({'city_name': city_names, 'year': year_list})
    df = df.sort_values('year', ascending=False).drop_duplicates('city_name')
    city_names = df['city_name'].to_list()
    min_year_list = df['year'].to_list()

    weather_station_info = find_weather_stations(city_names, weather_json_file=w_folder+'weather_station_info.json')
    print('number of station to scrape', len(weather_station_info))

    #year_list =  np.arange(1996, 2022)[::-1]
    year_list =  np.arange(2001, 2022)[::-1]
     
    for i, city_json in enumerate(weather_station_info):
        print(city_json['city_name'])
        for year in year_list:
            if year >= min_year_list[i]:
                print(year)
                start_date = datetime(year,1,1)
                end_date = datetime(year,3,1)
                update_weather(city_json, data_folder=w_folder, start_date=start_date, end_date=end_date)

                start_date = datetime(year,3,1)
                end_date = datetime(year,6,1)
                update_weather(city_json, data_folder=w_folder, start_date=start_date, end_date=end_date)

                start_date = datetime(year,6,1)
                end_date = datetime(year,9,1)
                update_weather(city_json, data_folder=w_folder, start_date=start_date, end_date=end_date)

                start_date = datetime(year,9,1)
                end_date = datetime(year,10,1)
                update_weather(city_json, data_folder=w_folder, start_date=start_date, end_date=end_date)

                if year == datetime.now().year:
                    tart_date = datetime(year,10,1)
                    end_date = datetime.now()
                    update_weather(city_json, data_folder=w_folder, start_date=start_date, end_date=end_date)
                else:
    
                    start_date = datetime(year,10,1)
                    end_date = datetime(year+1,1,1)
                    update_weather(city_json, data_folder=w_folder, start_date=start_date, end_date=end_date)
                

if __name__ == '__main__':

    main()