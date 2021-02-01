# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path
from tqdm import tqdm, tqdm_notebook
import json
import numpy as np
import pandas as pd
from glob import glob

# webscraping
import requests
import wget
from bs4 import BeautifulSoup
from selenium import webdriver

import time
from datetime import datetime, date, timedelta


from selenium.webdriver.support.select import Select
from selenium.webdriver.common.keys import Keys

def extract_bkp_stations(soup):
    """ Extract station selector and station name

    """
    station_list_html = soup.find(attrs = {"name":"MeasIndex"})
    station_children = station_list_html.findChildren(
        'option', recursive=False)
    station_selector_list = []
    station_name_list = []
    for station_child in station_children:
        station_selector_list.append(station_child.attrs['value'])
        station_name_list.append(station_child.string)
    
    return station_selector_list, station_name_list

def select_data(url, browser, sta_id, start_date, start_hour, wait_time=10):
    """Select station_name (sta_id), start_date  on the webpage and display it. And wait for wait_time.

    """
    # select station id
    browser.get(url)
    time.sleep(wait_time)
    station = Select(browser.find_element_by_css_selector(
            'select[name="MeasIndex"]'))
    # select stations
    station.select_by_value(sta_id)

    # select start_date   
    datewidget = browser.find_element_by_xpath('//*[@id="date_from"]') 
    browser.execute_script(f"arguments[0].value = '{start_date}';", datewidget)

    # select time

    starttimewidget = Select(browser.find_element_by_xpath('//*[@id="time_from"]/select'))
    endtimewidget = Select(browser.find_element_by_xpath('//*[@id="time_to"]/select'))
    starttimewidget.select_by_value(start_hour)
    endtimewidget.select_by_value('23:00')

    # display data 
    button = browser.find_element_by_xpath('//*[@id="form1"]/table/tbody/tr/td[3]/button[1]')
    button.click()
    time.sleep(wait_time)

def extract_bpk_data(browser):
    """ Click all the link on the selenium browser object.
    Parse the html table to panda dataframe. concatentate all the dataframe in each page.
    Return dataframe of all the data for that station

    """

    # find number of pages to click
    page = browser.page_source
    soup = BeautifulSoup(page, features="lxml")
     
    try:
        num_click_nodes = soup.find_all(attrs={'class': "paginate_button"})[-2]
        num_click = int(num_click_nodes.string)
        num_click = np.arange(num_click)
    except:
        num_click = [0]


    data_all = []
    for i in num_click:
        # extract table from page
        page = browser.page_source
        soup = BeautifulSoup(page, features="lxml")
        df = pd.read_html(str(soup))[-1]
         
        # append to the data
        data_all.append(df)
        # click the next page except for the last page
        if i < np.max(num_click):
            next_button = browser.find_element_by_id('example_next')
            next_button.click()

    return pd.concat(data_all)
         

def get_bkp_station_data_save(url, browser, sta_id, sta_name, data_folder, wait_time=5):

    """ Display the data in para_selector_list for the corresponding station id (sta_id).
    #. Parse the data into the dataframe and save in the data_folder
    #. add datetime columns
    #. load existing file and get the last time stamp
    #. keep the data from the last timestamp
    #. save by appending to the old file

    """  
    filename = data_folder + 'bkp' + sta_id + 't.csv'
    if os.path.exists(filename):
        old_data = pd.read_csv(filename)
        old_data['datetime'] = pd.to_datetime(old_data['datetime'])
        # add 1 hour to the last hour 
        last_datetime = old_data['datetime'].max() + pd.Timedelta('1hour')
        old_columns = old_data.columns
    else:
        last_datetime = pd.to_datetime('2019-01-01 00:00:00')

    start_date = last_datetime.strftime('%d-%m-%Y')
    start_hour = str(last_datetime.hour) + ':00'
     
    #print(start_date, start_hour)
     
    # display the data on the webpage
    select_data(url, browser, sta_id, start_date, start_hour, wait_time=wait_time)

    try:
        # parse data into dataframe
        data = extract_bpk_data(browser)
        # drop average row 
        data = data[data['Datetime'] !='Average']
    except:
        data = pd.DataFrame()

     

    if len(data) > 1:
        # change columns name 
        data.columns = data.columns.str.replace('Datetime', 'datetime')
        data.columns = [s.split('(')[0] for s in data.columns]
        # drop average row 
        data = data[data['datetime'] !='Average']
        data['datetime'] = pd.to_datetime(data['datetime'])
        data = data.sort_values('datetime')
        data = data.drop_duplicates('datetime')

        # remove negative number 
        data = data.set_index('datetime')
        # remove negative number
        # remove 0
        for col in data.columns:
            idxs = data[data[col] < 0].index
            data.loc[idxs, col] = np.nan

        if 'PM2.5' in data.columns:
            idxs = data[data['PM2.5'] > 300 ].index
            data.loc[idxs, 'PM2.5'] = np.nan

        if 'PM10' in data.columns:
            idxs = data[data['PM10'] > 600 ].index
            data.loc[idxs, 'PM10'] = np.nan
        
        
        if 'NO2' in data.columns:
            idxs = data[data['NO2'] > 300 ].index
            data.loc[idxs, 'NO2'] = np.nan

        data = data.reset_index()

        # add station id and station name
        data['station_id'] =  'bkp' + sta_id + 't'
        data['station_name'] = sta_name

        # save the data
        if os.path.exists(filename):
            # file already exists, concat file before save
            data = pd.concat([data, old_data], ignore_index=True)
            data = data.drop_duplicates('datetime')
            data = data.sort_values('datetime')
           # data.to_csv(filename, mode='a', header=False, index=False, encoding='utf-8')
        else:
            # file does not exist, create the file
            print('create new', filename)
        data.to_csv(filename, index=False, encoding='utf-8')

    try:
        temp = pd.read_csv(filename)
        
    except:
        print(f'filename {filename} contains error')

    return data


def update_bkp(url: str = 'https://bangkokairquality.com/bma/report?lang=th', data_folder: str = '../data/bkp_hourly/'):
    """Scrape new BKP data.

    Append new data to an exsiting file. Create a new file if the file does not exist.

    Args:
        url (str, optional): [description]. Defaults to 'https://bangkokairquality.com/bma/report?lang=th'.
        data_folder (str, optional): [description]. Defaults to '../data/bkp_hourly/'.


    """

    print('download more pollution data from Bangkok Air Quality  Website')

    # use Firefox to open the website
    dir_name = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')
    executable_path = f'{dir_name}/geckodriver.exe'

    browser = webdriver.Firefox(executable_path=executable_path)

    browser.get(url)
    time.sleep(1)

    # find all station names and parameters from the webpage
    page = browser.page_source
    soup = BeautifulSoup(page, features="lxml")

    # extract station name and selector
    sta_selector_list, station_name_list = extract_bkp_stations(soup)

    # process station name

    station_name_list = [' '.join(s.split(' ')[1:]) for s in station_name_list]
    station_name_list = [ s.lstrip().rstrip() for s in station_name_list]


    # update station info file 
    station_info = {}
    for station_id, district in zip(sta_selector_list, station_name_list):
        k = 'bkp' + station_id + 't'
        station_info[k] = district
    
    station_name  = data_folder + 'station_info.json'
    with open(station_name, 'w', encoding='utf8') as f:
        json.dump(station_info, f)

    if not os.path.exists(data_folder):
        print(f'create data folder {data_folder}')
        os.mkdir(data_folder)

    for sta_id, sta_name in tqdm(zip(sta_selector_list, station_name_list)):
        print('station id', sta_id)
         
        data = get_bkp_station_data_save(url, browser, sta_id, sta_name, data_folder)

    browser.close()


    #return sta_selector_list, station_name_list

def build_station_info(pcd_json:str='../data/air4thai_hourly/station_info.json', bkp_folder:str='../data/bkp_hourly/'):
    """Merge station information from both json files and save 

    Args:
        pcd_json (str, optional):  . Defaults to '../data/bkp_hourly/add_stations_location_process.csv'.
        bkp_folder (str, optional):  . Defaults to '../data/bkp_hourly/station_info.json'.

    """
    # extract bkp station location
    # load BKP station json 
    with open(bkp_folder + 'station_info.json', encoding="utf-8") as f:
        bkp_station = json.load(f)

    # extract a list of data 
    bkp_station_list = pd.Series(bkp_station).index.to_list()

    # load PCD station
    with open(pcd_json, encoding="utf-8") as f:
        station_info = json.load(f)
        
    station_info = pd.DataFrame(station_info['stations'])

    # bkp station in PCD info 
    bkp_station = station_info[station_info['stationID'].isin(bkp_station_list)]
    # add additional location
    temp = pd.read_csv(bkp_folder + 'add_stations_location_process.csv')
    temp = temp[temp['stationID'].isin(bkp_station_list)]
    temp = add_merc_col(temp, lat_col='lat', long_col='long')
    # combine stations information 
    bkp_station = pd.concat([bkp_station, temp], ignore_index=True)

    # save 
    bkp_station.to_csv( bkp_folder + 'station_info.csv', index=False)

    print(bkp_station.shape)

def main(
        main_folder: str = '../data/'):
    """Download all data 

    Args:
        main_folder: main data_folder

    """

    # fix relative folder 
    main_folder = os.path.abspath(main_folder).replace('\\', '/') + '/'
    print(f'main data folder ={main_folder}')

    print('update BKP data')
    update_bkp(data_folder=f'{main_folder}bkp_hourly/')


if __name__ == '__main__':
    main(main_folder='../../data/' )