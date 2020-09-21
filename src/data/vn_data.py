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


def extract_vn_data(browser, wait_time=20):
    """Extract hourly data from a Vietnamese EPA station

    Args:
        browser: firefox object
        wait_time: time to wait after the click

    Returns: dataframe

    """

    df = []
    xpath_list = ['//*[@id="custom_datatable_1_paginate"]/ul/li[3]/a',
                  '//*[@id="custom_datatable_1_paginate"]/ul/li[4]/a']
    for xpath in xpath_list:
        try:
            # sometimes the page load slowly resulting in a crash. 
            # If the first attemp fail, try again.
            browser.find_elements_by_xpath(xpath)[0].click()
            time.sleep(wait_time)
        except:
            browser.find_elements_by_xpath(xpath)[0].click()
            time.sleep(wait_time)
        # find number of pages to click
        page = browser.page_source
        soup = BeautifulSoup(page, features="lxml")
        # read the first table
        df.append(pd.read_html(str(soup))[3])

    # reset the page to the original
    back_button = browser.find_elements_by_xpath(
        '//*[@id="custom_datatable_1_previous"]/a')[0]
    try:
        back_button.click()
        time.sleep(5)
    #    back_button.click()
    #    time.sleep(5)
    except BaseException:
        pass
    df = pd.concat(df, ignore_index=True)

    # convert the data columns and datetime format
    df = df = df.drop('#', axis=1)

    replace_dict = {
        'PM-10': 'PM10',
        'PM-2-5': 'PM2.5',
        'Ngày giờ': 'datetime',
        'VN_AQI giờ': 'VN_AQI'}
    col = pd.DataFrame(df.columns)
    col = col.replace(replace_dict).iloc[:, 0].to_list()
    df.columns = col
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.replace('-', np.nan)

    return df.drop_duplicates()


def extract_vn_stations(
        browser,
        url='http://enviinfo.cem.gov.vn/',
        wait_time=10):
    """Get a list of Vietnamese monitoring station

    Args:
        browser:
        url: website with the station informations
        wait_time: time to wait after the click

    Returns: list
        a list of all stations

    """

    page = browser.page_source
    soup = BeautifulSoup(page, features="lxml")

    station_list = []
    for child in soup.find_all(id="cbbStations")[0].children:
        station = child.text
        station_list.append(station)

    return station_list[1:]


def select_vn_station(browser, station, wait_time=15):
    """Select the name of the station and wait.

    """
    # get the search box
    inputelement = browser.find_element_by_xpath(
        '//*[@id="cbbStations_chosen"]/div/div/input')
    inputelement.send_keys(station)
    inputelement.send_keys(Keys.RETURN)
    time.sleep(wait_time)


def extract_station_info(browser):
    """Extract station name, address and location from the website.
    Use after loading the station data up.

    Returns: dictionary of station information

    """
    # extract city information
    city = browser.find_elements_by_xpath(
        '//*[@id="station_province"]')[0].text.split(': ')[-1]
    address = browser.find_elements_by_xpath(
        '//*[@id="station_address"]')[0].text.split(': ')[-1]
    long, lat = browser.find_elements_by_xpath(
        '//*[@id="longlat"]')[0].text.split(' - ')
    long = long.split(': ')[-1]
    lat = lat.split(': ')[-1]

    return {
        'city_vn': city,
        'address': address,
        'longitude': long,
        'latitude': lat}


def download_vn_data(
        url='http://enviinfo.cem.gov.vn/',
        save_folder='../data/vn_epa/'):
    """Download hourly data for all monitoring stations from Vietnamese EPA.

    All data are saved into the save files

    Args:
        url: link to the data website
        save_folder: folder to save the data to

    """
    # open the webpage and wait
    # the page sometimes is slow, so need to wait a bit.

    executable_path = f'C:/Users/Benny/Documents/Fern/aqi_thailand2/notebooks/geckodriver.exe'
    browser = webdriver.Firefox(executable_path=executable_path)
    browser.get(url)
    time.sleep(60)

    # prepare filename and turn to absolute path
    date = datetime.now().strftime('%Y-%m-%d_%H-%M')
    filename = os.path.abspath(f'{save_folder}{date}.csv')
    meta_filename = os.path.abspath(f'{save_folder}stations_info.json')

    # extract stations name
    try:
        station_list = extract_vn_stations(browser)
    except BaseException:
        time.sleep(60)
        station_list = extract_vn_stations(browser)

    print(
        f'number of station {len(station_list)}. Takes about 1 mins per station')

    city_dict = {'Hà Nội': 'Hanoi',
                 'Phú Thọ': 'Phu Tho',
                 'Quảng Ninh': 'Quang Ninh',
                 'Thừa Thiên Huế': 'Thua Thien Hue',
                 'Bắc Ninh': 'Bac Ninh',
                 'Thái nguyên': 'Thai Nguyen',
                 'Gia Lai': 'Gia Lai',
                 'Cao Bằng': 'Cao Bang'}

    data = []
    station_info_list = []

    for station in tqdm(station_list):
        # select station
        select_vn_station(browser, station, wait_time=20)
        city_name = station.split(':')[0]
        # change city_name to english
        if city_name in city_dict.keys():
            city_name = city_dict[city_name]

        station_info = extract_station_info(browser)
        station_info['city'] = city_name
        station_info_list.append(station_info)

        # extract data
        df = extract_vn_data(browser, wait_time=20)
        df['city'] = city_name
        df['station'] = station
        #print(station, 'has data len ', len(df))
        data.append(df)

    browser.close()

    station_len = len(df)
    #print('the last station has len', station_len )
    data = pd.concat(data, ignore_index=True)
    print('the total data has len', len(data))
    print('save file as ', filename)
    data.to_csv(filename, index=False)

    with open(meta_filename, 'w') as f:
        json.dump(station_info_list, f)

    return data, station_info_list


if __name__ == '__main__':

    download_vn_data(save_folder='../../data/vn_epa/')
