# -*- coding: utf-8 -*-
from ..imports import *
from .weather_data import *
from selenium.webdriver.support.select import Select


"""Functions for downloading Berkeley Data, air4thai data.

"""


def download_b_data(
        data_folder: str = '../data/pm25/',
        url: str = 'http://berkeleyearth.lbl.gov/air-quality/maps/cities/Thailand/'):
    """Download all files from the Berkeley Earth directory in url to data_folder.

    """
    print('download Berkeley PM2.5 data for all files in', url)
    res = requests.get(url)
    # create a soup object of Berkeley earth website
    soup = BeautifulSoup(res.text, features="lxml")
    # find all provinces in this database
    provinces = soup.find_all(href=re.compile('/'))[1:]
    assert os.path.exists(data_folder), f'no data folder {data_folder}'
    for province in provinces:
        grab_url = url + province['href']
        download_province_data(grab_url, data_folder=data_folder)


def download_province_data(grab_url: str, data_folder: str):
    """Download a province data.

    Remove existing file before download.

    """
    prov_r = requests.get(grab_url)
    prov_s = BeautifulSoup(prov_r.text, features="lxml")
    for tag in prov_s.find_all(href=re.compile('.txt')):
        # build province url
        data_url = grab_url + tag['href']
        name = data_folder + tag['href']
        # remove existing file if exist
        try:
            os.remove(name)
        except BaseException:
            pass
        # download the data
        wget.download(data_url, name)


def get_city_info(data_folder='../data/pm25/'):
    """Obtain city information from .txt files in Berkeley data, and save as json.

    """
    # find all .txt file
    txt_files = glob(data_folder + '*.txt')
    cities_info = []
    for file in txt_files:
        # inspecting the top of the files
        with open(file, 'r') as f:
            city_info = {}
            for i in range(9):
                line = f.readline()
                # remove %
                line = line.replace('% ', '')
                line = line.replace('\n', '')
                k, v = line.split(': ')
                city_info[k] = v

        cities_info.append(city_info)

    with open(data_folder + 'cities_info.json', 'w', encoding='utf8') as f:
        json.dump(cities_info, f)


def update_last_air4Thai(
        url: str = 'http://air4thai.pcd.go.th/webV2/history/',
        data_folder: str = '../data/air4thai_hourly/'):
    """Scrape new air4Thai data.

    Append new data to an exsiting file. Create a new file if the file does not exist.

    """
    print('download more pollution data from Thailand PCD')
    # use Firefox to open the website
    browser = webdriver.Firefox()
    browser.get(url)
    time.sleep(1)
    # find all station names and parameters from the webpage
    page = browser.page_source
    soup = BeautifulSoup(page, features="lxml")

    # extract statopm name and selector
    sta_selector_list, station_name_list = extract_stations(soup)
    # extract pollution parameters
    para_selector_list, para_name_list = extract_parameters(soup)
    print(sta_selector_list)
    # print(para_selector_list)
    assert os.path.exists(data_folder), f'no data folder {data_folder}'

    for sta_id, sta_name in tqdm(zip(sta_selector_list, station_name_list)):
        get_station_data_save(
            url,
            browser,
            sta_id,
            sta_name,
            para_selector_list,
            data_folder)

    browser.close()


def get_station_data_save(
        url,
        browser,
        sta_id,
        sta_name,
        para_selector_list,
        data_folder):
    """ Display the data in para_selector_list for the corresponding station id (sta_id).
    #. Parse the data into the dataframe and save in the data_folder
    #. add datetime columns
    #. load existing file and get the last time stamp
    #. keep the data from the last timestamp
    #. save by appending to the old file

    """

    # display the data on the webpage
    select_data(url, browser, sta_id, para_selector_list, wait_time=5)
    # parse data into dataframe
    data = extract_data(browser)
    # add station id and station name
    data['station_id'] = sta_id
    data['station_name'] = sta_name

    # add datetime columns
    data = make_datetime(data)
    filename = data_folder + sta_id + '.csv'
    # check the last time from exisiting file if exists
    last_time = get_last_datetime(filename)
    # keep only the data after the timestamp
    data = data[data['datetime'] > last_time]

    # save the data
    if os.path.exists(filename):
        # file already exists, append the data
        data.to_csv(filename, mode='a', header=False, index=False)
    else:
        # file does not exist, create the file
        print('create new', filename)
        data.to_csv(filename, index=False)

    temp = pd.read_csv(filename)


def extract_stations(soup):
    """ Extract station selector and station name

    """
    station_list_html = soup.find_all(attrs={'id': 'station_name'})[0]
    station_children = station_list_html.findChildren(
        'option', recursive=False)
    station_selector_list = []
    station_name_list = []
    for station_child in station_children:
        station_selector_list.append(station_child.attrs['value'])
        station_name_list.append(station_child.string)

    return station_selector_list, station_name_list


def extract_parameters(soup):
    """Find all pollutant choices and their selectors

    """
    para_list_html = soup.find_all(attrs={'id': 'parameter_name'})[0]
    para_children = para_list_html.findChildren('option', recursive=False)

    para_selector_list = []
    para_name_list = []
    for child in para_children:
        para_selector_list.append(child.attrs['value'])
        para_name_list.append(child.string)
    return para_selector_list, para_name_list


def select_data(url, browser, sta_id, para_selector_list, wait_time=5):
    """Select station_name (sta_id) and all parameters in para_selector_list on the webpage and display it. And wait for wait_time.

    """
    # select station id
    browser.get(url)
    time.sleep(wait_time)
    station = Select(browser.find_element_by_css_selector(
        'select[id="station_name"]'))
    station.select_by_value(sta_id)
    # select station id

    parameter = Select(browser.find_element_by_css_selector(
        'select[id="parameter_name"]'))
    for para_name in para_selector_list:
        parameter.select_by_value(para_name)

    # click to display data
    button = browser.find_element_by_id('table_bt')
    button.click()
    time.sleep(wait_time)


def extract_data(browser):
    """ Click all the link on the selenium browser object.
    Parse the html table to panda dataframe. concatentate all the dataframe in each page.
    Return dataframe of all the data for that station

    """

    # find number of pages to click
    page = browser.page_source
    soup = BeautifulSoup(page, features="lxml")
    num_click_nodes = soup.find_all(attrs={'aria-controls': "table1"})[-2]
    num_click = int(num_click_nodes.string)

    data_all = pd.DataFrame()
    for i in range(num_click):
        # extract table from page
        page = browser.page_source
        soup = BeautifulSoup(page, features="lxml")
        df = pd.read_html(str(soup))[2]
        df = df.set_index('No.')

        # append to the data
        data_all = pd.concat([data_all, df])
        # click the next page except for the last page
        if i < num_click:
            next_button_head = browser.find_element_by_id('table1_next')
            next_button = next_button_head.find_elements_by_css_selector(
                "*")[0]
            next_button.click()

    return data_all


def make_datetime(stat_df):
    # create datetime columns from df scraped from website

    # replace thai date time name with english
    stat_df.columns = stat_df.columns.str.replace('วันที่', 'date')
    stat_df.columns = stat_df.columns.str.replace('ช่วงเวลา', 'time_range')

    # split data
    stat_df['startTime'] = stat_df['time_range'].str.split('-', expand=True)[0]
    stat_df['datetime'] = stat_df['date'] + ' ' + stat_df['startTime']
    stat_df['datetime'] = pd.to_datetime(
        stat_df['datetime'], format='%Y-%m-%d %H:%M ')

    return stat_df


def get_last_datetime(filename, chunksize=500):
    # load file in chunk of chunksize, retrun the last datetime value in datetime format
    # if the filename does not exist return old time
    # load file in chunk to minimize memory use
    if os.path.exists(filename):

        last_time = None
        for chunk in pd.read_csv(filename, chunksize=chunksize):
            last_time = chunk['datetime'].iloc[-1]
            last_time = pd.to_datetime(last_time)
    else:
        last_time = pd.to_datetime('1800-01-01 00:00:00')

    return last_time


def download_cdc_data(
        station_url: str = 'https://www.cmuccdc.org/api/ccdc/stations',
        dl_url: str = 'https://www.cmuccdc.org/download_json/',
        data_folder: str = '../data/cdc_data/'):
    """Download cdc data and stations info.

    """
    print('download data from Chiang Mai University Project (CDC)')

    try:
        # obtain station info from the API, if possible
        station_info_list = requests.get(station_url).json()
        print('number of stations', len(station_info_list))
        # save station info json
        with open(data_folder + 'station_info.json', 'r') as f:
            json.dump(station_info_list, f)
    except BaseException:

        with open(data_folder + 'station_info.json', 'r') as f:
            station_info_list = json.load(f)

    # download data for all station
    for station_dict in tqdm(station_info_list):

        station_id = station_dict['dustboy_id']
        # download the data
        download_url = dl_url + station_id
        try:
            data_json = requests.get(download_url)
            # extract the json part
            data_dict = data_json.json()[0]
            # print(station_id)
        except BaseException:
            pass
        else:
            # parse to panda dataframe
            data_df = pd.DataFrame.from_dict(data_dict['value'])
            filename = data_folder + station_id + '.csv'
            data_df.to_csv(filename, index=False)


def download_us_emb_data(
        data_folder: str = '../data/us_emb/',
        year: int = None):
    """Download pollution data taken at the US Embabby in Hanoi and Jakata

    """
    if year is None:
        year = datetime.now().year

    city_list = ['Hanoi', 'JakartaSouth', 'JakartaCentral']
    print(f'\n Download us embassy data for Hanoi and Jakata for {year}')

    for city in city_list:
        filename = f'{data_folder}{city}_PM2.5_{year}_YTD.csv'
        if os.path.exists(filename):
            os.remove(filename)
        url = f'http://dosairnowdata.org/dos/historical/{city}/{year}/{city}_PM2.5_{year}_YTD.csv'
        wget.download(url, filename)


def main(
        main_folder: str = '../data/',
        cdc_data=True,
        build_json: bool = False):
    """
    Args:
        main_folder: main data_folder
        build_json: if True also build city information

    """

    # gather all data for Thailand
    download_b_data(
        data_folder=f'{main_folder}pm25/',
        url='http://berkeleyearth.lbl.gov/air-quality/maps/cities/Thailand/')

    print('\n Download Data for Hanoi, Ha dong and Jarkata')
    download_province_data(
        grab_url='http://berkeleyearth.lbl.gov/air-quality/maps/cities/Viet_Nam/Ha_Noi/',
        data_folder=f'{main_folder}pm25/')
    download_province_data(
        grab_url='http://berkeleyearth.lbl.gov/air-quality/maps/cities/Indonesia/Jakarta/',
        data_folder=f'{main_folder}pm25/')

    if build_json:
        # Build City info json for Berkeley Data
        get_city_info(data_folder=f'{main_folder}pm25/')
        # Load Air4 Thai station info
        station_info = requests.get(
            'http://air4thai.pcd.go.th/services/getNewAQI_JSON.php').text
        station_info = json.loads(station_info)
        with open(f'{main_folder}/aqm_hourly2/stations_locations.json', 'w') as f:
            json.dump(station_info, f)

    download_us_emb_data(data_folder=f'{main_folder}us_emb/')

    update_last_air4Thai(
        url='http://air4thai.pcd.go.th/webV2/history/',
        data_folder=f'{main_folder}air4thai_hourly/')

    if cdc_data:
        download_cdc_data(
            station_url='https://www.cmuccdc.org/api/ccdc/stations',
            dl_url='https://www.cmuccdc.org/download_json/',
            data_folder=f'{main_folder}cdc_data/')

    # extract station information
    print('Update weather data for all cities')
    city_names = [
        'Mueang Chiang Mai',
        'Soc Son',
        'Bangkok',
        'Mueang Chiang Rai',
        'Mueang Tak',
        'Yangon',
        'Tada-U',
        'Sikhottabong',
        'Luang Prabang District',
        'Kunming']
    w_folder = f'{main_folder}weather_cities/'
    weather_station_info = find_weather_stations(
        city_names, weather_json_file=w_folder + 'weather_station_info.json')
    len(weather_station_info)

    for city_json in tqdm(weather_station_info):
        print('update weather data for ', city_json['city_name'])
        start_date = datetime(2020, 1, 1)
        end_date = datetime.now()
        update_weather(
            city_json,
            data_folder=w_folder,
            start_date=start_date,
            end_date=end_date)


if __name__ == '__main__':

    main()
