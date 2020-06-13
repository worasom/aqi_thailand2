# -*- coding: utf-8 -*-
from ..imports import *

"""Scrape weather data and weather information

"""

# find station in the city name


def find_weather_stations(city_names: list, weather_json_file: str):
    """Find a list of weather stations in city_names list
    """
    # load weather information
    with open(weather_json_file, 'r') as f:
        station_dict_list = json.load(f)

    weather_stations_info = []
    for city_name in city_names:
        for i, station in enumerate(station_dict_list):
            if city_name in station['city_name']:
                weather_stations_info.append(station)

    return weather_stations_info


def get_data_n_soup(browser, date_str, header_url, waittime=30):
    ''' Input: date in string
    - Ask Selenium to open the website, and execute inner javascript.
    - Parse data into beautifulsoup object and look for the hourly table
    - Parse the table into a panda dataframe
    - remove the unit
    - add date column

    return: daily weather dataframe and beauitfulsoup object of that table
    '''
    # url=f'https://www.wunderground.com/history/daily/th/bang-phut/VTBD/date/{date_str}'
    url = header_url + date_str
    # print(url)
    browser.get(url)
    time.sleep(waittime)
    innerhtml = browser.execute_script("return document.body.innerHTML")
    soup = BeautifulSoup(innerhtml, features="lxml")
    div_table = soup.find_all('table')
    daily_df = pd.read_html(str(div_table))[-1]

    # add date columns
    daily_df['datetime'] = pd.to_datetime(
        date_str + ' ' + daily_df['Time'],
        format="%Y-%m-%d %I:%M %p")
    return daily_df, div_table


def convert_temp_col(data_df, temperature_col):
    # convert string temperature in F to celcious in float
    for col in temperature_col:
        if col in data_df.columns:
            data_series = data_df[col].copy()
            data_series = data_series.str.replace('F', '')
            data_series = data_series.astype(float)
            data_series = ((data_series - 32) * 5 / 9).round(2)
            data_df[col] = data_series
            data_df.columns = data_df.columns.str.replace(col, col + '(C)')

    return data_df


def convert_wind_col(data_df, win_col):
    # convert string wind speed and wind gust in mph to kph
    for col in win_col:
        if col in data_df.columns:
            data_series = data_df[col].copy()
            # remove unit in the data
            data_series = data_series.str.replace('mph', '')
            data_series = data_series.str.replace(',', '')
            data_series = data_series.astype(float)
            # convert the value
            data_series = (data_series * 1.60934).round(0)
            data_df[col] = data_series
            data_df.columns = data_df.columns.str.replace(col, col + '(kmph)')

    return data_df


def convert_pressure_col(data_df, pressure_col):
    # convert string pressure in 'inch' to float
    for col in pressure_col:
        if col in data_df.columns:
            data_series = data_df[col].copy()
            data_series = data_series.str.replace('in', '')
            data_series = data_series.astype(float)
            data_df[col] = data_series
            data_df.columns = data_df.columns.str.replace(col, col + '(in)')

    return data_df


def convert_humidity_col(data_df, humidity_col):
    # convert string temperature in F to celcious in float
    for col in humidity_col:
        if col in data_df.columns:
            data_series = data_df[col].copy()
            data_series = data_series.str.replace('%', '')
            data_series = data_series.astype(int)
            data_df[col] = data_series
            data_df.columns = data_df.columns.str.replace(col, col + '(%)')

    return data_df


def convert_unit(data_df):
    # convert string data into number by removing the text in the unit. Put the text in the columns name.
    # convert temperature wand windspeed into metric system
    temperature_col = ['Temperature', 'Dew Point']
    wind_col = ['Wind Speed', 'Wind Gust']
    pressure_col = ['Pressure', 'Precip.']
    humidity_col = ['Humidity']
    data_df = convert_temp_col(data_df, temperature_col)
    data_df = convert_wind_col(data_df, wind_col)
    data_df = convert_pressure_col(data_df, pressure_col)
    data_df = convert_humidity_col(data_df, humidity_col)
    return data_df


def scrape_weather(city_json, date_range):
    # scrape weather data from a city in date in date range
    # save filename using city name
    # return weather data and bad data df and date that fail
    # append the data to an exsiting file if the files does not exists
    browser = webdriver.Firefox()
    time.sleep(2)
    weather = pd.DataFrame()
    bad_date_df = pd.DataFrame()

    # Build header URL
    specific_url = city_json['specific_url']
    header_url = 'https://www.wunderground.com/history/daily/' + specific_url + 'date/'

    for i, date in tqdm(enumerate(date_range)):

        try:
            # obtain daily weather dataframe
            daily_df, div_table = get_data_n_soup(
                browser, date, header_url=header_url, waittime=25)

        except BaseException:
            # fail query,
            bad_date_df = pd.concat(
                [
                    bad_date_df,
                    pd.DataFrame(
                        {
                            'header_url': 'https://www.wunderground.com/history/daily/th/mueang-chiang-mai/VTCC/date/',
                            'date': date},
                        index=[0])],
                ignore_index=True)

        else:
            if len(daily_df) == 0:
                # fail query,
                bad_date_df = pd.concat(
                    [
                        bad_date_df,
                        pd.DataFrame(
                            {
                                'header_url': 'https://www.wunderground.com/history/daily/th/mueang-chiang-mai/VTCC/date/',
                                'date': date},
                            index=[0])],
                    ignore_index=True)
            else:
                # good query
                # convert unit of the data
                daily_df = convert_unit(daily_df)
                # combine the weather for each day
                weather = pd.concat([weather, daily_df], axis=0, join='outer')

    browser.close()
    try: 
        # sort weather value
        weather = weather.sort_values('datetime')
    except:
        print(date_range, weather.columns)

    return weather, bad_date_df


def fix_temperature(df, lowest_t: int = 5, highest_t: int = 65):
    # remove abnormal tempearture reading from weather data

    idx = df[df['Temperature(C)'] < lowest_t].index
    df.loc[idx, ['Temperature(C)', 'Dew Point(C)', 'Humidity(%)']] = np.nan

    idx = df[df['Temperature(C)'] > highest_t].index
    df.loc[idx, ['Temperature(C)', 'Dew Point(C)', 'Humidity(%)']] = np.nan

    return df


def fill_missing_weather(df, limit: int = 12):
    # make the timestamp to be 30 mins interval. Fill the missing value
    # roud datetiem to whole 30 mins
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['datetime'] = df['datetime'].dt.round('30T')
    df = df.sort_values('datetime')
    df = df.drop_duplicates('datetime')

    dates = df['datetime'].dropna().dt.date.unique()

    # fill in the missing value
    new_datetime = pd.date_range(
        start=dates[0], end=dates[-1] + timedelta(days=1), freq='30T')
    new_weather = pd.DataFrame(new_datetime[:-1], columns=['datetime'])
    new_weather = new_weather.merge(df, on='datetime', how='outer')
    new_weather = new_weather.fillna(method='ffill', limit=12)
    new_weather = new_weather.fillna(method='bfill', limit=12)
    new_weather = new_weather.set_index('datetime')
    new_weather = new_weather.dropna(how='all').reset_index()

    return new_weather


def update_weather(
        city_json,
        data_folder,
        start_date=datetime(
            2000,
            10,
            1),
        end_date=datetime.now()):
    """Update weather for the city specified by city_json and save.

    """

    # read existing file
    city_name = ('_').join(city_json['city_name'].split(' '))
    current_filename = data_folder + city_name + '.csv'
    print('updateing file:', current_filename)

    # obtain a list of existed dates if exists
    if os.path.exists(current_filename):
        df = pd.read_csv(current_filename)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.drop_duplicates('datetime')
        # find exisiting date
        ex_date = df['datetime'].dt.strftime('%Y-%m-%d').unique()
        ex_date = set(ex_date)
    else:
        df = pd.DataFrame()
        ex_date = {}

    # calculate the missing dates
    date_range = pd.date_range(start_date, end_date).strftime('%Y-%m-%d')
    missing_date = sorted(set(date_range).difference(ex_date))
    print('missing date', len(missing_date))

    # obtain new  data
    new_weather, _ = scrape_weather(city_json, date_range=missing_date)

    # fix bad temperature data and missing timestamp
    new_weather = fix_temperature(new_weather)
    new_weather = fill_missing_weather(new_weather)

    # merge to existing value
    df = pd.concat([df, new_weather], ignore_index=True)
    df = df.sort_values('datetime')
    df = df.drop_duplicates('datetime')
    df.to_csv(current_filename, index=False)
