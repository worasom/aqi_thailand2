# -*- coding: utf-8 -*-
from ..imports import *

def get_data_n_soup(browser, date_str,header_url,waittime=30):
    ''' Input: date in string
    - Ask Selenium to open the website, and execute inner javascript.
    - Parse data into beautifulsoup object and look for the hourly table
    - Parse the table into a panda dataframe
    - remove the unit 
    - add date column
    
    return: daily weather dataframe and beauitfulsoup object of that table
    '''
    #url=f'https://www.wunderground.com/history/daily/th/bang-phut/VTBD/date/{date_str}'
    url = header_url+date_str
    #print(url)
    browser.get(url)
    time.sleep(waittime)
    innerhtml= browser.execute_script("return document.body.innerHTML")
    soup = BeautifulSoup(innerhtml,features="lxml")
    div_table = soup.find_all('table')
    daily_df = pd.read_html(str(div_table))[-1]
     
    
    # add date columns
    daily_df['datetime'] = pd.to_datetime(date_str + ' ' +daily_df['Time'], format="%Y-%m-%d %I:%M %p")
    return daily_df, div_table

def convert_temp_col(data_df, temperature_col):
    # convert string temperature in F to celcious in float 
    for col in temperature_col:
        if col in data_df.columns:
            data_series = data_df[col].copy()
            data_series = data_series.str.replace('F','')
            data_series = data_series.astype(float)
            data_series = ((data_series - 32)*5/9).round(2)
            data_df[col] = data_series
            data_df.columns = data_df.columns.str.replace(col, col+'(C)')
            
    return data_df

def convert_wind_col(data_df, win_col):
    # convert string temperature in F to celcious in float 
    for col in win_col:
        if col in data_df.columns:
            data_series = data_df[col].copy()
            data_series = data_series.str.replace('mph','')
            data_series = data_series.str.replace(',','')
            data_series = data_series.astype(int)
            data_df[col] = data_series
            data_df.columns = data_df.columns.str.replace(col, col+'(mph)')
            
    return data_df

def convert_pressure_col(data_df, pressure_col):
    # convert string temperature in F to celcious in float 
    for col in pressure_col:
        if col in data_df.columns:
            data_series = data_df[col].copy()
            data_series = data_series.str.replace('in','')
            data_series = data_series.astype(float)
            data_df[col] = data_series
            data_df.columns = data_df.columns.str.replace(col, col+'(in)')
            
    return data_df

def convert_humidity_col(data_df, humidity_col):
    # convert string temperature in F to celcious in float 
    for col in humidity_col:
        if col in data_df.columns:
            data_series = data_df[col].copy()
            data_series = data_series.str.replace('%','')
            data_series = data_series.astype(int)
            data_df[col] = data_series
            data_df.columns = data_df.columns.str.replace(col, col+'(%)')
            
    return data_df

def convert_unit(data_df):
    temperature_col = ['Temperature','Dew Point']
    wind_col = ['Wind Speed','Wind Gust']
    pressure_col = ['Pressure','Precip.']
    humidity_col = ['Humidity']
    data_df = convert_temp_col(data_df, temperature_col)
    data_df = convert_wind_col(data_df, wind_col)
    data_df = convert_pressure_col(data_df,pressure_col)
    data_df = convert_humidity_col(data_df, humidity_col)
    return data_df

def scrape_weather( city_json, date_range, data_folder):
    # scrape weather data from a city in date in date range 
    # save filename using city name 
    # return bad data df and date that fail 
    # append the data to an exsiting file if the files does not exists
    browser = webdriver.Firefox()
    time.sleep(2)
    weather = pd.DataFrame()
    bad_date_df = pd.DataFrame()
    
    # Build header URL 
    specific_url = city_json['specific_url']
    header_url = 'https://www.wunderground.com/history/daily/' + specific_url + 'date/'
    city_name = specific_url.split('/')[1]
    
    # build filename 
    filename = data_folder + city_name+f'_weather.csv'
    if os.path.exists(filename):
        filename = filename.replace('.csv','_1.csv')
    print(filename, date_range[0])
    
    for i, date in enumerate(date_range):
    
        try:
            # obtain daily weather dataframe
            daily_df, div_table = get_data_n_soup(browser,date,header_url=header_url,waittime=30)
        
        except:
            # fail query, 
            bad_date_df = pd.concat([bad_date_df, pd.DataFrame({'header_url':'https://www.wunderground.com/history/daily/th/mueang-chiang-mai/VTCC/date/',
                                                    'date': date},index=[0])],ignore_index=True)
        
        else: 
            if len(daily_df)==0:
                    # fail query, 
                bad_date_df = pd.concat([bad_date_df, pd.DataFrame({'header_url':'https://www.wunderground.com/history/daily/th/mueang-chiang-mai/VTCC/date/',
                                                    'date': date},index=[0])],ignore_index=True)
            else:
                # good query 
                # convert unit of the data 
                daily_df = convert_unit(daily_df)
                #combine the weather for each day
                weather = pd.concat([weather,daily_df], axis=0, join='outer')
        # save the data ocationally    
        if i%10==0:
            weather.to_csv(filename,index=False)
    # save everything
    weather.to_csv(filename,index=False)
    browser.close()

