<h1>Table of Contents<span class="tocSkip"></span></h1>
<div class="toc"><ul class="toc-item"><li><span><a href="#Model-Air-Pollution-Thailand-and-South-East-Asian-Countries" data-toc-modified-id="Model-Air-Pollution-Thailand-and-South-East-Asian-Countries-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Model Air Pollution Thailand and South East Asian Countries</a></span></li><li><span><a href="#Requirements" data-toc-modified-id="Requirements-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Requirements</a></span></li><li><span><a href="#Directory-Tree" data-toc-modified-id="Directory-Tree-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Directory Tree</a></span></li><li><span><a href="#Data-Sources" data-toc-modified-id="Data-Sources-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Data Sources</a></span><ul class="toc-item"><li><span><a href="#Pollution-Data" data-toc-modified-id="Pollution-Data-4.1"><span class="toc-item-num">4.1&nbsp;&nbsp;</span>Pollution Data</a></span></li><li><span><a href="#Weather-Data" data-toc-modified-id="Weather-Data-4.2"><span class="toc-item-num">4.2&nbsp;&nbsp;</span>Weather Data</a></span></li><li><span><a href="#Hotspot-Data" data-toc-modified-id="Hotspot-Data-4.3"><span class="toc-item-num">4.3&nbsp;&nbsp;</span>Hotspot Data</a></span></li></ul></li><li><span><a href="#AQI-Convention" data-toc-modified-id="AQI-Convention-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>AQI Convention</a></span></li><li><span><a href="#Modeling-Air-Pollution-in-Chiang-Mai-Data-:-A-Case-Study" data-toc-modified-id="Modeling-Air-Pollution-in-Chiang-Mai-Data-:-A-Case-Study-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Modeling Air Pollution in Chiang Mai Data : A Case Study</a></span><ul class="toc-item"><li><span><a href="#Casual-Diagram" data-toc-modified-id="Casual-Diagram-6.1"><span class="toc-item-num">6.1&nbsp;&nbsp;</span>Casual Diagram</a></span></li><li><span><a href="#A-Dataset-Object" data-toc-modified-id="A-Dataset-Object-6.2"><span class="toc-item-num">6.2&nbsp;&nbsp;</span>A Dataset Object</a></span></li><li><span><a href="#Exploratory-Data-Analysis" data-toc-modified-id="Exploratory-Data-Analysis-6.3"><span class="toc-item-num">6.3&nbsp;&nbsp;</span>Exploratory Data Analysis</a></span><ul class="toc-item"><li><span><a href="#Geography" data-toc-modified-id="Geography-6.3.1"><span class="toc-item-num">6.3.1&nbsp;&nbsp;</span>Geography</a></span></li><li><span><a href="#PM2.5-Pollution" data-toc-modified-id="PM2.5-Pollution-6.3.2"><span class="toc-item-num">6.3.2&nbsp;&nbsp;</span>PM2.5 Pollution</a></span></li><li><span><a href="#Agricultural-Burning" data-toc-modified-id="Agricultural-Burning-6.3.3"><span class="toc-item-num">6.3.3&nbsp;&nbsp;</span>Agricultural Burning</a></span></li></ul></li><li><span><a href="#Model-Optimization" data-toc-modified-id="Model-Optimization-6.4"><span class="toc-item-num">6.4&nbsp;&nbsp;</span>Model Optimization</a></span><ul class="toc-item"><li><span><a href="#Training" data-toc-modified-id="Training-6.4.1"><span class="toc-item-num">6.4.1&nbsp;&nbsp;</span>Training</a></span></li><li><span><a href="#Model-Performance" data-toc-modified-id="Model-Performance-6.4.2"><span class="toc-item-num">6.4.2&nbsp;&nbsp;</span>Model Performance</a></span></li></ul></li><li><span><a href="#Simulation" data-toc-modified-id="Simulation-6.5"><span class="toc-item-num">6.5&nbsp;&nbsp;</span>Simulation</a></span></li></ul></li><li><span><a href="#Conclusions" data-toc-modified-id="Conclusions-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Conclusions</a></span></li></ul></div>

# Model Air Pollution Thailand and South East Asian Countries

This project aims to use a machine learning model predict and identify sources of the air pollution South East Asian cities. Because this is a multi-variables problem with complex interaction among features, and time-lag, a machine learning approch has an advantage over a traditional approach. In this study, random forest regressor(RF) is used to model a small particle pollution(PM2.5) level. After trying searching various machine learning model, I found that RF perform the best. In addition, the model can be easily interpreted. The model's feature of importances in combination data exploration helps identify the major sources of air pollution. The trained model is used to simulate the pollution level when an environmental policies are implemented. Here, I provide a case study for Chiang Mai, but the codes works with other cities such as Bangkok, and Hanoi (Given there are enough data, of course!)

To explain how to use my code to analyze the air pollution, I first explain the [packages](#Requirements-2) I used and the [directory tree](#Directory-Tree-3). Then I will described how to obtain [data](#Data-Sources-4). [AQI convention](#AQI-Convention-5) section explains color label used in this study. [An Analysis for Chiang Mai](#Modeling-Air-Pollution-in-Chiang-Mai-Data-:-A-Case-Study-6) is splits into [Exploring the data](Exploratory-Data-Analysis-6.3), [model optimization](Model-Optimization-6.4) and [simulation](#Simulation-6.5). I will include result figures and the code I used to generate the figures if applicable. The analysis results for Chiang Mai is summarized in the [conclusions](#Conclusions-8) 

# Requirements

numpy==1.18.1<br>
matplotlib==3.1.2<br>
pandas==1.0.0<br>
geopandas==0.6.2<br>
scikit_optimize==0.7.4 <br>
scikit_learn==0.23.2<br>
TPOT==0.11.5<br>
statsmodels==0.11.1<br>
scipy==1.4.1<br>
seaborn==0.10.0<br>
joblib==0.14.1<br>
tqdm==4.43.0<br>
Shapely==1.7.0<br>
pyproj==2.4.2.post1<br>
Fiona==1.8.13<br>
bokeh==2.1.1<br>
selenium==3.141.0 <br>
wget==3.2<br>
beautifulsoup4==4.9.1<br>
requests==2.22.0<br>
swifter==1.0.3<br>
Sphinx>=1.6.0<br>

# Directory Tree

<pre>
├── README.md   
├── requirements.txt : generated by pipreqs
├── data : raw data and processed data for each cities. Please see the data section for details about how to obtains the raw data
├── docs._build.html : code documentations generated by Sphinx 
├── models : each subfolder contains a model for a each city.  
│   ├──  chiang_mai : contains random forest models for Chiang Mai and model meta file containing setting
│   └── bangkok   
├── reports : plots for each city 
│   ├── chiang_mai : data and model visualizations for Chiang Mai
│   └── bangkok   
├── notebooks   
│   ├── 1_pollutions_data.ipynb : 
│   ├── 1.1_vn_power_plants.ipynb : 
│   ├── 2_analyze_pollution_data.ipynb : 
│   ├── 5.0-ML_Chiang_mai.ipynb : 
│   ├── 6.0_vis_ChiangMai.ipynb : 
│   ├── 6.1_BKK.ipynb : 
│   ├── 6.2_vis_Jarkata.ipynb : 
│   ├── 6.3_Hanoi.ipynb : 
│   └── 7_prediction.ipynb : 
│   
└── src : the source codes are meant to be ran as a module not as .py (except for vn_data.py) 
    ├── imports.py : 
    ├── gen_functions.py : general purpose functions such as color setting, AQI conversion and coordinate Conversion
    ├── data : download and preprocess data 
    │   ├── download_data.py :  download pollution data from various sources 
    │   ├── read_data.py :  read pollution data 
    │   ├── vn_data.py : scrape pollution data from Vietnamese EPA
    │   ├── weather_data.py : scrape, process and load weather data
    │   └── fire_data.py : process and load hotspots data
    │     
    ├── features   
    │   ├── build_features.py : 
    │   └── dataset.py : Dataset object is responsible for putting raw data together, 
    │                    feature engineering and preparing matricies for machine learning models. 
    │                    Call read_data.py when loading data
    │                    Call build_features.py for feature engineering functions 
    ├── models   
    │   ├── train_model.py :  model builder and hyperparameter searcher
    │   └── predict_model.py : load model and perform statistical simulations
    │   
    └── visualization  
        ├── vis_data.py : create plots for data exploration steps   
        └── vis_model.py : create plots for visualizing model performance and simulation 
</pre>

# Data Sources

## Pollution Data
 - [Thailand Pollution Department](http://air4thai.pcd.go.th/webV2/) Data only go back 2 months! so need to run scapper once a month using `src.data.download_data.update_last_air4Thai()` . Once can also writes a letter to ask for historical data directly. The request took about a month to process. The data has to be parsed from their excel files.  
 - [Vietnamese Pollution Department](http://enviinfo.cem.gov.vn/) Pollution data for major cities such as Hanoi. Data only go back 24 hours, so need to run scapper once a day. This is done using using `src.data.vn_data.download_vn_data()`
 - [Berkeley project](http://berkeleyearth.org/) provides historical PM2.5 data back until late 2016 
 - [US Embassy](http://dosairnowdata.org/dos/historical/) US embassy in some cities collect PM2.5 data. Use `src.data.download_data.download_us_emb_data()` to download data. 
 - [Chiang Mai University Monitoring Stations](https://www.cmuccdc.org/) provides data from the University monitoring stations in the northern part of Thailand. Use `src.data.download_data..download_cdc_data()` to download this data.  
 - [The World Air
Quality Project](https://aqicn.org/) has pollutions data from many cities, but only provide daily average data  

## Weather Data 

Weather data is from two sources. 

- [OpenWeathermap](https://openweathermap.org/history) provides a bulk historial data to purchase. The data is processed using `src.data.weather_data.proc_open_weather()`
- Additional weather data is constantly scraped from [Weather Underground)(https://www.wunderground.com/) `src.data.weather_data.update_weather()` 

Both the pollution data and weather data are can be downloaded using one single command `src.data.download_data.main()`


## Hotspot Data 

Satellite data showing location of burning activities(hotspots) from NASA https://firms.modaps.eosdis.nasa.gov/download/. There are three data product. This study uses MODIS Collection 6 data because of the data available for a longer period that other products.

# AQI Convention

Countries have different standards, which convert the raw polution readings into air quality index(AQI) and interpret the harmful levels. The maximum AQIs for each pollutants (PM2.5, Pm10, SO2 etc) is reported at a single AQI. The figure belows compare US and Thailand AQI. The standards are comparable except for SO$_2$, where the US has a stricker standard. This study use US AQI conversion standard for calculating AQI for different pollutants. For example, in the PM2.5 case 
- 0 - 50 AQI is in a healthy range. This corresponds to PM2.5 level between 0 - 12 $\mu g/m^3$ (green). 
- 50 - 100 AQI is a moderate range, corresponding to PM2.5 12- 35.4  $\mu g/m^3$ (orange)
- 100 - 150 AQI is a unhealthy for sensitive group, corresponding to PM2.5 35.5- 55.4  $\mu g/m^3$ (red)
- 150 - 200 AQI is a unhealthy range, corresponding to PM2.5 55.5- 150.4  $\mu g/m^3$ (red)

For simplicity, the color code in this study groups the  100 - 150 and 150 - 200 as unhealthy range(red). 


```python
#  Load the "autoreload" extension so that code can change
%reload_ext autoreload
%autoreload 2
import src
```


```python
src.visualization.vis_data.compare_aqis(filename='../reports/chiang_mai/aqi.png')
```

![png](https://github.com/worasom/aqi_thailand2/blob/master/reports/chiang_mai/aqi.png)

#  Modeling Air Pollution in Chiang Mai Data : A Case Study

## Casual Diagram

There are four possible sources of air pollution: local traffic, power plants, industrial activities, and agricultural burning. For the latter three sources, the pollution is generated from the other provinces. In all of these sources, the local weather (temperature, humidity, and wind speed) decides how long the pollution stays in the air, and thus the pollution level. 

Since each pollution source results in different seasonal pollution patterns and chemical characteristics(chemical finger print), inspecting the pollutions data, weather, and several burning hotspots seen from the satellite images and the patterns could help identify or rule out air pollution contribution factors. This is done in [exploratory data analysis section](#Exploratory-Data-Analysis-6.2).


<img src="https://github.com/worasom/aqi_thailand2/blob/master/reports/chiang_mai/casual_di.PNG" width="500"/>

## A Dataset Object

It is more convenience to have a `Dataset` object that keep tracks of all relavant data for a city along with necessary meta information such as city location etc. This is object is under `src.features.dataset.py`.

The `Dataset` object is also in charge of compile raw pollution, weather, fire data from the data folder into a ready-to-use format. The processed data are saved under ../data/city_name/. The code below illustrates how to `Dataset` object compile the data using a build_all_data command. This object also keep track of feature engineering parameters during [model optmization](#Training-7.1) step. 


```python
# init a dataset object and build the data from scratch 
# only perform this when new data files are added 
dataset = src.features.dataset.Dataset('Chiang Mai')

# build pollution,  weather data and (optional) fire data
dataset.build_all_data( build_fire=True, build_holiday=True)
```

After the building process, which might take sometimes because of the size of the fire data (building the fire data is optional and can be set to false (`build_fire=False`). The complied data can be loaded using `_load()` command.


```python
# reinit the data and load saved process data 
dataset = src.features.dataset.Dataset('Chiang Mai')
dataset.load_()
```

The hourly pollution data, weather data, and fire data are under `dataset.poll_df`, `dataset.wea` and `dataset.fire` attributes accordingly. Each data is a panda dataframe with datetime index. For example, the pollution data for Chiang Mai looks like


```python
print(dataset.poll_df.tail(2).to_markdown())
```

    | datetime            |   PM2.5 |   PM10 |   O3 |   CO |   NO2 |   SO2 |
    |:--------------------|--------:|-------:|-----:|-----:|------:|------:|
    | 2020-06-17 15:00:00 |     8.5 |   19.5 |   15 | 0.4  |     5 |     1 |
    | 2020-06-17 16:00:00 |     7.5 |   16.5 |   11 | 0.43 |     5 |     1 |
    

The weather data is under `wea` attribute

Additionally the dataset also has city information under `city_info` attribute


```python
dataset.city_info
```




    {'Country': 'Thailand',
     'City': 'Chiang Mai',
     'City (ASCII)': 'Chiang Mai',
     'Region': 'Chiang Mai',
     'Region (ASCII)': 'Chiang Mai',
     'Population': '200952',
     'Latitude': '18.7904',
     'Longitude': '98.9847',
     'Time Zone': 'Asia/Bangkok',
     'lat_km': 2117.0,
     'long_km': 11019.0}



## Exploratory Data Analysis

The first step before any machine learning model is to understand the data. The file `src.visualization.vis_data.py` contains many useful functions for quick data visualization. Here, pollution data in Chiang Mai is used to illustrate visualization functions. 

### Geography

It is important to understand the geography of the city. The picture belows show a map of Chiang Mai. It is a medium size surrounded by high mountains. Locations of the monitoring stations, near by industrial complex and power plants are also shown.

I have two handy functions to convert logtitude, latitude coordinates to Mercator projection and vice versa. They are `src.gen_functions.merc_x`, `src.gen_functions.merc_y`, and `src.gen_functions.to_latlon`

![map of Chiang Mai](https://github.com/worasom/aqi_thailand2/blob/master/reports/chiang_mai/cm_map.png)

### PM2.5 Pollution

![different pollutants](https://github.com/worasom/aqi_thailand2/blob/master/reports/chiang_mai/all_pol_aqi.png)

Seasonal patterns of PM2.5 level(top), number of hotspots within 1000 km from Chiang Mai(middle), and temperature(bottom). The shaded regions are 95% confident interval from different years. (top) the horizontal lines indicate the values corresponded to AQI 100 (moderate) and 150 (unhealthy) accordingly. The number of hotspots has a similar seasonal pattern as the PM2.5’s.

![seasonal pattern](https://github.com/worasom/aqi_thailand2/blob/master/reports/chiang_mai/fire_PM25_t_season.png)

![seasonal pattern](https://github.com/worasom/aqi_thailand2/blob/master/reports/chiang_mai/fire_PM25_season.png)

### Agricultural Burning 

Since the concentration of hotspots differs at various distances from Chiang Mai, it is important to divide the burning activity into distance-based zones. Figure below shows four fire zone. The first zone is within 100 km from Chiang Mai within the Thailand border. The second and third zone is 100 - 200 km and 200 - 400 km from Chiang Mai. A lot of fire activities in this zone are in Myanmar and Laos(red spots). The third zone is between 400 and 700 km from Chiang Mai. Most burning activities here are in Laos.  The last zone is between 700 and 1000 km. This zone also not only includes the burning activities in Myanmar, Laos, and Vietnam but also those in Cambodia. However, most fire activities in Cambodia, which is much further away and the burning activities concentrate in December(blue spots), which is not the month with the highest pollution in Chiang Mai. 

![fire_zone](https://github.com/worasom/aqi_thailand2/blob/master/reports/chiang_mai/fire_zone.png)

## Model Optimization

### Training

Model Optimization breakdown into the following steps

1. Build dataset using default fire parameter. Split the data into train, validation and test set. Optimize for a reasonable RandomForestRegressor model  
2. Remove lower importance features from the model  
3. Optimize for the best fire features
4. Improve model performance by adding lag columns (of weather and fire)
5. Remove lower importance lag columns 
6. Optimize for better model parameters again  
7. Save model and model meta information 

These steps are carried out in a single line of code. The entire optimization tooks about 3 hours to complete in my computer.


```python
dataset, rf_model, model_meta = src.models.train_model.train_city_s1(city='Chiang Mai', pollutant='PM2.5')
```

### Model Performance

Look at the model performance by plotting actual data and the prediction data. In the [simulation](#Simulation-6.5) section, I will be using the daily average of the prediction, therefore I will also look at the prediction error of the daily average data.



![model performance](https://github.com/worasom/aqi_thailand2/blob/master/reports/chiang_mai/PM25_model_perfomance.png)


```python
_, df = plot_model_perf(dataset=data, model=rf_model, split_list=[0.7, 0.3], xlim=[], to_save=False)
```

When loading a model using `src.models.predict_model.load_model`, the function calculate the R2-score for the test set(which is an hourly pollution data). Then, its calculate the R2-score of a daily average of both the training and test data. 

For Chiang Mai, hourly prediction R2-score is 0.71. The daily average test data has R2-score of 0.79. This is pretty good.   


```python
dataset, model, *argv = src.models.predict_model.load_model(city='Chiang Mai', pollutant='PM2.5',split_list=[0.7, 0.3])
```

    data no fire has shape (77747, 14)
    raw model performance {'test_r2_score': 0.7181850826320375, 'test_mean_squared_error': 220.3070792867193, 'test_mean_absolute_error': 9.069185235196183}
    daily avg training error {'avg_trn_r2_score': 0.9056688733734121, 'avg_trn_mean_squared_error': 61.629098357499416, 'avg_trn_mean_absolute_error': 4.718201297339313}
    daily avg test error {'avg_test_r2_score': 0.7880245398552946, 'avg_test_mean_squared_error': 142.22351861838922, 'avg_test_mean_absolute_error': 7.67540429113698}
    

## Simulation


The figure below ranks input’s order of importance. fire_100_200 means the hotspots between 100 – 200 km from Chiang Mai. The fire columns are the most important features indicating that agricultural burning is the major source of air pollution. 

![feature_importance](https://github.com/worasom/aqi_thailand2/blob/master/reports/chiang_mai/PM25_rf_fea_op2_nolag.png)


```python

```

Compare the set aside test data and the simulation. 


![test_data_vs_inference](https://github.com/worasom/aqi_thailand2/blob/master/reports/chiang_mai/test_data_vs_inference.png)


```python

```

Seasonal pattern of the test data and the simulation. The simulation capture the correct seasonal patterns. 

![test_data_vs_inference_season](https://github.com/worasom/aqi_thailand2/blob/master/reports/chiang_mai/test_data_vs_inference_season.png)


```python

```



In the figure below, (top) seasonal pattern of the pollution level when the burning activities are reduced to 100%, 90%, 50%, and 10% in 0 – 100 km radius. 100% means no fire reduction. (bottom) corresponding numbers of hotspots per day in different scenarios. The horizontal lines indicate the moderate(orange) and unhealthy(red) AQIs

![effect_fire_0_700km_sea](https://github.com/worasom/aqi_thailand2/blob/master/reports/chiang_mai/effect_fire_0_700km_sea.png)


```python

```


```python

```

average pollution level in December-April upon reducing the burning activities to different percent. The horizontal lines indicate the moderate(orange) and unhealthy(red) AQIs. 

![effect_of_fire_reduction_3m](https://github.com/worasom/aqi_thailand2/blob/master/reports/chiang_mai/effect_of_fire_reduction_3m.png)


```python

```

Putting togehter, this animation summarized the effect of fire reduction as the radius of fire reduction policy increases.

![animation](https://github.com/worasom/aqi_thailand2/blob/master/reports/ani_plot/eff_reduced_fire1.png)


```python

```

# Conclusions

I study the sources of the air pollution problem in Chiang Mai. The major problem is from the high AQI of PM2.5 pollution. The PM2.5 level has a seasonal pattern with value exceed the moderate AQI between December-April and often exceeds unhealthy AQI in March. Other pollutants such as PM10 and O¬3, also exhibit similar seasonal patterns but with peak AQIs still in the moderate range. By inspecting the seasonal pattern of the pollution level, I rule out traffic, industrial activities, and power plants from the possible sources of air pollution and identify that agricultural burning as the major source. The number of burning hotspots seen from the MODIS satellite has the same seasonal pattern as that of the PM2.5 level. Moreover, in the years with abnormally low burning activities correspond to those with lower pollution levels. The number of burning activities and thus the pollution level is not a result of climate change because of their decreasing yearly trends, which is opposite to the increasing temperature trend. 

To quantify the effect of the burning activities toward the PM2.5 level, I trained a random forest regressor to predict the hourly PM2.5 level with the weather, hotspots within 1000 km from Chiang Mai, and auxiliary date-time information as a tabular input. The hotspots data are divided into distance-based zones to measure the effect of each zone. The model achieved 0.71 R2-score for hourly values in the 3 recent years of unseen dataset, and 0.78 R2-score when considering the daily average values. As expected, the model order of importance ranks the hotspots columns as the tops with the first in 100-200 km zone as most important. The trained model is used to simulate a range of possible pollution levels using the historical input values in the training dataset. The average pollution level has correct seasonal patterns and values as the data in the unseen dataset. The simulated pollution levels under different scenarios with reduced burning activities. The simulation shows that reducing the burning activities within the Thailand border (100 km area from Chiang Mai) alone could not lower to PM2.5 level from the unhealthy AQI, and aggressive reduction in the area covered as far a 1000 km is needed. 

