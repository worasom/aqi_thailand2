<h1>Table of Contents<span class="tocSkip"></span></h1>
<div class="toc"><ul class="toc-item"><li><span><a href="#Model-Air-Pollution-Thailand-and-South-East-Asian-Countries" data-toc-modified-id="Model-Air-Pollution-Thailand-and-South-East-Asian-Countries-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Model Air Pollution Thailand and South East Asian Countries</a></span></li><li><span><a href="#Requirements" data-toc-modified-id="Requirements-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Requirements</a></span></li><li><span><a href="#Directory-Tree" data-toc-modified-id="Directory-Tree-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Directory Tree</a></span></li><li><span><a href="#Data-Sources" data-toc-modified-id="Data-Sources-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Data Sources</a></span><ul class="toc-item"><li><span><a href="#Pollution-Data" data-toc-modified-id="Pollution-Data-4.1"><span class="toc-item-num">4.1&nbsp;&nbsp;</span>Pollution Data</a></span></li><li><span><a href="#Weather-Data" data-toc-modified-id="Weather-Data-4.2"><span class="toc-item-num">4.2&nbsp;&nbsp;</span>Weather Data</a></span></li><li><span><a href="#Hotspot-Data" data-toc-modified-id="Hotspot-Data-4.3"><span class="toc-item-num">4.3&nbsp;&nbsp;</span>Hotspot Data</a></span></li></ul></li><li><span><a href="#AQI-Standards" data-toc-modified-id="AQI-Standards-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>AQI Standards</a></span></li><li><span><a href="#Modeling-Air-Pollution-in-Chiang-Mai-Data-:-A-Case-Study" data-toc-modified-id="Modeling-Air-Pollution-in-Chiang-Mai-Data-:-A-Case-Study-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Modeling Air Pollution in Chiang Mai Data : A Case Study</a></span><ul class="toc-item"><li><span><a href="#Causal-Diagram" data-toc-modified-id="Causal-Diagram-6.1"><span class="toc-item-num">6.1&nbsp;&nbsp;</span>Causal Diagram</a></span></li><li><span><a href="#Exploratory-Data-Analysis" data-toc-modified-id="Exploratory-Data-Analysis-6.2"><span class="toc-item-num">6.2&nbsp;&nbsp;</span>Exploratory Data Analysis</a></span><ul class="toc-item"><li><span><a href="#Geography" data-toc-modified-id="Geography-6.2.1"><span class="toc-item-num">6.2.1&nbsp;&nbsp;</span>Geography</a></span></li><li><span><a href="#PM2.5-Pollution" data-toc-modified-id="PM2.5-Pollution-6.2.2"><span class="toc-item-num">6.2.2&nbsp;&nbsp;</span>PM2.5 Pollution</a></span></li><li><span><a href="#Agricultural-Burning" data-toc-modified-id="Agricultural-Burning-6.2.3"><span class="toc-item-num">6.2.3&nbsp;&nbsp;</span>Agricultural Burning</a></span></li></ul></li><li><span><a href="#Model-Optimization" data-toc-modified-id="Model-Optimization-6.3"><span class="toc-item-num">6.3&nbsp;&nbsp;</span>Model Optimization</a></span><ul class="toc-item"><li><span><a href="#Fire-Features" data-toc-modified-id="Fire-Features-6.3.1"><span class="toc-item-num">6.3.1&nbsp;&nbsp;</span>Fire Features</a></span></li><li><span><a href="#Training" data-toc-modified-id="Training-6.3.2"><span class="toc-item-num">6.3.2&nbsp;&nbsp;</span>Training</a></span></li><li><span><a href="#Model-Performance" data-toc-modified-id="Model-Performance-6.3.3"><span class="toc-item-num">6.3.3&nbsp;&nbsp;</span>Model Performance</a></span></li></ul></li><li><span><a href="#Model-Feature-of-Importances" data-toc-modified-id="Model-Feature-of-Importances-6.4"><span class="toc-item-num">6.4&nbsp;&nbsp;</span>Model Feature of Importances</a></span></li><li><span><a href="#Simulation" data-toc-modified-id="Simulation-6.5"><span class="toc-item-num">6.5&nbsp;&nbsp;</span>Simulation</a></span></li><li><span><a href="#Effect-of-Reduced-Burning-Activities" data-toc-modified-id="Effect-of-Reduced-Burning-Activities-6.6"><span class="toc-item-num">6.6&nbsp;&nbsp;</span>Effect of Reduced Burning Activities</a></span></li></ul></li><li><span><a href="#Conclusions" data-toc-modified-id="Conclusions-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Conclusions</a></span></li></ul></div>

# Model Air Pollution Thailand and South East Asian Countries

This project aims to use a machine learning model to predict and identify sources of air pollution southeast Asian cities. Because this is a multi-variables problem with complex interaction among features, and time-lag, a machine learning approach has an advantage over a traditional approach. In this study, a random forest regressor(RF) is used to model a small particle pollution(PM2.5) level. After trying to search for various machine learning models, I found that RF performs the best. Besides, the model can be easily interpreted. The model's feature of importances in a combination of data exploration helps identify the major sources of air pollution. The trained model is used to simulate the pollution level when environmental policies are implemented. Here, I provide a case study for Chiang Mai, but the codes work with other cities such as Bangkok, and Hanoi (Given there is enough data, of course!)

To explain how to use my code to analyze the air pollution, I first explain the [packages](#Requirements-2) I used and the [directory tree](#Directory-Tree-3). Then I will describe how to obtain [data](#Data-Sources-4). [AQI convention](#AQI-Convention-5) section explains the color label used in this study. [An Analysis for Chiang Mai](#Modeling-Air-Pollution-in-Chiang-Mai-Data-:-A-Case-Study-6) is split into [Exploring the data](Exploratory-Data-Analysis-6.3), [model optimization](Model-Optimization-6.4), and [simulation](#Simulation-6.5). I will include result figures and the code I used to generate the figures if applicable. The analysis results for Chiang Mai is summarized in the [conclusions](#Conclusions-8) 


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

# AQI Standards

Countries have different standards, which convert the raw pollution readings into air quality index(AQI) and interpret the harmful levels. The maximum AQIs for each pollutant (PM2.5, PM10, SO2 etc) is reported at a single AQI. The figure below compares the US and Thailand AQI. The standards are comparable except for SO$_2$, where the US has a stricter standard. This study uses US AQI standard for calculating AQI for different pollutants. For example, in the PM2.5 case

- 0 - 50 AQI is in a good/satisfactory range. This corresponds to PM2.5 level between 0 - 12 $\mu g/m^3$ (green). 
- 51 - 100 AQI is a moderate range, corresponding to PM2.5 12.1- 35.4  $\mu g/m^3$ (darkyellow)
- 101 - 150 AQI is unhealthy for sensitive group, corresponding to PM2.5 35.5- 55.4  $\mu g/m^3$ (orange)
- 151 - 200 AQI is an unhealthy range, corresponding to PM2.5 55.5- 150.4  $\mu g/m^3$ (red)
- 200+ AQI is a very unhealthy range, corresponding to PM2.5 > 150.5  $\mu g/m^3$ (purple)



![png](https://github.com/worasom/aqi_thailand2/blob/master/reports/chiang_mai/aqi.png)

#  Modeling Air Pollution in Chiang Mai Data : A Case Study

For command used in this analysis, please refer to [EPA notebook](https://github.com/worasom/aqi_thailand2/blob/master/notebooks/4.0_vis_ChiangMai.ipynb) for data analysis, [model optimization notebook](https://github.com/worasom/aqi_thailand2/blob/master/notebooks/5.0-ML_ChiangMai.ipynb) for training, and [simulation notebook](https://github.com/worasom/aqi_thailand2/blob/master/notebooks/6_prediction.ipynb) for simulation.

## Causal Diagram

There are four possible sources of air pollution: local traffic, power plants, industrial activities, and agricultural burning. For the latter three sources, the pollution is generated from the other provinces. In all of these sources, the local weather (temperature, humidity, and wind speed) decides how long the pollution stays in the air, and thus the pollution level. 

Since each pollution source results in different seasonal pollution patterns and chemical characteristics(chemical fingerprint), inspecting the pollutions data, weather, and several burning hotspots seen from the satellite images and the patterns could help identify or rule out air pollution contribution factors. This is done in the [exploratory data analysis section](#Exploratory-Data-Analysis-6.2).


<img src="https://github.com/worasom/aqi_thailand2/blob/master/reports/chiang_mai/casual_di.PNG" width="500"/>

## Exploratory Data Analysis

The first step before any machine learning model is to understand the data. The file `src.visualization.vis_data.py` contains many useful functions for quick data visualization. Again the code for this section can be founded in the [notebook](https://github.com/worasom/aqi_thailand2/blob/master/notebooks/4.0_vis_ChiangMai.ipynb)  

### Geography

It is important to understand the geography of the city. The picture below shows a map of Chiang Mai. It is a medium-sized city surrounded by high mountains. Locations of the monitoring stations, a nearby industrial complex, and power plants are also shown.

![map of Chiang Mai](https://github.com/worasom/aqi_thailand2/blob/master/reports/chiang_mai/cm_map.png)

### PM2.5 Pollution

Compare the AQI for different pollutions. PM2.5 has the higest AQI, follow by O$_3$ and PM10. NO$_2$, CO and SO$_2$ are in the good AQI range.

![different pollutants](https://github.com/worasom/aqi_thailand2/blob/master/reports/chiang_mai/all_pol_aqi.png)

Since data for PM2.5 does not go back as far. I have to infer PM2.5 using other pollutants. Judging from the Pearson coefficient between pollutants in the correlation map below. PM2.5, and PM10 are highly correlated. Later, I will use trend PM10 to infer the trend of PM2.5. 

![correlation](https://github.com/worasom/aqi_thailand2/blob/master/reports/chiang_mai/poll_corr.png)

Air pollution often exhibits seasonal behavior. Understanding this pattern can help to identify the source of air pollution. In the figure below, Seasonal patterns of PM2.5 level(top), number of hotspots within 1000 km from Chiang Mai(middle), and temperature(bottom). The shaded regions are 95% confident interval from different years. In the PM2.5 plot, the horizontal lines indicate the values corresponded to AQI 50, 100, 150, and 200 accordingly. This means that the AQI in the unhealthy range falls between the red and the purple lines. Note that the number of hotspots has a similar seasonal pattern as the PM2.5’s.

![seasonal pattern](https://github.com/worasom/aqi_thailand2/blob/master/reports/chiang_mai/fire_PM25_t_season.png)

On average, the PM2.5 level significantly increases in the winter season. The average values reach the moderate AQI (horizontal yellow line) between the 1st of December and the 30th of April, and an unhealthy AQI between the 15th of February and the 15th of April. The number of hotspots in a 1000 km radius from Chiang Mai has the same seasonal pattern as the PM2.5 level. Most burning activities occur in March, where the pollution level peaks. Moreover, when overlaying the seasonal pattern plots of the PM2.5 and the number of hotspots, they have almost the same pattern. This highly suggests that the burning activities are the causes of high PM2.5 in Chiang Mai.


![seasonal pattern](https://github.com/worasom/aqi_thailand2/blob/master/reports/chiang_mai/fire_PM25_season_aqi.png)

The figure below compares the yearly average of PM2.5, PM10, number of hotspots, and the temperature. Only the values in the pollution season (between Dec 1 and April 30) are used for calculating the average. The rising temperature due to climate change is visible, while the trend for the pollution level(PM10) has a downward trend. Since PM10 and PM2.5 are highly correlated, we can infer that the trend for PM2.5 is the same. Since the yearly trends of pollutions and temperature are in the opposite direction, I can rule out the direct and indirect effects of global warming toward high air pollution.

Again, the yearly trends of both kinds of particle pollutions and the number of hotspots are highly correlated. Moreover, in the years with a particularly low number of hotspots, the both PM2.5 and PM10 are also low, for example, in the pollution season year 2010 (between December 2010 – April 2011), and year 2016 (between December 2016 – April 2017). 

![linear trend](https://github.com/worasom/aqi_thailand2/blob/master/reports/chiang_mai/compare_ln_trends.png)

### Agricultural Burning 

Preliminary analysis(in [EPA notebook](https://github.com/worasom/aqi_thailand2/blob/master/notebooks/4.0_vis_ChiangMai.ipynb)) suggests that the hotspots are from agricultural burning. The figures below show the seasonal pattern of hotspots within 1000 km from Chiang Mai separated by country. Myanmar and Laos are the countries with the most burning activities. 

![hotspots_country](https://github.com/worasom/aqi_thailand2/blob/master/reports/chiang_mai/hotspots_country.png)

Notice that the burning season in Cambodia peaks in January- February, which is earlier than in the other countries.

When divided by the area, Cambodia has the highest concentration hotspots, followed by Laos and Myanmar. However, these hotspots are much further away than those in Myanmar and Laos.

![hotspots_country_per_area](https://github.com/worasom/aqi_thailand2/blob/master/reports/chiang_mai/hotspots_per_km2.png)

## Model Optimization

The commands used for optimization can be found in [model optimization notebook](https://github.com/worasom/aqi_thailand2/blob/master/notebooks/5.0-ML_ChiangMai.ipynb)

### Fire Features

Since the concentration of hotspots differs at various distances from Chiang Mai, it is important to divide the burning activity into distance-based zones. The figure below shows four fire zone. The first zone is within 100 km from Chiang Mai within the Thailand border. The second and third zone is 100 - 200 km and 200 - 400 km from Chiang Mai. A lot of fire activities in this zone are in Myanmar and Laos(red spots). The third zone is between 400 and 700 km from Chiang Mai. Most burning activities here are in Laos. The last zone is between 700 and 1000 km. This zone also not only includes the burning activities in Myanmar, Laos, and Vietnam but also those in Cambodia. However, most fire activities in Cambodia, which is much further away and the burning activities concentrate in December(blue spots), which is not the month with the highest pollution in Chiang Mai. 


![fire_zone](https://github.com/worasom/aqi_thailand2/blob/master/reports/chiang_mai/fire_zone.png)

### Training

Model Optimization breakdown into the following steps

1. Find a reasonable parameters RandomForestRegressor model. Build a model input data using default fire parameters. Split the data into train, validation, and test set. Using the training and validation sets.
2. Remove lower importance features from the model input. This is done by try to drop the data and see if the error decrease. 
3. Optimize for the best fire features. Assuming that the pollution from a hotspot travels to the city at a certain average speed and linger in the environment for an unknown duration. This optimization step finds out the average travel speed and average linger duration.  
4. Improve model performance by adding lag columns (of weather and fire). The effects of weather and hotspots can have a time lag effect. This step search for the amount of lagged to add to the model. 
5. Remove lower importance lag columns. The earlier step often adds too many lagged columns. I prune these columns here. 
6. Optimize for RandomForestRegressor parameters again.  
7. Save model and model meta information. The Model meta would contain fire feature parameters, the lagged values, and columns to use.


### Model Performance

Look at the model performance by plotting actual data and the prediction data. In the [simulation](#Simulation-6.5) section, I will be using the daily average of the prediction, therefore I will also look at the prediction error of the daily average data.



![model performance](https://github.com/worasom/aqi_thailand2/blob/master/reports/chiang_mai/PM25_model_perfomance.png)

When loading a model using `src.models.predict_model.load_model`, the function calculate the R2-score for the test set(which is an hourly pollution data). Then, its calculate the R2-score of a daily average of both the training and test data. 

For Chiang Mai, hourly prediction R2-score is 0.70. The daily average test data has R2-score of 0.79. This is pretty good.   

## Model Feature of Importances

The figure below ranks the input’s order of importance. This is from `feature_importances_` attribute of the random forest(another reason to like random forest !). The fire_100_200 columns means the hotspots between 100 – 200 km from Chiang Mai. The importance unit measures the decrease in model accuracy if a column data is replaced with noise; therefore, the columns from the sources or areas most contributed to the pollution level will have a high order of importance. 

The fire columns from different distances from Chiang Mai are the most important. All fire columns are in the top 6 of the order of importance, in agreement with the data analysis in section 3. The fire activities in 100 – 200 km and 200 - 400 km zones are the two most important columns because of the large number of burning activities and short distances from Chiang Mai. The weather pattern such as temperature and wind speed only has a weak effect on the PM2.5 level. 

![feature_importance](https://github.com/worasom/aqi_thailand2/blob/master/reports/chiang_mai/PM25_rf_fea_op2_nolag.png)

## Simulation

The data between 1st October 2017 and 15th June 2020 are set aside as a test dataset during the model optimization process. In this section, the prediction dataset is used for studying the effect of reduced burning activities, which is the major source of air pollution in Chiang Mai. The fitted model will be used to predict the pollution levels in a scenario where the burning activities decreased. 

The simulation is not the same as the prediction of the test data in the [model performance](#Model-Performance-6.3.3) section, where the actual fire activities and the weather conditions were used to predict the pollution levels. In the simulation and the actual situation, the fire and the weather information is not known, and one has to use the ranges of possible values from previous years to obtain a range of pollution levels in the prediction dataset. I call this method of statistical simulation. The commands used for simulation can be found in the [simulation notebook](https://github.com/worasom/aqi_thailand2/blob/master/notebooks/6_prediction.ipynb)

The figure below compares the set-aside test data(blue) and the simulation(red). The range of possible hotspots and weather is used to predict a range of possible hourly PM2.5 levels with the same day of the year. The values from the same date-time are then averaged to produce final prediction values. The statistical prediction is very similar to the values of the actual data.

![test_data_vs_inference](https://github.com/worasom/aqi_thailand2/blob/master/reports/chiang_mai/test_data_vs_inference.png)

Seasonal pattern of the test data and the simulation. The simulation captures the correct seasonal patterns. Since the simulation is calculated from the same history. Only one seasonal pattern is needed. This seasonal pattern will be used to study the effect of reduced burning activities in the next section.

![test_data_vs_inference_season](https://github.com/worasom/aqi_thailand2/blob/master/reports/chiang_mai/test_data_vs_inference_season.png)

## Effect of Reduced Burning Activities

In the figure below, (top) seasonal pattern of the pollution level when the burning activities are reduced to 100%, 90%, 50%, and 10% in 0 – 700 km radius. 100% means no fire reduction. (bottom) corresponding numbers of hotspots per day in different scenarios. The values above the red line are in the unhealthy AQI range.


![effect_fire_0_700km_sea](https://github.com/worasom/aqi_thailand2/blob/master/reports/chiang_mai/effect_fire_0_700km_sea.png)

The figure below summarizes the average pollution level in December-April upon reducing the burning activities in different areas at different percent. The values above the red line are in the unhealthy AQI range.  Reducing the burning activities in the 0 - 100km area should be the most effective in reducing the pollution level because of the close distance from the city. My simulation shows that if the burning activities in 0 - 100km(blue curve) are 10% of the original value, the average pollution level between December - April would go below the unhealthy limit < 55 $\mu g/m^3$. However, such measurement would not drastically decrease the peak pollution level in March.

![effect_of_fire_reduction_3m](https://github.com/worasom/aqi_thailand2/blob/master/reports/chiang_mai/effect_of_fire_reduction_3m.png)

The figure below summarizes the average pollution level in December-April upon reducing the burning activities in different areas at different percent. The values above the red line are in the unhealthy AQI range. My simulation shows that even with a drastic reduction to 10% of the original value in the 0 - 100 km zone could only reduce the peak pollution level down to around 70 $\mu g/m^3$(blue line), still unhealthy AQI. The reason for the minuscule pollution reduction when reducing only the activities in the 100 km zone is because the burning activities in the outer zones occur in much larger as shown in Figure 10. During the peak of the pollution season, there are about 35 hotspots per day, which is a much smaller amount compared to 900 in the other zones. This means tackling the burning activity within a 100 km zone is not sufficient and reducing the burning has to be done in a wider area.

![effect_of_fire_reduction_mar](https://github.com/worasom/aqi_thailand2/blob/master/reports/chiang_mai/effect_of_fire_reduction_mar.png)

Since reducing the burning activities in the area within 100 km from Chiang Mai alone is not enough to drastically reduce the air pollution level into the moderate limit, it is important to study the effect the burning activities in larger areas.
If the reduction policy is implemented in a larger radius such as 1000 km (purple line), the average pollution level in December-April would drop to the moderate range, and the level in march would drop below the unhealthy range.

It might be counter-intuitive why 10% of the burning activities still cannot result in good air quality range. This is not due to the bias in the model, but because 10% is still a lot of burning activities in the area. For example, during the peak of the pollution season in March, there are a large amount of the burning activities in the 200-400, 400-700, 700-1000 km zones. With 10% of the original burning activities, there are still about 80 hotspots/day on average. This explains why the pollution level in March remains above the moderate AQI range even though 90% of burning activities have been cut.

Putting together, this animation summarized the effect of fire reduction as the radius of fire reduction policy increases.


Putting togehter, this animation summarized the effect of fire reduction as the radius of fire reduction policy increases.

![animation](https://github.com/worasom/aqi_thailand2/blob/master/reports/ani_plot/eff_reduced_fire1.gif)

# Conclusions

I study the sources of the air pollution problem in Chiang Mai. The major problem is from the high AQI of PM2.5 pollution. The PM2.5 level has a seasonal pattern with value exceed the moderate AQI between December-April and often exceeds unhealthy AQI in March. Other pollutants such as PM10 and O¬3, also exhibit similar seasonal patterns but with peak AQIs still in the moderate range. By inspecting the seasonal pattern of the pollution level, I rule out traffic, industrial activities, and power plants from the possible sources of air pollution and identify that agricultural burning as the major source. The number of burning hotspots seen from the MODIS satellite has the same seasonal pattern as that of the PM2.5 level. Moreover, in the years with abnormally low burning activities correspond to those with lower pollution levels. The number of burning activities and thus the pollution level is not a result of climate change because of their decreasing yearly trends, which is opposite to the increasing temperature trend. 

To quantify the effect of the burning activities toward the PM2.5 level, I trained a random forest regressor to predict the hourly PM2.5 level with the weather, hotspots within 1000 km from Chiang Mai, and auxiliary date-time information as a tabular input. The hotspots data are divided into distance-based zones to measure the effect of each zone. The model achieved a 0.70 R2-score for hourly values in the 3 recent years of unseen dataset, and 0.78 R2-score when considering the daily average values. As expected, the model order of importance ranks the hotspots columns as the tops with the first in 100-200 km zone as most important. The trained model is used to simulate a range of possible pollution levels using the historical input values in the training dataset. The average pollution level has correct seasonal patterns and values as the data in the unseen dataset. The simulated pollution levels under different scenarios with reduced burning activities. The simulation shows that reducing the burning activities within the Thailand border (100 km area from Chiang Mai) alone could not lower to PM2.5 level from the unhealthy AQI, and aggressive reduction in the area covered as far a 1000 km is needed. 



```python

```
