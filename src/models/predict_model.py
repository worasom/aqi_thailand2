# -*- coding: utf-8 -*-
from ..imports import *
from ..gen_functions import *
from ..features.dataset import Dataset
from ..features.build_features import *
from .train_model import *
from ..visualization.vis_data import *
from ..visualization.vis_model import *


def cal_error(dataset, model, data_index):
    """Calculate model performance over training and test data.
     
    - take the daily average and plot actual vs prediction for training and test set
    return xtrn, ytrn, xtest, ytext in dataframe format

    Args:
        dataset: dataset object
        model: fitted model 
        split_list(optional): train/test spliting ratio[default:[0.7,0.3]]
        xlim(optional): if passed, use these value for xlim
        filename(optional): if not None, save the figure using the filename 

    Returns:
        trn_pred_df: actual, prediction of the training set (no averaging)
        test_pred_df: actual, prediction of the test set (no averaging)
    
    """ 

    #dataset.split_data(split_ratio=split_list)
    xtrn, ytrn, _ = dataset.get_data_matrix(use_index=data_index, x_cols=dataset.x_cols)
    #xtest, ytest, _ = dataset.get_data_matrix(use_index=dataset.split_list[1], x_cols=dataset.x_cols)

    ytrn_pred = model.predict(xtrn)
    #ytest_pred = model.predict(xtest)

     # turn into df 
    ytrn_pred_df = pd.DataFrame(ytrn, index=data_index, columns=['actual'])
    ytrn_pred_df['pred'] = ytrn_pred
    ytrn_pred_df['error'] = ytrn_pred_df['actual'] - ytrn_pred_df['pred'] 
    ytrn_pred_df['rmse'] = np.sqrt(ytrn_pred_df['error']**2)
     
    #ytest_pred_df = pd.DataFrame(ytest, index=dataset.split_list[1], columns=['actual'])
    #ytest_pred_df['pred'] = ytest_pred
    #ytest_pred_df['error'] = ytest_pred_df['actual'] - ytest_pred_df['pred'] 
    #ytest_pred_df['rmse'] = np.sqrt(ytest_pred_df['error']**2)

    return ytrn_pred_df.dropna() 

def cal_season_error(error_df, roll_win=15, agg='mean'):
    """ Calculate seasonal error 
    
    Args:
        error_df: hourly training error 
        roll_win: rolling window 
        agg: aggegration statistic

    Returns: pd.DataFrame
        seasonal pattern of the error 

    """
    sea_error, _ = season_avg(error_df, cols=['error','rmse'], roll=False, agg='mean', offset=182)
    sea_error = sea_error.groupby('winter_day').mean()
    sea_error = sea_error.drop(['dayofyear','year'],axis=1)
    return sea_error.rolling(roll_win, min_periods=0, center=True).agg(agg)

    
def get_year_sample(year_list, n_samples=100):
    """Calculate the number of sample from each year. 
    
    Args:
        year_list: a list of year to sample the data. Ex from trn_data.index.year.unique()
        n_samples(optional): number of total samples [default:100]

    Return: list 

    """

    year_sam = np.arange(len(year_list)) + 1
    year_sam = year_sam.cumsum()/year_sam.cumsum().sum()
    year_sam = (year_sam*n_samples).astype(int)

    return year_sam

def get_sample(test_datetime, wea, fire, year_list, year_samples, day_err=7, hour_err=2):
    """Randomsamples weather and fire data from previous years around the same dayofyear
    
    Args:
        test_datetime: datetime for sampling from the train data
        wea: weather dataframe from training data
        fire: fire data df from training data 
        year_list: a list of years in training data
        year_samples: a list of sample from each year 
        day_error: plus/minus date range to sample from 
        hour_err: plus/minus hour to sample from 
        
    Returns: 
        data_samples: sample dataframe 
    
    """
    
    #get date range
    start_date = test_datetime - pd.Timedelta(f'{day_err} days')
    end_date = test_datetime + pd.Timedelta(f'{day_err} days')
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    date_range = date_range.dayofyear.to_list()
     
    # get hour range 
    start_date = test_datetime - pd.Timedelta(f'{hour_err} hours')
    end_date = test_datetime + pd.Timedelta(f'{hour_err} hours')
    hour_range = pd.date_range(start=start_date, end=end_date, freq='h')
    hour_range = hour_range.hour
    
    # get wether sample 
    # select the data
    fire_samples = []
    wea_samples = []
    for year, year_sample in zip(year_list, year_samples):
        # select deata for the year
        wea_sam = wea[(wea['year']==year) & wea['day_of_year'].isin(date_range) & wea['hour'].isin(hour_range)]
         
        if len(wea_sam) > 0:
             wea_samples.append(wea_sam.sample(year_sample,replace=True))
            
        # select deata for the year
        fire_sam = fire[(fire['year']==year) & fire['day_of_year'].isin(date_range) & fire['hour'].isin(hour_range)]
     
        if len(fire_sam) > 0:
             fire_samples.append(fire_sam.sample(year_sample,replace=True))

    wea_samples = pd.concat(wea_samples, ignore_index=True).sample(frac=1).reset_index(drop=True)
    fire_samples = pd.concat(fire_samples, ignore_index=True).sample(frac=1).reset_index(drop=True)
    wea_samples['datetime'] = test_datetime
    
    data_samples = pd.concat([wea_samples, fire_samples], axis=1)
    
    return data_samples.drop(['day_of_year','hour','year'],axis=1)

def add_lag(df, lag_dict):
    """Build the lag data using number in lag_range. 
    Add the new data as self.data attribute. 

    Args:
        df: dataframe to build lag 
        lag_dict: a dictionary containing laging information such n_max, 'step', 'roll'

    Returns: dataframe containing original data and lags

    """

    lag_range = np.arange(1, lag_dict['n_max'], lag_dict['step'])
    roll = lag_dict['roll']
        
    lag_list = [df]
    for n in lag_range:
        lag_df = df.copy()
        lag_df.columns = [ s+ f'_lag_{n}' for s in lag_df.columns] 
        if roll:
            # calculate the rolling average
            lag_df = lag_df.rolling(n,min_periods=None).mean()
            lag_df = lag_df.shift(1)
        else:
            lag_df = lag_df.shift(n)

        lag_list.append(lag_df)
        
    new_data = pd.concat(lag_list, axis=1, ignore_index=False)
    return new_data.dropna()

def get_data_samples(dataset, time_range=[], n_samples=100, step=1,day_err=10,hour_err=2):
    """Sample the possible test data from train data. The dataset must alredy has the lag columns built

    Args:
        dataset: load data using load_model1 function 
        time_range: time range for inference 
        n_samples: number of sample per hours 
        step: if not 1, skip some data to make the draw faster 
        day_err
        hour_err
    
    Return pd.DataFame
        sample of weather and fire conditon for each hour. Each hour will have n_samples of data

    """

    # look for the name of weather cols 
    data_cols = dataset.data.columns
    wea_cols = ['Temperature(C)', 'Humidity(%)', 'Wind Speed(kmph)', 'wind_CALM', 'wind_E', 'wind_N', 'wind_S', 'wind_W', 'is_rain']
    wea_cols = np.hstack([data_cols[data_cols.str.contains(s[:4])].to_list() for s in wea_cols])
    wea_cols = np.unique(wea_cols)
    
    # look for fire_cols 
    fire_cols = data_cols[data_cols.str.contains('fire')]
    
    date_cols = ['is_holiday', 'is_weekend', 'day_of_week', 'time_of_day']
    date_cols = np.hstack([data_cols[data_cols.str.contains(s[:6])].to_list() for s in date_cols])
    date_cols = np.unique(date_cols)

    # create train and test dfs
    trn_index = dataset.split_list[0]
    trn_data = dataset.data.loc[trn_index]
    test_index = dataset.split_list[1]
    test_data = dataset.data.loc[test_index]

    # number of sample per year
    year_list = trn_index.year.unique()
    year_sam = get_year_sample(year_list=year_list, n_samples=100)

    # extract fire & weather dfs from the train data
    wea = dataset.data[wea_cols].loc[trn_index]
    wea['year'] = wea.index.year
    wea['day_of_year'] = wea.index.dayofyear
    wea['hour'] = wea.index.hour

    fire = dataset.data[fire_cols].loc[trn_index]
    fire['year'] = fire.index.year
    fire['day_of_year'] = fire.index.dayofyear
    fire['hour'] = fire.index.hour

    if len(time_range)==0:
        # time range from the test data 
        time_range = pd.date_range(start=test_data.index.min(), end=test_data.index.max(), freq='h')

    ## sample the data 
    #data_samples = []
    #for test_datetime in tqdm_notebook(time_range[::step]):
    
    #    samples = get_sample(test_datetime, wea, fire,year_list, year_samples=year_sam,day_err=10,hour_err=2)
    #    data_samples.append(samples)
         
     # use joblib to speed up sampling process
    data_samples = Parallel(n_jobs=2)(delayed(get_sample)(test_datetime, wea, fire,year_list, year_sam,day_err,hour_err) for test_datetime in time_range[::step])
    #data_samples = pd.concat(fire, ignore_index=True)


    data_samples = pd.concat(data_samples, ignore_index=True)
    # create date_data
    date_data = pd.DataFrame(index=time_range)
    date_data = add_calendar_info(date_data, holiday_file=dataset.data_folder + 'holiday.csv')
    date_data = add_lag(date_data, dataset.lag_dict)

    # add calenda information by merging with data_data
    data_samples = data_samples.set_index('datetime')
    data_samples = data_samples.merge(date_data[date_cols], right_index=True, left_index=True, how='left')

    return data_samples.dropna()[dataset.x_cols]

def make_band(ypred_df, q_list=[0.01, 0.25, 0.5, 0.75,  0.99]):
    """Convert aggregate prediction for the same timestamp into upper and lower band. 

    Args:
        ypred_df: prediction dataframe
        q_list: a list of quantile to calculate
    
    Returns: dataframe

    """
    band_df = []
    for q in q_list:
    
        band = ypred_df.groupby(level=0).quantile(q=q)
        band_index = band.index
        #band = smooth(band.values,window_len=41)
    
        band_df.append(pd.DataFrame(band, index=band_index, columns=['q'+ str(q)]))
    
    return pd.concat(band_df, axis=1)

def make_senario(model, data_samples, features, per_cut):
    """Make prediction of the data sample with some feature value reduced. 

    Args:
        model: model for prediction
        data_samples: test data sample for different data
        features: columns to cut down
        per_cut: percent reduction must be between 0 - 1
        x_cols: x_columns to use for the model

    Returns: pd.DataFrame
        predicted value for calculate band 

    """
    cols_to_cut = []
    

    for feature in features:
        cols_to_cut = cols_to_cut + data_samples.columns[data_samples.columns.str.contains(feature)].to_list()
    
   
    data_senario = data_samples.copy()
    data_senario[cols_to_cut] = data_samples[cols_to_cut]*(1-per_cut)
    x = data_senario.values
    y = model.predict(x)

    return pd.Series(y, index = data_samples.index) 


def cal_season_band(band_df, sea_error):
    """Convert daily prediction to seasonal prediction 
    
    Args:
        band_df: daily prediction value after rolling average
        sea_error: seasonal prediction error 
    
    Returns: pd.DataFrame
        seasonal prediction with error corrected 

    """
    sea_pred, _ = season_avg(band_df, cols=[], roll=True, agg='mean', offset=182)
    sea_pred = sea_pred.groupby('winter_day').mean()
    sea_pred = sea_pred.drop(['dayofyear','year'],axis=1)


    # Correct bias 
    for col in sea_pred.columns:
        sea_pred[col] += sea_error['error'] 

    return sea_pred

def _reduct_effect_q(model, data_samples, features, sea_error, q, per_cut):
    """Calculate the reduction effect of a single q value.

    Convert to seasonal pattern 

    Args:
        model: model for prediction
        data_samples: weather and fire data 
        features: a list of features to reduce
        sea_error: correction factor by dayofyear
        q: quantile value to sample from 
    
    Returns: seasonal pattern dataframe 
        
    """
    ypred_df = make_senario(model, data_samples, features, per_cut= per_cut)
    band_df = make_band(ypred_df, q_list=[q])
    sea_pred = cal_season_band(band_df, sea_error)
    sea_pred.columns = [int(round(1-per_cut,2)*100)]

    return sea_pred 

def reduc_effect(model, data_samples, features, sea_error, q, red_list= [0.90, 0.75, 0.5, 0.25, 0.10, 0] ):
    """Calculate effect of reduction for feature. 

    Args:
        model: model for prediction
        data_samples: weather and fire data 
        features: a list of features to reduce
        sea_error: correction factor by dayofyear
        q: quantile value to sample from 
        red_list: list of reduction fraction 

    Return:
        sea_pred_all 

    """
    # sea_pred_all = []
   
    # for per_cut in  red_list:    
    #     ypred_df = make_senario(model, data_samples, features, per_cut= per_cut)
    #     band_df = make_band(ypred_df, q_list=[q])
    #     sea_pred = cal_season_band(band_df, sea_error)
    #     sea_pred.columns = [round(1-per_cut,2)]
    #     sea_pred_all.append(sea_pred)

    sea_pred_all = Parallel(n_jobs=2)(delayed(_reduct_effect_q)(model, data_samples, features, sea_error, q, per_cut) for per_cut in red_list)


    return  pd.concat(sea_pred_all,axis=1)


class Inferer():
    """Inferer object is in charge of predicting and infering from previous training data.

    Required a trained model. 

    Args:
        city_name: lower case of city name
        pollutant:str='PM2.5'
        split_list(optional)

    Attributes:
        dataset 

    Raises:
        AssertionError: if the city_name is not in city_names list

    """

    def __init__(self, city_name: str, pollutant:str='PM2.5',  split_list=[0.7, 0.3]):

        """Initialize 
        
        #. Check if the city name exist in the database
        #. Setup main, data, and model folders. Add as attribute
        #. Check if the folders exists, if not create the folders 
        #. Load city information and add as atribute 

        """

        city_names = ['Chiang Mai', 'Bangkok', 'Hanoi', 'Jakarta']

        if city_name not in city_names:
            raise AssertionError(
                'city name not in the city_names list. No data for this city')

        else:
            # load model and add as attribute
            self.dataset, self.model, fire_cols, self.zone_list, self.feat_imp, self.rolling_win = load_model1(city=city_name, pollutant=pollutant, split_list=split_list)
            self.cal_error()
            self.report_folder = self.dataset.report_folder

            levels = self.dataset.transition_dict[pollutant][1:4]
            colors = ['orange','red', 'purple']
            self.color_zip = [*zip(levels, colors)]
            

    def cal_error(self):
        """Calculate the training and seasonal error. Add trn_error and seasonal_error as attribute

        """

        # calculate error of the train dataset and turn it into the seasonal error 
        self.trn_error = cal_error(self.dataset, self.model, data_index=self.dataset.split_list[0])
        self.sea_error = cal_season_error(self.trn_error, roll_win=14, agg='mean')
        print('max error', np.max(self.sea_error.values))

    def _get_data_sample(self, n_samples=100, step=1,day_err=10,hour_err=2):
        """Sample the possible test data from train data. Add as data_samples attribute

        Args:

            Args:
                n_samples: number of sample per hours 
                step: if not 1, skip some data to make the draw faster 
                day_err
                hour_err

        """
        print('obtaining inference samples. This will take about 20 mins') 
        self.data_samples = get_data_samples(dataset=self.dataset, n_samples=n_samples,step=step,day_err=day_err,hour_err=day_err)

    def compare_inf_act(self, q_list=[ 0.5, 0.75,  0.95]):
        """Compare inference and actual data. Save the results plot. 

        Args:
            q_list(optional): a list of inference quantile[defaul=[0.05, 0.25, 0.5, 0.75,  0.95]]

        """

        # get data sample
        plot_sea_error(self.trn_error, self.sea_error, filename=self.report_folder+'season_error.png')

        # compare inference with actual data 
        # predict the data
        ypred_df = make_senario(self.model, self.data_samples, ['fire_0_100'], per_cut= 0)
        band_df = make_band(ypred_df, q_list=q_list)
        # smooth the data
        band_df = band_df.rolling(self.rolling_win, min_periods=0).mean()
        #band_df = band_df.resample('d').mean()

        ytest_pred_df = cal_error(self.dataset, self.model, data_index=self.dataset.split_list[1])
        plt_infer_actual(ytest_pred_df.resample('d').mean().dropna(), band_df, filename=self.report_folder+'test_data_vs_inference.png')
     
        # plot seasonal predicton vs real data 
        sea_pred = cal_season_band(band_df, self.sea_error)

        # compare seasonal behavior with inference 
        plot_infer_season(self.dataset.poll_df.loc['2015-01-01':], self.dataset.pollutant, sea_pred, self.color_zip, filename=self.report_folder+'test_data_vs_inference_season.png' )

    
    def features_effect_season(self, features:list, q, red_list=[0, 0.1, 0.25, 0.5, 0.75, 0.9], save=False):
        """Show an effect of reducing feature or features on the seasonal patterns 

        Args: 
            features: a list of feature to observe
            q: quantile for picking the inference distribution [default:0.75]
            red_list: a list of reducting 

        """

        fea_effect = reduc_effect(self.model, self.data_samples, features, self.sea_error, q=q, red_list= red_list)

        _, ax = plt.subplots(1, 1, figsize=(10, 4))
        ax.plot(fea_effect) 
        

        title_str = '&'.join(features)
        ax.set_title(f'Effect of Reducing (% reduction) \n'+fill(title_str,50))

        for l, c in self.color_zip:
            ax.axhline(l, color=c)

        new_ticks = ['07-01', '08-20', '10-09', '11-28', '01-16', '03-06', '04-25', '06-14', '']         
        
        ax.set_xticklabels(new_ticks)
        ax.set_xlim([fea_effect.index.min(), fea_effect.index.max()])
        ax.legend(fea_effect.columns.to_list())
        ax.set_xlabel('month-date')
        ax.set_ylim([0,110])


        
        if save:
            plt.savefig(self.report_folder + 'effect_of_' +title_str+'.png')

        return fea_effect
    

    def features_effect_sum(self, features_list, q, red_list=[0, 0.1, 0.25, 0.5, 0.75, 0.9], time_range=[0,-1],agg='mean' ):
        """Summarize effect of reduction in red_list of the features in the feature list. 

        Calculate the summary value between time_range and use aggegration method specified by agg

        Return: pd.DataFrame.
            The average prediction pollution level if reduction of each feature occure
            
        """

        fea_effect_df = []
        columns_list = []
        for feature in tqdm_notebook(features_list):
            sea_pred_all = reduc_effect(self.model, self.data_samples, feature, self.sea_error, q=q, red_list= red_list )
            sea_pred_all_mean = sea_pred_all.loc[time_range[0]:time_range[1]].agg(agg)
            columns_list.append(' & '.join(feature))
            fea_effect_df.append(sea_pred_all_mean)
    
    
        fea_effect_df = pd.concat(fea_effect_df, axis=1)
        fea_effect_df.columns = columns_list

        return fea_effect_df


