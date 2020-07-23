# -*- coding: utf-8 -*-
from ..imports import *
from ..gen_functions import *
from ..features.dataset import Dataset
from .train_model import *

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

def get_data_samples(dataset, n_samples=100,time_range=[]):
    """Sample the possible test data from train data. The dataset must alredy has the lag columns built

    Args:
        dataset: load data using load_model1 function 
        split_list(optional): ratio to split the data between the test and train
        n_samples: number of sample per hours 
    
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

    # sample the data 
    data_samples = []
    for test_datetime in tqdm_notebook(time_range):
    
        samples = get_sample(test_datetime, wea, fire,year_list, year_samples=year_sam,day_err=10,hour_err=2)
        data_samples.append(samples)
         
    
    data_samples = pd.concat(data_samples, ignore_index=True)
    # add calenda information by merging with test_data
    data_samples = data_samples.set_index('datetime')
    data_samples = data_samples.merge(test_data[date_cols], right_index=True, left_index=True, how='left')

    return data_samples.dropna()

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

def make_senario(model, data_samples, features, per_cut, x_cols):
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
    x = data_senario[x_cols].values
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



def reduc_effect(model, data_samples, x_cols, features, sea_error, q, red_list= [0.90, 0.75, 0.5, 0.25, 0.10, 0] ):
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
    sea_pred_all = []
   
    for per_cut in  red_list:    
        ypred_df = make_senario(model, data_samples, features, per_cut= per_cut, x_cols=x_cols)
        band_df = make_band(ypred_df, q_list=[q])
        sea_pred = cal_season_band(band_df, sea_error)
        sea_pred.columns = [(1-per_cut)]
        sea_pred_all.append(sea_pred)

    return  pd.concat(sea_pred_all,axis=1)



