# -*- coding: utf-8 -*-
 
import os
import sys
import logging
 


if __package__: 
    from ..imports import *
    from ..gen_functions import *
    from ..features.prov_dataset import ProvDataset
    from ..visualization.vis_model import *
    from .predict_model import *
    from .train_model import *

else:
    # import anything in the upper directory 
    _i = os.path.dirname(os.path.dirname(os.path.abspath("..")))
    if _i not in sys.path:
        sys.path.insert(0, _i)
    from imports import *
    from gen_functions import *
    from features.prov_dataset import ProvDataset
     
    from visualization.vis_model import *
    # import files in the same directory
    _i = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _i not in sys.path:
        sys.path.insert(0, _i)

    from predict_model import *
    from train_model import *

def get_xtrn(city_name, main_data_folder, pollutant='PM2.5', size=3000, x_cols=[], build=True):
    """Initialize prov_dataset, split the data into train and test. Randomly select the data and return xtrn, and ytrn

    Args:
        city_name
        pollutant
        size: 
        x_cols: 
        build: if True, build dataset for each province

    Returns:
        xtrn[type]: 
        ytrn: 
        x_cols

    """
    
    if city_name in ['Songkhla', 'Saraburi']:
        
        split_ratio1 = [0.8, 0.2]
    else:
         
        split_ratio1 = [0.7, 0.3]
    
    # build data 
    provdataset = ProvDataset(city_name, main_data_folder=main_data_folder)
    print(provdataset.city_name)
    if build:
        provdataset.build_all_data()
        provdataset.save_()
    else:
        provdataset.load_()
        
    provdataset.build_feature()
    provdataset.save_()
    
    print(provdataset.data.index.min(), provdataset.data.index.max())
    print(provdataset.data.dropna().shape)
    
    # training split the data and build data matrix 
    provdataset.split_data(split_ratio=split_ratio1)
    use_index = np.random.choice(provdataset.split_list[0], size=size)
    provdataset.monitor =  pollutant
    xtrn, ytrn, x_cols, weights = provdataset.get_data_matrix(use_index=use_index, x_cols=x_cols)
    
    return xtrn, ytrn, x_cols

def get_xtest(city_name, main_data_folder, x_cols, pollutant='PM2.5', build=False):
    """Initialize prov_dataset, split the data into train and test. 
    return xtest and ytest for evaluation

    Args:
        city_name
        pollutant
        size: 
        x_cols: 
        build: if True, build dataset for each province

    Returns:
        xtrn[type]: 
        ytrn: 
         
    """    
    
    if city_name in ['Songkhla', 'Saraburi']:
        
        split_ratio1 = [0.8, 0.2]
    else:
         
        split_ratio1 = [0.7, 0.3]
    
    # build data 
    provdataset = ProvDataset(city_name, main_data_folder=main_data_folder)
    print(provdataset.city_name)
    if build:
        provdataset.build_all_data()
        provdataset.save_()
    else:
        provdataset.load_()
    provdataset.build_feature()
    
    provdataset.split_data(split_ratio=split_ratio1)
    
    # training split the data and build data matrix 
    provdataset.split_data(split_ratio=split_ratio1)
    use_index = np.random.choice(provdataset.split_list[1], size=int(len(provdataset.split_list[1])*booth_frac))
    provdataset.monitor =  pollutant
    xtest, ytest, x_cols, weights = provdataset.get_data_matrix(use_index=use_index, x_cols=x_cols)
    
    return xtest, ytest

def get_avg_test(model, city_name, main_data_folder, x_cols, booth_frac = 0.7, pollutant='PM2.5', build=False):
    
    if city_name in ['Songkhla', 'Saraburi']:
        
        split_ratio1 = [0.8, 0.2]
    else:
         
        split_ratio1 = [0.7, 0.3]
    
    # build data 
    provdataset = ProvDataset(city_name, main_data_folder=main_data_folder)
    print(provdataset.city_name)
    if build:
        provdataset.build_all_data()
        provdataset.save_()
    else:
        provdataset.load_() 

    
    provdataset.load_()
    provdataset.build_feature()
    data_size = len(provdataset.data)
    
    # training split the data and build data matrix 
    provdataset.split_data(split_ratio=split_ratio1)
    provdataset.monitor =  pollutant
    
    
    score_list = []
    for i in range(10):
        use_index = np.random.choice(provdataset.split_list[1], size=int(len(provdataset.split_list[1])*booth_frac))
        xtest, ytest, x_cols, weights = provdataset.get_data_matrix(use_index=use_index, x_cols=x_cols)

        preds = model.predict(xtest)
        score = cal_scores(ytest, preds)
        score_list.append(cal_scores(ytest, preds))
    
    score_df = pd.DataFrame(score_list)
    mean_score = score_df.mean()
    mean_std = score_df.std()
    mean_std.index = [s+'_std' for s in mean_std.index]

    return pd.DataFrame(pd.concat([mean_score, mean_std])).transpose(), data_size

def train_impute_model(main_data_folder: str = '../data/', model_folder='../models/', report_folder='../reports/', build=True):
    """"Train linear regression model for imputing PM2.5 data 
    
    Args:
        build: if True, build dataset for each province


    Return: dict
        pollutant_meta dictionary for that pollutant
    
    """

    model_folder = model_folder + 'Thailand/'
    poll_name = 'PM25'

    # list of province to train 
    prov_list = ['Rayong',
     'Chiang Mai',
     'Songkhla',
     'Saraburi',
     'Bangkok',
     'Samut Sakhon',
     'Nan',
     'Sa Kaeo',
     'Loei',
     'Chon Buri',
     'Khon Kaen',
     'Ratchaburi']

    
    # collect all xtrn, ytrn
    xtrn, ytrn = [] , []
    x_cols = [
     'PM10',
     'O3',
     'CO',
     'NO2',
     'Humidity(%)',
     'Temperature(C)',
     'Pressure(hPa)',
     'Wind_Gust(kmph)',
     'Precip.(mm)',
     'Wind_Speed(kmph)',
     'Dew_Point(C)']

    for prov in prov_list:
        prv_xtrn, prv_ytrn, x_cols = get_xtrn(city_name=prov, pollutant='PM2.5', size=3000, x_cols=x_cols, build=build, main_data_folder=main_data_folder)
        print(prv_xtrn.shape, prv_ytrn.shape)
        xtrn.append(prv_xtrn)
        ytrn.append(prv_ytrn)
        
    xtrn = np.vstack(xtrn)
    ytrn = np.hstack(ytrn)

    print('xtrn and ytrn shape ', xtrn.shape, ytrn.shape)

    # linear fit 
    model = LinearRegression()
        
    # feature selection 
    selector  = RFECV(
            estimator=model,
            step=1,
            min_features_to_select=min_features_to_select)
    selector.fit(xtrn, ytrn)
    new_x_cols = np.array(x_cols)[selector.get_support()].tolist()
    
    
    print('column to choose ', new_x_cols)

    # retrain 
    xtrn = []
    ytrn = []
    for prov in prov_list:
        prv_xtrn, prv_ytrn, x_cols = get_xtrn(city_name=prov, pollutant='PM2.5', size=3000, x_cols=new_x_cols, build=False, main_data_folder=main_data_folder)
        xtrn.append(prv_xtrn)
        ytrn.append(prv_ytrn)
        
    xtrn = np.vstack(xtrn)
    ytrn = np.hstack(ytrn)
    
    # refit the model 
    model.fit(xtrn, ytrn)

    # save the model and model meta 
    model_filename = model_folder + f'{poll_name}_impute_model.pkl'
    pickle.dump( model, open( model_filename, 'wb'))
    meta_filename =  model_folder + f'{poll_name}_impute_model_meta.json'
    model_meta = {'x_cols': new_x_cols}
    model_meta = save_meta(meta_filename, model_meta)

    
def eval_prov(main_data_folder: str = '../data/', model_folder='../models/', report_folder='../reports/', build=False):
    """Evaluate model performance for all the province in the province list. Save the evaluation result as provincea name .csv 

    """

    prov_list = ['Rayong',
    'Chiang Mai',
    'Songkhla',
    'Saraburi',
    'Bangkok',
    'Samut Sakhon',
    'Nan',
    'Sa Kaeo',
    'Loei',
    'Chon Buri',
    'Khon Kaen',
    'Ratchaburi']

    prov_list = []

    prov_list += [
    #'Phuket',
    #'Pathum Thani',
    #'Samut Prakan',
    # 'Nonthaburi',
    # 'Phra Nakhon Si Ayutthaya',
    # 'Lampang',
    # 'Narathiwat',
    # 'Yala',
    # 'Mae Hong Son',
    # 'Chiang Rai',
    # 'Lamphun',
    # ###'Phrae',
    # 'Nakhon Ratchasima',
    ###'Phayao',
    #########################'Nakhon Sawan',
    #'Surat Thani',
    ###'Chachoengsao',
    ###'Prachin Buri',
    # 'Tak',
    # 'Kanchanaburi',
    # 'Satun',
    # 'Nakhon Pathom',
    ####'Sukhothai',
    ###'Si Sa Ket',
    ###'Amnat Charoen',
    ###'Mukdahan',
    #'Bueng Kan',
    ###'Nong Bua Lam Phu',
    ###'Yasothon',
    #'Nakhon Nayok',
    #'Sakon Nakhon',
    # 'Ubon Ratchathani',
    # 'Nong Khai',
    # 'Samut Songkhram',
    ### 'Chai Nat',
    ###'Phetchaburi',
    ###'Ang Thong',
    #'Nakhon Si Thammarat',
    #'Nakhon Phanom',
    ###'Phitsanulok',
    ###'Suphan Buri',
    'Trat'
    ]

    prov_list = ['Tak', 'Bangkok', 'Samut Songkhram', 'Samut Prakan', 'Chiang Mai', 'Nan', 'Nakhon Pathom', 'Lamphun', 'Samut Sakhon', 'Sa Kaeo', 'Nakhon Ratchasima', 'Nakhon Phanom', 'Chiang Rai', 'Ratchaburi', 'Khon Kaen', 'Lampang', 'Loei', 'Chon Buri', 'Rayong', 'Pathum Thani', 'Phra Nakhon Si Ayutthaya', 'Ubon Ratchathani', 'Nonthaburi', 'Bueng Kan', 'Phuket']

    #prov_list = ['Samut Prakan']

    model_folder = model_folder + 'Thailand/'
    poll_name = 'PM25'
    model_choice = '1lr'

    # load the model and x_cols 
    model_filename =  model_folder + f'{poll_name}_impute_model.pkl'
    model = pickle.load(open(model_filename, 'rb')) 
    meta_filename =  model_folder + f'{poll_name}_impute_model_meta.json'
    model_meta = load_meta(meta_filename)
    x_cols = model_meta['x_cols']

    # performance by province
    for prov in prov_list:
        city_score_df, data_size = get_avg_test(model=model,city_name=prov, main_data_folder=main_data_folder, x_cols=x_cols, booth_frac = 0.7, pollutant='PM2.5', build=build)
        city_score_df['province'] = prov
        city_score_df['data_size'] = data_size
        city_score_df.to_csv(model_folder + f'{prov}_{model_choice}_imput_score.csv', index=False)


def sum_score(model_folder='../models/'):

    model_folder = model_folder + 'Thailand/'
    poll_name = 'PM25'
    model_choice = '1lr'

    # summarize score 
    files = glob(model_folder + f'*_{model_choice}_*.csv')
    files = [s for s in files if '\\imput_score.csv' not in s]
    all_score = []
    for file in files:
        df = pd.read_csv(file)
        df['model_choice'] = model_choice
        all_score.append(df)
    
    all_score = pd.concat(all_score)
    all_score = all_score.reset_index(drop=True)
    all_score = all_score.sort_values('test_r2_score', ascending=False)
    all_score.to_csv(model_folder + f'{model_choice}_imput_score.csv', index=False)

def impute_pm25(model, prov, x_cols, main_data_folder: str = '../data/', model_folder='../models/', report_folder='../reports/', build=True):
    """"Impute and save pm25 

    Args:
        model ([type]): [description]
        prov ([type]): [description]
        x_cols ([type]): [description]
        main_data_folder (str, optional): [description]. Defaults to '../data/'.
        model_folder (str, optional): [description]. Defaults to '../models/'.
        report_folder (str, optional): [description]. Defaults to '../reports/'.
        build (bool, optional): [description]. Defaults to True.
    """

    # build data 
    provdataset = ProvDataset(city_name=prov, main_data_folder=main_data_folder)
    print(provdataset.city_name)
    if build:
        provdataset.build_all_data()
        #provdataset.save_()
    else:
        provdataset.load_() 

    # load and build feature without dropping columns
    provdataset.build_feature(dropna=False)
    provdataset.monitor = 'PM2.5'

    xtest = provdataset.data[ x_cols]
    fill_cols = ["PM10", "O3", "CO", "NO2", "Temperature(C)", 'Dew_Point(C)']
    for col in fill_cols:
        mean_value = xtest[col].mean()
        xtest[col] = xtest[col].fillna(mean_value)
    xtest = xtest.dropna()
    datetime = xtest.index.values
    preds = model.predict(xtest.values)

    pred_df = pd.DataFrame({'datetime': datetime, 
                            provdataset.monitor: preds})
    provdataset.impute_pm25 = pred_df
    provdataset.save_()


def impute_all_prov(main_data_folder: str = '../data/', model_folder='../models_w_impute/', report_folder='../reports_w_impute/'):
    """ 

    Args:
        main_data_folder (str, optional): [description]. Defaults to '../data/'.
        model_folder (str, optional): [description]. Defaults to '../models/'.
        report_folder (str, optional): [description]. Defaults to '../reports/'.

    """

    prov_list = ['Tak', 'Bangkok', 'Samut Songkhram', 'Samut Prakan', 'Chiang Mai', 'Nan', 'Nakhon Pathom', 'Lamphun', 'Samut Sakhon', 'Sa Kaeo', 'Nakhon Ratchasima', 'Nakhon Phanom', 'Chiang Rai', 'Ratchaburi', 'Khon Kaen', 'Lampang', 'Loei', 'Chon Buri', 'Rayong', 'Pathum Thani', 'Phra Nakhon Si Ayutthaya', 'Ubon Ratchathani', 'Nonthaburi', 'Bueng Kan', 'Phuket']

    # prov_list = ['Rayong', 'Khon Kaen',
    # 'Songkhla',
    # 'Samut Sakhon',
    # 'Nan',
    # 'Sa Kaeo',
    # 'Loei',
    # 'Ratchaburi','Chiang Mai', 'Bangkok',
    # 'Chon Buri', 'Saraburi']

    poll_name = 'PM25'
    # load the model and x_cols 
    model_filename =  model_folder + 'Thailand/'+ f'{poll_name}_impute_model.pkl'
    model = pickle.load(open(model_filename, 'rb')) 
    meta_filename =  model_folder + 'Thailand/' + f'{poll_name}_impute_model_meta.json'
    model_meta = load_meta(meta_filename)
    x_cols = model_meta['x_cols']
    bad_prov = []
    for prov in prov_list:

        try:

            impute_pm25(model=model, prov=prov, x_cols=x_cols, main_data_folder=main_data_folder, model_folder=model_folder, report_folder=report_folder, build=True)

        except:
            bad_prov.append(prov)
    
    print('province with no imputation ', bad_prov)

     
