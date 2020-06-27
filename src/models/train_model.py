# -*- coding: utf-8 -*-
from ..imports import *
from ..gen_functions import *
from ..features.dataset import Dataset
from ..visualization.visualize import *

def load_meta(meta_filename:str):
    """Read model_meta dictionary and return model_meta dicitonary

    Args:
        meta_filename: model meta filename 
      

    Return: dict
        pollutant_meta dictionary for that pollutant 

    """
    if os.path.exists(meta_filename):
        with open(meta_filename) as f:
            model_meta = json.load(f)
    else:
        model_meta = {}

    return model_meta

def save_meta(meta_filename:str, model_meta):
    """Save model_meta

    Args:
        meta_filename: model_meta filename 
        
    Returns(optional): model meta dictionary 
        
    """ 

    with open(meta_filename, 'w') as f:
        json.dump(model_meta, f)


def do_rf_search(x_trn:np.array, y_trn:np.array, cv_split:str='time', n_splits:int=5,param_dict:dict=None):
    """Perform randomize parameter search for randomforest regressor return the best estimator 
    Args: 
        x_trn: 2D array of x data 
        y_trn: 2D np.array of y data
        cv_split(optional): if "time". Use TimeSeriesSplit, which don't shuffle the data.
        n_splits(optional): number of cross validation split [default:5]
        params_dict(optional): search parameter dictionary [default:None]

    Returns: best estimator 
    """
    # rf 
    m = RandomForestRegressor(n_jobs=-1, random_state=42)
    
    if param_dict==None:
        param_dict = {'n_estimators':range(20,200,20),
              'max_depth': [3, None],
              'min_samples_split' : [2, 5, 10, 20], 
              'max_features' : range(2,x_trn.shape[1]),
               'bootstrap' : [True, False],
              'min_samples_leaf': range(1, 8)}
    
    if cv_split =='time':
        cv = TimeSeriesSplit(n_splits=n_splits)
        
    else:
        cv = n_splits
    #hyper parameter tuning
    search = RandomizedSearchCV(m, param_distributions=param_dict,
                            n_iter=100,n_jobs=-1, cv=cv, random_state=40)

    search.fit(x_trn,y_trn)
    
    print(search.best_params_, search.best_score_)
    
    return search.best_estimator_

def reduce_cols(dataset, x_cols:list, to_drop:list, model,trn_i, val_i):
    """Try droping the columns in to_drop. Keep track of the columns which can be remove from the model.
    
    Args:
        dataset: dataset object
        x_cols: current x columns 
        to_drop: data to drop 
        model: model object to fit and predict
        trn_i: index in dataset.split_list for training data 
        val_i: index in dataset.split_list for validation data
    
    Returns:
        model: fitted model
        x_cols: a list of new x_cols data 
        
    """
    print('old cols', x_cols)
    trn_index = dataset.split_list[trn_i]
    val_index = dataset.split_list[val_i]
    
    for col in to_drop:
        
        xtrn, ytrn, x_cols = dataset.get_data_matrix(use_index=trn_index,x_cols=x_cols)
        xval, yval, _ = dataset.get_data_matrix(use_index=val_index,x_cols=x_cols)
        
        # obtain the baseline data
        model.fit(xtrn, ytrn)
        base_score = cal_scores(yval, model.predict(xval), header_str ='')['r2_score']
     
        new_cols = x_cols.drop(col).copy()
        xtrn, ytrn, new_x_cols = dataset.get_data_matrix(use_index=trn_index,x_cols=new_cols)
        xval, yval, _ =  dataset.get_data_matrix(use_index=val_index,x_cols=new_cols)
    
        params_dict = model.get_params()
        # function for tree-based model. this step prevent the code from breaking 
        # when removeing some features and the dimension is less than 'max_features'
        
        if ('max_features' in params_dict.keys()) and (params_dict['max_features'] > xtrn.shape[1]):
            params_dict['max_features'] = xtrn.shape[1]
            model.set_params(**params_dict)
            
        model.fit(xtrn,ytrn)
        score = cal_scores(yval, model.predict(xval), header_str ='')['r2_score']
        
        if score> base_score:
            x_cols = x_cols.drop(col)
            print('drop', col)
        
    # obtain the final model 
    
    xtrn, ytrn, x_cols = dataset.get_data_matrix(use_index=trn_index,x_cols=x_cols)
    xval, yval, _ = dataset.get_data_matrix(use_index=val_index,x_cols=x_cols)
    model.fit(xtrn, ytrn)
    score = cal_scores(yval, model.predict(xval), header_str ='')['r2_score']   
    
    print('use columns', x_cols)
    print('r2_score after dropping columns', score)
    return model, x_cols

def sk_op_fire(dataset, model, trn_index, val_index,wind_range:list=[2,20],shift_range:list=[-48,48],roll_range:list=[24, 120],vis:bool=False)-> dict:
    """Search for the best fire parameter using skopt optimization 
    
    Args: 
        dataset: dataset object 
        model: model object
        trn_index: datetime index for training set
        val_index: datetime index of validation set 
        wind_range(optional): min and max value of wind speed 
        shift_range(optional): min and max value of shift parameter
        roll_range(optional): min and max value of roll parameter
        vis(optional): if True, also plot the search space
        
    Return: fire_dict fire dictionary 
    
    """
    
    # check the baseline 
    _ = dataset.merge_fire(dataset.fire_dict)
    x_cols = dataset.x_cols
    print('skop_ fire use x_cols', x_cols)
    # establish the baseline 
    xtrn, ytrn, x_cols = dataset.get_data_matrix(use_index= trn_index,x_cols=x_cols)
    xval, yval, _ = dataset.get_data_matrix(use_index=val_index, x_cols=x_cols)
    
    model.fit(xtrn,ytrn)
    best_score = r2_score(yval,model.predict(xval))
    best_fire_dict = dataset.fire_dict
    print('old score', best_score, 'fire dict', best_fire_dict)
    
    print('optimizing fire parameter using skopt optimizer. This will take about 20 mins')
    # build search space 
    wind_speed = Real(low=wind_range[0], high=wind_range[1], name='wind_speed')
    shift = Integer(low=shift_range[0], high=shift_range[1], name='shift')
    roll = Integer(low=roll_range[0],high=roll_range[1], name='roll')
    
    dimensions = [wind_speed, shift, roll]
    # setup the function for skopt
    @use_named_args(dimensions)
    def fit_with( wind_speed, shift, roll):
        # function to return the score (smaller better)
        fire_dict = { 'w_speed': wind_speed, 
                      'shift': shift,
                      'roll': roll}
        _ = dataset.merge_fire(fire_dict)
    
        xtrn, ytrn, x_cols = dataset.get_data_matrix(use_index= trn_index, x_cols=dataset.x_cols)
        xval, yval, _ = dataset.get_data_matrix(use_index=val_index, x_cols=dataset.x_cols)
    
        model.fit(xtrn,ytrn)
        y_pred = model.predict(xval)
    
        return -r2_score(yval,y_pred)
    gp_result = gp_minimize(func=fit_with,dimensions=dimensions,n_jobs=-1,random_state=30)
    
    wind_speed,shift,roll = gp_result.x
    score = -gp_result.fun
    if score> best_score:
        print('r2 score for the best fire parameters', -gp_result.fun)
        best_fire_dict = {'w_speed': int(wind_speed), 'shift': int(shift), 'roll': int(roll)}
        if vis:
            plot_objective(gp_result)
    else:
        print(f'old fire parameter {best_score} is still better than optimized score ={score}' )
        
    return best_fire_dict


def feat_importance(model, x, y, x_cols, score=r2_score, n_iter=20):
    """Computes the feature importance by shuffle the data
    Args:
        model : the model
        x: the training data
        y: target variable
        x_cols: name of the x columns
        metric: either r2_score of mean_squared_error
    
    Returns: feature of importance pd.DataFrame 
    
    """
 
    baseline = score(y, model.predict(x))
     
    imp = []
    imp_std = []
    for i, col in tqdm(enumerate(x_cols)):
        shuffle = []
        for _ in range(n_iter):
            shuffle_x = x.copy()
            shuffle_x[:,i] = np.random.permutation(shuffle_x[:,i])
            shuffle_score = score(y, model.predict(shuffle_x))
            shuffle.append(shuffle_score)
        imp.append(np.mean(shuffle))
        imp_std.append(np.std(shuffle))
    
    # crate a feature of importance DataFrame
    fea_imp = pd.DataFrame({'index': x_cols, 'importance':imp,'imp_std':imp_std})
    # normalized 
    if score.__name__ == 'r2_score':
        fea_imp['importance'] = (baseline - fea_imp['importance'])/baseline
    elif score.__name__ == 'mean_squared_error':
        fea_imp['importance'] = (-baseline + fea_imp['importance'])/baseline
    
    return fea_imp.sort_values('importance', ascending=False).reset_index(drop=True)


def train_city(city:str='Chiang Mai', pollutant:str='PM2.5',build=False):
    """Training pipeline from process raw data, hyperparameter tune, and save model.

        #. If build True, build the raw data from files 
        #. Process draw data into data using default values
        #. Optimization 1: optimize for the best randomforest model 
        #. Optimization 2: remove unncessary columns 
        #. Optimization 3: find the best fire feature 
        #. Optimization 4: optimize for the best RF again and search for other model in TPOT
        #. Build pollution meta and save

    Args:
        city: city name
        pollutant(optional): pollutant name
        build(optional): if True, also build the data

    Returns: 
        dataset
        rf_model
        tpot_model 

    """

    data = Dataset(city)
    # remove . from pollutant name for saving file
    poll_name = pollutant.replace('.','')
    if build:
        # build data from scratch 
        data.build_all_data(build_fire=True,build_holiday=False)
    
    # load raw data 
    data.load_()
    # build the first dataset 
    data.feature_no_fire()
    # use default fire feature
    fire_cols = data.merge_fire()
    data.pollutant = pollutant
    data.save_()

    #. Optimization 1: optimize for the best randomforest model 
    # split the data into 4 set
    print('=================optimize 1: find the best RF model=================')
    data.split_data(split_ratio=[0.4, 0.2, 0.2, 0.2])
    xtrn, ytrn, x_cols = data.get_data_matrix(use_index=data.split_list[0])
    xval, yval, _ = data.get_data_matrix(use_index=data.split_list[1])
    data.x_cols = x_cols

    model = do_rf_search(xtrn,ytrn)
    score_dict = cal_scores(yval, model.predict(xval), header_str ='val_')
    print('optimize 1 score', score_dict)    

    importances = model.feature_importances_
    feat_imp = pd.DataFrame(importances, index=x_cols, columns=['importance']) 
    feat_imp = feat_imp.sort_values('importance',ascending=False).reset_index()
    show_fea_imp(feat_imp,filename=data.report_folder + f'{poll_name}_fea_imp1.png', title='rf feature of importance(raw)')

    print('=================optimize 2: remove unncessary columns=================')
    # columns to consider droping are columns with low importance
    to_drop = feat_imp['index']
    to_drop = [a for a in to_drop if 'fire' not in a]
    for s in ['Humidity(%)','Temperature(C)','Wind Speed(kmph)']:
        to_drop.remove(s)
    to_drop.reverse()
    model, new_x_cols = reduce_cols(dataset=data,x_cols=x_cols,to_drop=to_drop,model=model,trn_i=0, val_i=1)
    data.x_cols = new_x_cols

    print('================= optimization 3: find the best fire feature ===================')
    # reduce the number of split
    data.split_data(split_ratio=[0.6, 0.2, 0.2])
    data.fire_dict = sk_op_fire(data, model, trn_index=data.split_list[0], val_index=data.split_list[1])

    print('================= optimization 4: optimize for the best RF again and search for other model in TPOT =================')

    data.split_data(split_ratio=[0.7, 0.3])
    trn_index = data.split_list[0]
    test_index = data.split_list[1]
    fire_cols = data.merge_fire(data.fire_dict)
    xtrn, ytrn, x_cols = data.get_data_matrix(use_index=trn_index,x_cols=new_x_cols)
    xtest, ytest, _ = data.get_data_matrix(use_index=test_index,x_cols=new_x_cols)

    print('optimize RF')
    rf_model = do_rf_search(xtrn,ytrn)
    rf_score_dict = cal_scores(ytest, rf_model.predict(xtest), header_str ='test_')
    print(rf_score_dict)
    rf_dict = rf_model.get_params()
    # save rf model 
    #with open(data.model_folder +f'{poll_name}_rf_model.pkl','wb') as f:
    #    pickle.dump(rf_model, f)

    pickle.dump(rf_model, open(data.model_folder +f'{poll_name}_rf_model.pkl', 'wb'))

    # build feature of importance using build in rf
    importances = rf_model.feature_importances_
    feat_imp = pd.DataFrame(importances, index=x_cols, columns=['importance']) 
    feat_imp = feat_imp.sort_values('importance',ascending=False).reset_index()
    show_fea_imp(feat_imp,filename=data.report_folder + f'{poll_name}_rf_fea1.png', title='rf feature of importance(default)')
    # custom feature of importance
    fea_imp = feat_importance(rf_model,xtrn,ytrn,x_cols,n_iter=50)
    show_fea_imp(fea_imp,filename=data.report_folder + f'{poll_name}_rf_fea2.png', title='rf feature of importance(shuffle)')

    # print('optimize tpot')
    # tpot = TPOTRegressor( generations=5, population_size=50, verbosity=2,n_jobs=-1)
    # tpot.fit(xtrn, ytrn)
    # tpot.export(data.model_folder + 'tpot.py')
    # tpot_model = tpot.fitted_pipeline_
    # tpot_score_dict = cal_scores(ytest, tpot_model.predict(xtest), header_str ='test_')
    # print(tpot_score_dict)
    # tpot_dict = tpot_model.get_params()
    # # save tpot model 
    # with open(data.model_folder + f'{poll_name}_tpot_model.pkl','wb') as f:
    #     pickle.dump(tpot_model, f)

    # # custom feature of importance
    # fea_imp = feat_importance(tpot_model,xtrn,ytrn,x_cols,n_iter=50)
    # show_fea_imp(fea_imp,filename=data.report_folder + 'tpot_fea.png',title='tpot feature of importance')

    # # build model meta and save 
    # # create a pollution meta 
    # poll_meta =  { 'x_cols': x_cols.to_list(),
    #                 'fire_dict': data.fire_dict,
    #                 'rf_score': rf_score_dict,
    #                 'rf_params': rf_dict,
    #                 'tpot_score': tpot_score_dict
    # }

    poll_meta =  { 'x_cols': x_cols.to_list(),
                    'fire_cols':fire_cols.to_list(),
                    'fire_dict': data.fire_dict,
                    'rf_score': rf_score_dict,
                    'rf_params': rf_dict}

    model_meta = load_meta(data.model_folder + 'model_meta.json')
    model_meta[pollutant] = poll_meta
    save_meta(data.model_folder + 'model_meta.json', model_meta)
     
    return data, rf_model , poll_meta

    