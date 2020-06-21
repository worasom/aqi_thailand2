# -*- coding: utf-8 -*-
from ..imports import *
from ..gen_functions import *

def load_meta(meta_filename:str):
    """Read model_meta dictionary and return model_meta dicitonary

    Args:
        meta_filename: model meta filename 
      

    Return: dict
        pollutant_meta dictionary for that pollutant 

    """

    with open(meta_filename) as f:
        model_meta = json.load(f)

    return model_meta

def save_meta(meta_filename:str, model_meta):
    """Save model_meta

    Args:
        meta_filename: model_meta filename 
        
    Returns(optional): model meta dictionary 
        
    """ 

    with open(meta_filename, 'w') as f:
        json.dump(model_meta, meta_filename)


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
            print('old score', base_score, 'new score', score)
            print('drop', col)
        
    # obtain the final model 
    
    xtrn, ytrn, x_cols = dataset.get_data_matrix(use_index=trn_index,x_cols=x_cols)
    xval, yval, _ = dataset.get_data_matrix(use_index=val_index,x_cols=x_cols)
    model.fit(xtrn, ytrn)
    score = cal_scores(yval, model.predict(xval), header_str ='')['r2_score']   
    
    print('use columns', x_cols)
    print('r2 score', score)
    return model, x_cols

def sk_op_fire(dataset, model, trn_index, val_index,wind_range:list=[2,20],shift_range:list=[-48,48],roll_range:list=[24, 120])-> dict:
    """Search for the best fire parameter using skopt optimization 
    
    Args: 
        dataset: dataset object 
        model: model object
        trn_index: datetime index for training set
        val_index: datetime index of validation set 
        wind_range(optional): min and max value of wind speed 
        shift_range(optional): min and max value of shift parameter
        roll_range(optional): min and max value of roll parameter
        
    Return: fire_dict fire dictionary 
    
    """
    
    # check the baseline 
    dataset.merge_fire(dataset.fire_dict)
    x_cols = dataset.x_cols
    # establish the baseline 
    xtrn, ytrn, x_cols = dataset.get_data_matrix(use_index= trn_index,x_cols=x_cols)
    xval, yval, _ = dataset.get_data_matrix(use_index=val_index, x_cols=x_cols)
    
    model.fit(xtrn,ytrn)
    best_score = r2_score(yval,model.predict(xval))
    best_fire_dict = dataset.fire_dict
    print('old score', best_score, 'fire dict', fire_dict)
    
    print('optimizing fire parameter using skopt optimizer. This will take about 15 mins')
    # build search space 
    wind_speed = Real(low=wind_range[0], high=wind_range[1], name='wind_speed')
    shift = Integer(low=shift_range[0], high=shift_range[1], name='shift')
    roll = Integer(low=roll_range[0],high=roll_range[1], name='roll')
   
    # setup the function for skopt
    @use_named_args(dimensions)
    def fit_with( wind_speed, shift, roll):
        # function to return the score (smaller better)
        fire_dict = { 'w_speed': wind_speed, 
                      'shift': shift,
                      'roll': roll}
        data.merge_fire(fire_dict)
    
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
        best_fire_dict = {'w_speed': int(wind_speed), 'shift': shift, 'roll': roll}
        plot_objective(gp_result)
    else:
        print(f'old fire parameter {best_score} is still better than optimized score ={score}' )
        
    return best_fire_dict