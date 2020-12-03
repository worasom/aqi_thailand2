# -*- coding: utf-8 -*-
 
import os
import sys
import logging


if __package__: 
    from ..imports import *
    from ..gen_functions import *
    from ..features.dataset import Dataset
    from ..visualization.vis_model import *
    from .predict_model import *

else:
    # import anything in the upper directory 
    _i = os.path.dirname(os.path.dirname(os.path.abspath("..")))
    if _i not in sys.path:
        sys.path.insert(0, _i)
    from imports import *
    from gen_functions import *
    from features.dataset import Dataset
    from visualization.vis_model import *
    # run as a script, use absolute import
    _i = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _i not in sys.path:
        sys.path.insert(0, _i)

    from predict_model import *
    
"""Trainer object and optimization functions used during training.

"""

def load_meta(meta_filename: str):
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


def save_meta(meta_filename: str, model_meta):
    """Save model_meta

    Args:
        meta_filename: model_meta filename

    Returns(optional): model meta dictionary

    """

    with open(meta_filename, 'w') as f:
        json.dump(model_meta, f)


def do_rf_search(
        x_trn: np.array,
        y_trn: np.array, sample_weight=[],
        cv_split: str = 'time',
        n_splits: int = 5,
        param_dict: dict = None,
        x_tree=False,
        n_jobs=-2):
    """Perform randomize parameter search for randomforest regressor return the best estimator

    Args:
        x_trn: 2D array of x data
        y_trn: 2D np.array of y data
        sample_weight: sample weight for fitting 
        cv_split(optional): if "time". Use TimeSeriesSplit, which don't shuffle the dataset.
        n_splits(optional): number of cross validation split [default:5]
        params_dict(optional): search parameter dictionary [default:None]
        x_tree(optional): if True, use ExtraTreesRegressor instead of RandomForestRegressor
        n_jobs(optional): number of CPU use [default:-1]

    Returns: best estimator

    """
    logger = logging.getLogger(__name__)
    
    if x_tree:
        m = GradientBoostingRegressor(random_state=42)
    else:
        # rf
        m = RandomForestRegressor(random_state=42)

    if param_dict is None:
        if x_tree:

            param_dict = {
                'n_estimators': range(20, 200, 20),
                'min_samples_split': [2, 5, 10, 20, 50],
                'criterion': ['mse', 'mae'],
                'max_depth': [3, None],
                'max_features': ['auto', 'sqrt', 'log2']}
        else:
            param_dict = {'n_estimators': range(20, 300, 20),
                          'max_depth': [3, None],
                          'min_samples_split': [2, 5, 10, 20, 50],
                          'max_features': ['auto', 'sqrt', 'log2'],
                          'bootstrap': [True, False],
                          'min_samples_leaf': range(1, x_trn.shape[1])}

    if cv_split == 'time':
        cv = TimeSeriesSplit(n_splits=n_splits)

    else:
        cv = n_splits
    # hyper parameter tuning
    search = RandomizedSearchCV(
        m,
        param_distributions=param_dict,
        n_iter=100,
        n_jobs=n_jobs,
        cv=cv,
        random_state=40)

    if len(sample_weight)==0:
        # if didn't input anything, use uniform weights
        sample_weight = np.ones(len(y_trn))
    
    search.fit(x_trn, y_trn, sample_weight=sample_weight)

    logger.info(f'best estimator{search.best_params_}')
    msg = f'best rf score  {search.best_score_}'
    logger.info(msg)
    print(msg)

    return search.best_estimator_


def reduce_cols(dataset, x_cols: list, to_drop: list, model, trn_i, val_i):
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
    logger = logging.getLogger(__name__)
    logger.info(f'old cols length {len(x_cols)}')
    trn_index = dataset.split_list[trn_i]
    val_index = dataset.split_list[val_i]

    for col in tqdm(to_drop):

        xtrn, ytrn, x_cols, weights = dataset.get_data_matrix(
            use_index=trn_index, x_cols=x_cols)
        xval, yval, _, sample_weight = dataset.get_data_matrix(
            use_index=val_index, x_cols=x_cols)

        # obtain the baseline data
        model.fit(xtrn, ytrn, weights)
        base_score = cal_scores(
            yval,
            model.predict(xval),
            header_str='', sample_weight=sample_weight)['mean_squared_error']

        new_cols = x_cols.copy()
        new_cols.remove(col)
        xtrn, ytrn, new_x_cols, weights = dataset.get_data_matrix(
            use_index=trn_index, x_cols=new_cols)
        xval, yval, _, sample_weight = dataset.get_data_matrix(
            use_index=val_index, x_cols=new_cols)

        model.fit(xtrn, ytrn, weights)
        score = cal_scores(
            yval,
            model.predict(xval),
            header_str='', sample_weight=sample_weight)['mean_squared_error']

        if score < base_score:
            x_cols.remove(col)
            logger.info(f'drop  {col} improve the RMSE score from {base_score} to {score}')

    # obtain the final model

    xtrn, ytrn, x_cols, weights = dataset.get_data_matrix(
        use_index=trn_index, x_cols=x_cols)
    xval, yval, _, sample_weight = dataset.get_data_matrix(use_index=val_index, x_cols=x_cols)
    model.fit(xtrn, ytrn, weights)
    score_dict = cal_scores(yval, model.predict(xval), header_str='', sample_weight=sample_weight)

    #print('use columns', x_cols)
    msg = f'score after dropping columns  {score_dict}'
    logger.info(msg)
    print(msg)
    return model, x_cols


def sk_op_fire(dataset,
               model, split_ratio:list,
               wind_range: list = [0.5,
                                   20],
               shift_range: list = [-72,
                                    72],
               roll_range: list = [24,
                                   240],
               with_lag=False, mse=True, n_jobs=-2) -> dict:
    """Search for the best fire parameter using skopt optimization

    Args:
        dataset: dataset object
        model: model object
        split_ratio: a list of spliting ratio for train and validation set 
        wind_range(optional): min and max value of wind speed
        shift_range(optional): min and max value of shift parameter
        roll_range(optional): min and max value of roll parameter
        with_lag(optional): if True optimized the data with lag columns
        mse(optional): if True, use MSE for the loss function, if False, use -r2_score.
        n_jobs(optional): number of CPU

    Return: fire_dict fire dictionary

    """
    logger = logging.getLogger(__name__)
    # check the baseline
    _, *args = dataset.merge_fire(dataset.fire_dict, damp_surface=dataset.fire_dict['damp_surface'], wind_damp=dataset.fire_dict['wind_damp'], wind_lag=dataset.fire_dict['wind_lag'], split_direct=dataset.fire_dict['split_direct'] )

    if with_lag:
        logger.info('optimize fire dict with lag columns')
        dataset.data_org = dataset.data[[dataset.monitor] + dataset.x_cols_org]
        dataset.build_lag(
            lag_range=np.arange(
                1,
                dataset.lag_dict['n_max'],
                dataset.lag_dict['step']),
            roll=dataset.lag_dict['roll'])

    x_cols = dataset.x_cols
    logger.info(f'skop_ fire use x_cols {x_cols}')
    # establish the baseline
    dataset.split_data(split_ratio=split_ratio)
    trn_index=dataset.split_list[0] 
    val_index=dataset.split_list[1]
    xtrn, ytrn, x_cols, weights = dataset.get_data_matrix(
        use_index=trn_index, x_cols=x_cols)
    xval, yval, _, sample_weight= dataset.get_data_matrix(use_index=val_index, x_cols=x_cols)

    model.fit(xtrn, ytrn, weights)
    if mse:
        best_score = mean_squared_error(yval, model.predict(xval), sample_weight=sample_weight)
    else:
        best_score = -r2_score(yval, model.predict(xval), sample_weight=sample_weight)

    best_fire_dict = dataset.fire_dict
    fire_dict = best_fire_dict.copy()
    damp_surface = best_fire_dict['damp_surface']
    wind_damp = best_fire_dict['wind_damp']
    wind_lag = best_fire_dict['wind_lag']

    logging.info(f'old score {cal_scores(yval, model.predict(xval), sample_weight=sample_weight)} fire dict {best_fire_dict}')

    print('optimizing fire parameter using skopt optimizer. This will take about 20 mins')
    
    # build search space
    wind_speed = Real(low=wind_range[0], high=wind_range[1], name='wind_speed')
    shift = Integer(low=shift_range[0], high=shift_range[1], name='shift')
    roll = Integer(low=roll_range[0], high=roll_range[1], name='roll')

    dimensions = [wind_speed, shift, roll]
    # setup the function for skopt
    @use_named_args(dimensions)
    def fit_with(wind_speed, shift, roll):
        # function to return the score (smaller better)
        fire_dict.update({'w_speed': wind_speed,
                     'shift': shift,
                     'roll': roll})

        _, *args = dataset.merge_fire(fire_dict, damp_surface=damp_surface, wind_damp=wind_damp, wind_lag=wind_lag, split_direct=fire_dict['split_direct'])

        if with_lag:
            dataset.data_org = dataset.data[[
                dataset.monitor] + dataset.x_cols_org]
            dataset.build_lag(
                lag_range=np.arange(
                    1,
                    dataset.lag_dict['n_max'],
                    dataset.lag_dict['step']),
                roll=dataset.lag_dict['roll'])


        dataset.split_data(split_ratio=split_ratio)
        trn_index=dataset.split_list[0] 
        val_index=dataset.split_list[1]
        xtrn, ytrn, x_cols, weights = dataset.get_data_matrix(
                use_index=trn_index, x_cols=dataset.x_cols)
        xval, yval, _, sample_weight = dataset.get_data_matrix(
                use_index=val_index, x_cols=dataset.x_cols)

        model.fit(xtrn, ytrn, weights)
        y_pred = model.predict(xval)
        if mse:
            return mean_squared_error(yval, y_pred, sample_weight=sample_weight)
        else:
            return -r2_score(yval, y_pred, sample_weight=sample_weight)


    gp_result = gp_minimize(
        func=fit_with,
        dimensions=dimensions,
        n_jobs=n_jobs, random_state=30)

    wind_speed, shift, roll = gp_result.x
    logger.info(f'score for the optimized fire parameters {gp_result.fun}')
    score = gp_result.fun
    if score < best_score:
         
        best_fire_dict.update( {
            'w_speed': float(wind_speed),
            'shift': int(shift),
            'roll': int(roll), 
            'damp_surface': damp_surface, 
            'wind_damp': wind_damp, 'wind_lag': wind_lag})
         
        msg = f'new fire parameter {best_fire_dict} give score = {score}'
        logger.info(msg)
        print(msg)
        
    else:
        msg = f'old fire parameter {best_fire_dict} give score={best_score} is still better than optimized score ={score}'
        logger.info(msg)
        print(msg)

    return best_fire_dict, gp_result

def sk_op_fire_w_damp(dataset, model, split_ratio:list, wind_range: list = [0.5,
                    20], shift_range: list = [-72,
                     72], roll_range: list = [24,
                    240], surface_range:list = [1, 6], mse=True, n_jobs=-2):
    """Search for the best fire parameter using skopt optimization. This function search for possible of using wind_damp, wind_lag, and different damp_surface, which will take longer than sk_op_fire()

    Args:
        dataset: dataset object
        model: model object
        split_ratio: a list of spliting ratio for train and validation set 
        wind_range(optional): min and max value of wind speed
        shift_range(optional): min and max value of shift parameter
        roll_range(optional): min and max value of roll parameter
        surface_range(optional): min and max value of the damp_surface range 
        mse(optional): if True, use MSE for the loss function, if False, use -r2_score.
        n_jobs(optional): number of CPU
    
    Returns:
        best_fire_dict: dictionary of the best fire parameter 
        gp_result: gp_result object for visualization 

    """
    logger = logging.getLogger(__name__)
    _, *args = dataset.merge_fire(dataset.fire_dict, damp_surface=dataset.fire_dict['damp_surface'], wind_damp=dataset.fire_dict['wind_damp'], wind_lag=dataset.fire_dict['wind_lag'], split_direct=dataset.fire_dict['split_direct'])

    # establish the baseline
    x_cols = dataset.x_cols
    dataset.split_data(split_ratio=split_ratio)
    trn_index=dataset.split_list[0] 
    val_index=dataset.split_list[1]
    xtrn, ytrn, _, weights = dataset.get_data_matrix(use_index=trn_index, x_cols=x_cols)
    xval, yval, _, sample_weight = dataset.get_data_matrix(use_index=val_index, x_cols=x_cols)

    model.fit(xtrn, ytrn, weights)
    if mse:
        best_score = mean_squared_error(yval, model.predict(xval), sample_weight=sample_weight)
    else:
        best_score = -r2_score(yval, model.predict(xval), sample_weight=sample_weight)

    best_fire_dict = dataset.fire_dict
    fire_dict = best_fire_dict.copy()

    # build search space
    wind_speed = Real(low=wind_range[0], high=wind_range[1], name='wind_speed')
    shift = Integer(low=shift_range[0], high=shift_range[1], name='shift')
    roll = Integer(low=roll_range[0], high=roll_range[1], name='roll')
    damp_surface = Real(low=surface_range[0], high=surface_range[1], name='damp_surface')
    wind_damp = Categorical(categories=[True, False], name='wind_damp')
    wind_lag = Categorical(categories=[True, False], name='wind_lag')
    #split_direct = Categorical(categories=[True, False], name='split_direct')

    dimensions = [wind_speed, shift, roll, damp_surface, wind_damp, wind_lag]

    print('optimizing fire parameter using skopt optimizer. This will take about 4 hours')
    # setup the function for skopt
    @use_named_args(dimensions)
    def fit_with(wind_speed, shift, roll, damp_surface, wind_damp, wind_lag):
        # function to return the score (smaller better)
        fire_dict.update( {'w_speed': wind_speed,
                        'shift': shift,
                        'roll': roll, 
                        'damp_surface':damp_surface,
                    'wind_damp': wind_damp,
                    'wind_lag': wind_lag})
    
        # delete old damped_fire feature 
        try:
            del dataset.damped_fire 
        except:
            pass 
    
        _, *args = dataset.merge_fire(fire_dict, damp_surface=damp_surface, wind_damp=wind_damp, wind_lag=wind_lag, split_direct=fire_dict['split_direct'])

        dataset.split_data(split_ratio=split_ratio)
        trn_index=dataset.split_list[0] 
        val_index=dataset.split_list[1]
        xtrn, ytrn, x_cols, weights = dataset.get_data_matrix(
                    use_index=trn_index, x_cols=dataset.x_cols)
        xval, yval, _, sample_weight = dataset.get_data_matrix(
                    use_index=val_index, x_cols=dataset.x_cols)

        model.fit(xtrn, ytrn, weights)
        y_pred = model.predict(xval)
        if mse:
            return mean_squared_error(yval, y_pred, sample_weight=sample_weight)
        else:
            return -r2_score(yval, y_pred, sample_weight=sample_weight)
    
    gp_result = gp_minimize(func=fit_with, dimensions=dimensions, n_jobs=n_jobs)

    # unpack the result 
    wind_speed, shift, roll, damp_surface, wind_damp, wind_lag = gp_result.x
    logger.info(f'score for the best fire parameters {gp_result.fun}')
    score = gp_result.fun
    if score < best_score:
         
        best_fire_dict.update( {
            'w_speed': float(wind_speed),
            'shift': int(shift),
            'roll': int(roll), 
            'damp_surface': float(damp_surface), 
            'wind_damp': wind_damp, 
            'wind_lag': wind_lag})
       
        msg = f'old fire parameter {best_fire_dict} is still better than optimized score ={score}'
        logger.info(msg)
        print(msg)
         
    else:
        msg = f'old fire parameter {best_fire_dict} give score={best_score} is still better than optimized score ={score}'
        logger.info(msg)
        print(msg)

    return best_fire_dict, gp_result


def op_lag(
    dataset,
    model,
    split_ratio,
    lag_range=[
        1,
        100],
        step_range=[
            1,
        10], mse=True, n_jobs=-2):
    """Search for the best lag parameters using skopt optimization

    Args:
        dataset: dataset object
        model: model object
        split_ratio: list of split ratio
        lag_range(optional): min and max value of wind speed
        step_range(optional): min and max value of shift parameter
        mse(optional): if True, use MSE for the loss function, if False, use -r2_score
        n_jobs(optional): number of CPUs

    Return: fire_dict fire dictionary

    """
    logger = logging.getLogger(__name__)

    # function to return the score (smaller better)
    best_lag = dataset.lag_dict 
    dataset.build_lag(lag_range=np.arange(1, best_lag['n_max'], best_lag['step']), roll=best_lag['roll'])
    dataset.x_cols = dataset.data.columns.drop(dataset.monitor)
    dataset.split_data(split_ratio=split_ratio)
    xtrn, ytrn, x_cols, weights = dataset.get_data_matrix(
        use_index=dataset.split_list[0], x_cols=dataset.x_cols)
    xval, yval, _, sample_weight = dataset.get_data_matrix(
        use_index=dataset.split_list[1], x_cols=dataset.x_cols)
    model.fit(xtrn, ytrn, weights)
    if mse:
        best_score = mean_squared_error(yval, model.predict(xval), sample_weight=sample_weight)
    else:
        best_score = -r2_score(yval, model.predict(xval), sample_weight=sample_weight)

     
    logger.info(f'old score before adding lag {cal_scores(yval, model.predict(xval), header_str="val_", sample_weight=sample_weight)}')

    # build search space
    n_max = Integer(low=lag_range[0], high=lag_range[1], name='n_max')
    step = Integer(low=step_range[0], high=step_range[1], name='step')
    #roll = Categorical([True, False], name='roll')
    dimensions = [n_max, step]

    # setup the function for skopt
    @use_named_args(dimensions)
    def fit_with(n_max, step):
        # function to return the score (smaller better)
        dataset.build_lag(lag_range=np.arange(1, n_max, step), roll=True)
        dataset.x_cols = dataset.data.columns.drop(dataset.monitor)
        dataset.split_data(split_ratio=split_ratio)
        xtrn, ytrn, x_cols, weights = dataset.get_data_matrix(
            use_index=dataset.split_list[0], x_cols=dataset.x_cols)
        xval, yval, _, sample_weight = dataset.get_data_matrix(
            use_index=dataset.split_list[1], x_cols=dataset.x_cols)
        model.fit(xtrn, ytrn, weights)
        y_pred = model.predict(xval)

        if mse:
            return mean_squared_error(yval, y_pred, sample_weight=sample_weight)
        else:
            return -r2_score(yval, ypred, sample_weight=sample_weight)

    gp_result = gp_minimize(
        func=fit_with,
        dimensions=dimensions,
        n_jobs=n_jobs,
        random_state=30)
    n_max, step = gp_result.x
    
    score = gp_result.fun

    if score <= best_score:

        msg = f'score for the new lag_dict {score}'
        logger.info(msg)
        print(msg)

        # score after optimize is better 
        best_lag = {'n_max': int(n_max),
        'step': int(step),
        'roll': True}

    else:
        msg = f' using old lag is still better than lagged data ={best_score}'
        logger.info(msg)
        print(msg)
        

    msg = f'final lag dict is {best_lag}'
    logger.info(msg)
    print(msg)

    return best_lag, gp_result

def op_lag_fire(
    dataset,
    model,
    split_ratio,
    lag_range=[
        2,
        168],
        step_range=[
            1,
            25],
    wind_range: list = [
        0.5,
        20],
    shift_range: list = [
        -72,
        72],
    roll_range: list = [
        24,
        240],damp_surface='sphere', wind_damp=False, wind_lag=False, split_direct=False, n_jobs=-2):
    """Search for the best lag and fire parameters using skopt optimization

    Args:
        dataset: dataset object
        model: model object
        split_ratio: list of split ratio
        lag_range(optional): min and max value of wind speed
        step_range(optional): min and max value of shift parameter
        damp_surface(optional): damping factor due to distance 
        wind_damp(optional): if True, use fire feature version 2 
        wind_lag(optional): if True, use fire feature version 2 with wind time delay. 
        n_jobs(optional): number of CPUs 

    Return: fire_dict fire dictionary

    """
    # build search space
    n_max = Integer(low=lag_range[0], high=lag_range[1], name='n_max')
    step = Integer(low=step_range[0], high=step_range[1], name='step')

    # build search space
    wind_speed = Real(
        low=wind_range[0],
        high=wind_range[1],
        name='wind_speed')
    shift = Integer(low=shift_range[0], high=shift_range[1], name='shift')
    roll = Integer(low=roll_range[0], high=roll_range[1], name='roll')

    #roll = Categorical([True, False], name='roll')
    dimensions = [n_max, step, wind_speed, shift, roll]

    # setup the function for skopt
    @use_named_args(dimensions)
    def fit_with(n_max, step, wind_speed, shift, roll):
        # function to return the score (smaller better)
        fire_dict = {'w_speed': wind_speed,
                     'shift': shift,
                     'roll': roll}
        _, *args = dataset.merge_fire(fire_dict, damp_surface=damp_surface, wind_damp=wind_damp, wind_lag=wind_lag, split_direct=split_direct )
        dataset.data_org = dataset.data[[dataset.monitor] + dataset.x_cols_org]

        # function to return the score (smaller better)
        dataset.build_lag(lag_range=np.arange(1, n_max, step), roll=True)
        dataset.x_cols = dataset.data.columns.drop(dataset.monitor)

        dataset.split_data(split_ratio=split_ratio)
        xtrn, ytrn, x_cols, weights = dataset.get_data_matrix(
            use_index=dataset.split_list[0], x_cols=dataset.x_cols)
        xval, yval, _, sample_weight= dataset.get_data_matrix(
            use_index=dataset.split_list[1], x_cols=dataset.x_cols)
        model.fit(xtrn, ytrn, weights)
        y_pred = model.predict(xval)

        return mean_squared_error(yval, y_pred, sample_weight=sample_weight)

    gp_result = gp_minimize(
        func=fit_with,
        dimensions=dimensions,
        n_jobs=n_jobs,
        random_state=30)
    n_max, step, wind_speed, shift, roll = gp_result.x
    lag_dict = {'n_max': int(n_max),
                'step': int(step),
                'roll': True}
    fire_dict = {
        'w_speed': int(wind_speed),
        'shift': int(shift),
        'roll': int(roll),  'wind_damp': wind_damp, 'wind_lag': wind_lag }
    score = gp_result.fun
    print('new mean squared error', score, 'using', lag_dict and fire_dict)

    return lag_dict, fire_dict, gp_result


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
    for i, col in tqdm_notebook(enumerate(x_cols)):
        shuffle = []
        for _ in range(n_iter):
            shuffle_x = x.copy()
            shuffle_x[:, i] = np.random.permutation(shuffle_x[:, i])
            shuffle_score = score(y, model.predict(shuffle_x))
            shuffle.append(shuffle_score)
        imp.append(np.mean(shuffle))
        imp_std.append(np.std(shuffle))

    # crate a feature of importance DataFrame
    fea_imp = pd.DataFrame(
        {'index': x_cols, 'importance': imp, 'imp_std': imp_std})
    # normalized
    if score.__name__ == 'r2_score':
        fea_imp['importance'] = (baseline - fea_imp['importance']) / baseline
    elif score.__name__ == 'mean_squared_error':
        fea_imp['importance'] = (-baseline + fea_imp['importance']) / baseline

    return fea_imp.sort_values(
        'importance',
        ascending=False).reset_index(
        drop=True)


class Trainer():
    """Trainer object contains is incharge of various optimization process, and keeps track of the training parameters, which are stored as attributes.    

    Args:
        city_name: lower case of city name
        pollutant(optional): pollutant name [default: 'PM2.5']
        with_interact: add interaction term in the data matrix 
        main_data_folder(optional): main data folder for initializing Dataset object [default:'../data/]
        model_folder(optional): model folder  for initializing Dataset object [default:'../models/']
        report_folder(optional): folder to save figure for initializing Dataset object [default:'../reports/']
        n_jobs(optional): number of CPUs to use during optimization

    Attributes:
        Dataset: dataset object
        pollutant(str): pollutant type 
        poll_name(str): simplified pollutant name for saving file 
        poll_meta
        split_lists
        wind_damp
        fire_cols
        x_cols_org

    """
    # a defaul list of pollutants
    #gas_list = ['PM2.5', 'PM10', 'O3', 'CO', 'NO2', 'SO2']

    def __init__(
            self,
            city: str, pollutant: str = 'PM2.5', 
            main_data_folder: str = '../data/',
            model_folder='../models/', report_folder='../reports/', n_jobs=-2):

        """Initialize dataset object and add as attribute

        """
        logger = logging.getLogger(__name__)

        

        self.dataset = Dataset(city, main_data_folder, model_folder, report_folder)
        # remove. from pollutant name for saving file
        self.poll_name = pollutant.replace('.', '')
        # string to add to the model filename 
        
        
        # load model meta to setup parameters
        modelmeta_filename = self.dataset.model_folder + f'{self.poll_name}_model_meta.json' 
        logger.debug(f'model meta filename {modelmeta_filename}')

        try:
            self.poll_meta = load_meta(modelmeta_filename)
        except:
            print('get defaul meta')
            self.get_default_meta()

        logger.info(f'pollution meta {self.poll_meta}')

        # set number of cpu
        self.n_jobs = -2
        # load raw data
        self.dataset.load_()
        # assigning the pollution name
        self.dataset.monitor = self.dataset.pollutant = pollutant
        self.pollutant = pollutant
       
        # unpack meta setting 
        self.split_lists = self.poll_meta['split_lists']
        self.dataset.fire_dict = self.poll_meta['fire_dict']
        try:
            self.dataset.lag_dict = self.poll_meta['lag_dict']
        except:
            # no lag dict in the meta, use the default value
            self.dataset.lag_dict = {"n_max": 1, "step": 12, "roll": True}

        if 'zone_list' in self.poll_meta.keys():
            self.dataset.zone_list = self.poll_meta['zone_list']

        if 'with_interact' in self.poll_meta.keys():
            self.dataset.with_interact = self.poll_meta['with_interact']
        

        if 'log_poll' in self.poll_meta.keys():
            self.dataset.log_poll = self.poll_meta['log_poll']


        #build the first dataset only have to do this once 
        self.build_feature_no_fire()

        if 'with_traffic' in self.poll_meta.keys():
            with_traffic = self.poll_meta['with_interact']
        else:
            with_traffic = 1

        if not with_traffic:
            try: 
                # do not want traffic data 
                logger.info(f'before dropping traffic {trainer.dataset.data_no_fire.shape}')
                trainer.dataset.data_no_fire = trainer.dataset.data_no_fire.drop('traffic_index', axis=1)
                logger.info(f'after dropping traffic {trainer.dataset.data_no_fire.shape}')
            except:
                pass

        self.fire_cols, *args = self.dataset.merge_fire(self.dataset.fire_dict, damp_surface=self.dataset.fire_dict['damp_surface'], wind_damp=self.dataset.fire_dict['wind_damp'], wind_lag=self.dataset.fire_dict['wind_lag'], split_direct=self.dataset.fire_dict['split_direct'])

        # number of CPUS
        self.n_jobs = n_jobs
        # load model
        self.load_model()
        
    def build_feature_no_fire(self):
        """Call dataset.feature_no_fire function to build the new non-fire feature 

        """

        logger = logging.getLogger(__name__)

        #build the first dataset only have to do this once 
        self.dataset.feature_no_fire(
            pollutant=self.pollutant,
            rolling_win=self.poll_meta['rolling_win'],
            fill_missing=self.poll_meta['fill_missing'],
            cat_hour=self.poll_meta['cat_hour'],
            group_hour=self.poll_meta['group_hour'], cat_month=self.poll_meta['cat_month'])

        logger.info(f'data no fire columns {self.dataset.data_no_fire.columns}')


    def op_rf(self, fire_dict=None, and_save=True):
        """Optimization 1&6: optimize for the best randomforest model

        Args:
            fire_dict: fire dictonary 
            and_save(optional): if True, update and save model meta file and model file 

        """
        logger = logging.getLogger(__name__)

        print('=================find the best RF model=================')

        n_jobs_temp = self.n_jobs
        if (0 <= datetime.now().hour <= 5) or (21 <= datetime.now().hour <= 23):
            # use all cpu during the night
            self.n_jobs = -1

        if fire_dict is None:
            # use default fire feature
            self.fire_cols, *args = self.dataset.merge_fire( wind_damp=False, wind_lag=False)
        else:
            self.dataset.fire_dict = fire_dict
            self.fire_cols, *args = self.dataset.merge_fire(self.dataset.fire_dict, damp_surface=self.dataset.fire_dict['damp_surface'], wind_damp=self.dataset.fire_dict['wind_damp'], wind_lag=self.dataset.fire_dict['wind_lag'])
        
        # check x_cols attributes
        self.dataset.x_cols = self.dataset.data.columns.drop(self.pollutant).to_list()

        logger.info( f'x_cols  {self.dataset.x_cols}')
         
        self.dataset.split_data(split_ratio=self.split_lists[0])
        xtrn, ytrn, self.dataset.x_cols_org, weights = self.dataset.get_data_matrix(
            use_index=self.dataset.split_list[0], x_cols=self.dataset.x_cols)
        xval, yval, _, sample_weight = self.dataset.get_data_matrix(
            use_index=self.dataset.split_list[1], x_cols=self.dataset.x_cols)
        xtest, ytest, *args = self.dataset.get_data_matrix(
            use_index=self.dataset.split_list[2], x_cols=self.dataset.x_cols)
        

        logger.info(f'xtrn has shape {xtrn.shape}')

        self.model = do_rf_search(xtrn, ytrn, cv_split='other', sample_weight=weights, n_jobs=self.n_jobs)
        self.n_jobs = n_jobs_temp
        # set the number of cpu
        self.model.set_params(n_jobs=self.n_jobs)
        self.score_dict = cal_scores(yval,self.model.predict(xval),header_str='val_', sample_weight=sample_weight)
        msg = f'val score after op_rf {self.score_dict}'
        print(msg)
        logger.info(msg)
        score_dict = cal_scores(ytest, self.model.predict(xtest), header_str='test_')
        msg = f'test score after op_rf {score_dict}'
        logger.info(msg)
        self.update_poll_meta(x_cols = self.dataset.x_cols, x_cols_org = self.dataset.x_cols)
        if and_save:
            self.save_model()   
            self.save_meta()


    def choose_cat_hour(self, and_save=True):
        """Try to see if changing cat hour option is better. Perform after the first op_rf 
        
        Args:
            and_save(optional): if True, update and save model meta file and model file 
        
        """
        logger = logging.getLogger(__name__)

        old_score = self.score_dict['val_mean_squared_error']
        new_meta = self.poll_meta.copy()
         

        if self.poll_meta['cat_hour']:
            # use cat hour before, try to use cat hour ==False
            new_meta['cat_hour'] = 0

        else:
            # did not cat hour before, try to cat hour 
            new_meta['cat_hour'] = 1
            new_meta['group_hour'] = 3

        #build the new dataset 
        self.dataset.feature_no_fire(
            pollutant=self.pollutant,
            rolling_win=self.poll_meta['rolling_win'],
            fill_missing=self.poll_meta['fill_missing'],
            cat_hour=new_meta['cat_hour'],
            group_hour=new_meta['group_hour'], cat_month=self.poll_meta['cat_month'])

        self.fire_cols, *args = self.dataset.merge_fire(self.dataset.fire_dict, damp_surface=self.dataset.fire_dict['damp_surface'], wind_damp=self.dataset.fire_dict['wind_damp'], wind_lag=self.dataset.fire_dict['wind_lag'], split_direct=self.dataset.fire_dict['split_direct'])
        self.dataset.x_cols = self.dataset.data.columns.drop(self.pollutant).to_list()
        self.dataset.split_data(split_ratio=self.split_lists[0])
        xtrn, ytrn, self.dataset.x_cols, weights = self.dataset.get_data_matrix(
            use_index=self.dataset.split_list[0], x_cols=self.dataset.x_cols)
        xval, yval, _, sample_weight = self.dataset.get_data_matrix(
            use_index=self.dataset.split_list[1], x_cols=self.dataset.x_cols)
         
        self.model.fit(xtrn, ytrn, weights)
        new_score = cal_scores(yval,self.model.predict(xval),header_str='val_', sample_weight=sample_weight)['val_mean_squared_error']
        
        if new_score < old_score:
            msg = 'change cat hour option from ' + str(self.poll_meta['cat_hour']) + ' to ' + str(new_meta['cat_hour'])
            print(msg)
            logger.info(msg)
            # update the model 
            self.poll_meta.update(new_meta)

        else:
            msg = 'keep old cat hour option, which is ' + str( self.poll_meta['cat_hour'])
            print(msg)
            logger.info(msg)
            
        # update the model 
        #build the new dataset 
        self.dataset.feature_no_fire(
            pollutant=self.pollutant,
            rolling_win=self.poll_meta['rolling_win'],
            fill_missing=self.poll_meta['fill_missing'],
            cat_hour=self.poll_meta['cat_hour'],
            group_hour=self.poll_meta['group_hour'], cat_month=self.poll_meta['cat_month'])

        self.fire_cols, *args = self.dataset.merge_fire(self.dataset.fire_dict, damp_surface=self.dataset.fire_dict['damp_surface'], wind_damp=self.dataset.fire_dict['wind_damp'], wind_lag=self.dataset.fire_dict['wind_lag'], split_direct=self.dataset.fire_dict['split_direct'])
        self.dataset.x_cols = self.dataset.data.columns.drop(self.pollutant).to_list()
        self.dataset.split_data(split_ratio=self.split_lists[0])
        xtrn, ytrn, self.dataset.x_cols_orgs, weights = self.dataset.get_data_matrix(
            use_index=self.dataset.split_list[0], x_cols=self.dataset.x_cols)
        xval, yval, _, sample_weight = self.dataset.get_data_matrix(
            use_index=self.dataset.split_list[1], x_cols=self.dataset.x_cols)

        self.model.fit(xtrn, ytrn, weights)
        self.score_dict = cal_scores(yval,self.model.predict(xval),header_str='val_', sample_weight=sample_weight)
        logger.debug(f'dataset x_cols = {self.dataset.x_cols}')
        logger.debug(f'dataset x_cols_org = {self.dataset.x_cols_org}')
        #logger.debug(f'dataset x_cols_org = {self.dataset.x_cols_org}')
        msg = 'val score after cat_hour()' + str(self.score_dict)
        logger.info(msg)
        print(msg)
        
        
    def choose_cat_month(self, and_save=True):
        """Try to see if changing cat hour option is better. Perform after the first op_rf 
    
        Args:
            and_save(optional): if True, update and save model meta file and model file 
        
        """
        logger = logging.getLogger(__name__)
    
        old_score = self.score_dict['val_mean_squared_error']
        new_meta = self.poll_meta.copy()
         
    
        if self.poll_meta['cat_month']:
            # use cat hour before, try to use cat hour ==False
            new_meta['cat_month'] = 0
    
        else:
            # did not cat hour before, try to cat hour 
            new_meta['cat_month'] = 1
      
        #build the new dataset 
        self.dataset.feature_no_fire(
            pollutant=self.pollutant,
            rolling_win=self.poll_meta['rolling_win'],
            fill_missing=self.poll_meta['fill_missing'],
            cat_hour=new_meta['cat_hour'],
            group_hour=new_meta['group_hour'], cat_month=self.poll_meta['cat_month'])

        self.fire_cols, *args = self.dataset.merge_fire(self.dataset.fire_dict, damp_surface=self.dataset.fire_dict['damp_surface'], wind_damp=self.dataset.fire_dict['wind_damp'], wind_lag=self.dataset.fire_dict['wind_lag'], split_direct=self.dataset.fire_dict['split_direct'])
        self.dataset.x_cols = self.dataset.data.columns.drop(self.pollutant).to_list()
        self.dataset.split_data(split_ratio=self.split_lists[0])
        xtrn, ytrn, self.dataset.x_cols, weights = self.dataset.get_data_matrix(
            use_index=self.dataset.split_list[0], x_cols=self.dataset.x_cols)
        xval, yval, _, sample_weight = self.dataset.get_data_matrix(
            use_index=self.dataset.split_list[1], x_cols=self.dataset.x_cols)
         
        self.model.fit(xtrn, ytrn, weights)
        new_score = cal_scores(yval,self.model.predict(xval),header_str='val_', sample_weight=sample_weight)['val_mean_squared_error']
        
        
        if new_score < old_score:
            msg = 'change cat month option from ' + str(self.poll_meta['cat_month']) + ' to ' + str(new_meta['cat_month'])
            print(msg)
            logger.info(msg)
            # update the model 
            self.poll_meta.update(new_meta)

        else:
            msg = 'keep old cat hour option, which is ' + str( self.poll_meta['cat_month'])
            print(msg)
            logger.info(msg)

        # update the model 
        #build the new dataset 
        self.dataset.feature_no_fire(
            pollutant=self.pollutant,
            rolling_win=self.poll_meta['rolling_win'],
            fill_missing=self.poll_meta['fill_missing'],
            cat_hour=self.poll_meta['cat_hour'],
            group_hour=self.poll_meta['group_hour'], cat_month=self.poll_meta['cat_month'])

        self.fire_cols, *args = self.dataset.merge_fire(self.dataset.fire_dict, damp_surface=self.dataset.fire_dict['damp_surface'], wind_damp=self.dataset.fire_dict['wind_damp'], wind_lag=self.dataset.fire_dict['wind_lag'], split_direct=self.dataset.fire_dict['split_direct'])
        self.dataset.x_cols = self.dataset.data.columns.drop(self.pollutant).to_list()
        self.dataset.split_data(split_ratio=self.split_lists[0])
        xtrn, ytrn, self.dataset.x_cols_orgs, weights = self.dataset.get_data_matrix(
            use_index=self.dataset.split_list[0], x_cols=self.dataset.x_cols)
        xval, yval, _, sample_weight = self.dataset.get_data_matrix(
            use_index=self.dataset.split_list[1], x_cols=self.dataset.x_cols)

        self.model.fit(xtrn, ytrn, weights)
        self.score_dict = cal_scores(yval,self.model.predict(xval),header_str='val_', sample_weight=sample_weight)
        logger.debug(f'dataset x_cols = {self.dataset.x_cols}')
        logger.debug(f'dataset x_cols_org = {self.dataset.x_cols_org}')
        #logger.debug(f'dataset x_cols_org = {self.dataset.x_cols_org}')
        msg = 'val score after cat_month()' + str(self.score_dict)
        logger.info(msg)
        print(msg)


    def op2_rm_cols(self, and_save=True): 
        """ optimize 2: remove unncessary columns

        Args:
            and_save(optional): if True, update and save model meta file.

        Raises:
            AssertionError: if the self.model attribute doesn't exist. 

        """

        logger = logging.getLogger(__name__)
        print('================ remove unncessary columns no lag=================')

        if not hasattr(self, 'model'):
            raise AssertionError('model not load, use self.op_rf(), or load the model')
        
        importances = self.model.feature_importances_
        feat_imp = pd.DataFrame(
            importances,
            index=self.dataset.x_cols,
            columns=['importance'])
        feat_imp = feat_imp.sort_values(
            'importance', ascending=False).reset_index()
        
        to_drop = feat_imp['index'].to_list()
        # keep weather and fire columns for in case it matter later on 
        to_drop = [a for a in to_drop if 'fire' not in a]
        for s in ['Humidity(%)', 'Temperature(C)', 'Wind_Speed(kmph)']:
            to_drop.remove(s)
        to_drop.reverse()
        self.model, self.dataset.x_cols_org = reduce_cols(
            dataset=self.dataset, x_cols=self.dataset.x_cols, to_drop=to_drop, model=self.model, trn_i=0, val_i=1)

        self.dataset.x_cols = self.dataset.x_cols_org
        logger.debug(f'dataset x_cols = dataset x_cols_org = {self.dataset.x_cols_org}')

        self.update_poll_meta(x_cols=self.dataset.x_cols, x_cols_org = self.dataset.x_cols)

        if and_save:
            self.save_meta()
            # save feature of importance 
            feat_imp.to_csv(self.dataset.model_folder+f'{self.poll_name}_op2_featimp.py', index=False )
            show_fea_imp(feat_imp,filename=self.dataset.report_folder + f'_{self.poll_name}_rf_fea_op2.png', title='rf feature of importance(op2)')

    def op_fire(self, x_cols, mse=True, search_wind_damp=False, with_lag=False, and_save=True):
        """optimization 3: find the best fire feature before lag columns 

        Args:
            x_cols: data columns
            mse(optional): if True, use mse for the loss function, if False, use -r2_score
            search_wind_damp(optional): if True, search in four options of the fire features.
            with_lag(optional): if True, search op_fire with lag option
            and_save(optional): if True, update and save model meta file 


        Raises:
            AssertionError: if the self.dataset.fire_dict attribute doesn't exist. 
            AssertionError: if the self.model attribute doesn't exist. 

        """
        logger = logging.getLogger(__name__)
        print('================= find the best fire feature ===================')
        n_jobs_temp = self.n_jobs
        if (0 <= datetime.now().hour <= 5) or (21 <= datetime.now().hour <= 23):
            # use all cpu during the night
            self.n_jobs = -1
        # check attributes
        if not hasattr(self, 'model'):
            raise AssertionError('model not load, use self.op_rf(), or load the model')
        if not hasattr(self.dataset, 'fire_dict'):
            raise AssertionError('dataset object need fire_dict attribute, assign the default fire_dict first')

        self.dataset.x_cols = x_cols
        logger.info( f'x_cols  {self.dataset.x_cols}')
        if search_wind_damp:
            # look for fire dict using default wind_damp and wind_lag 
            self.dataset.fire_dict, gp_result = sk_op_fire_w_damp(self.dataset, self.model, split_ratio=self.split_lists[0], mse=mse, n_jobs=self.n_jobs)
        else:
            # look for fire dict using default wind_damp and wind_lag 
            self.dataset.fire_dict, gp_result = sk_op_fire(self.dataset, self.model, split_ratio=self.split_lists[0], with_lag=with_lag, mse=mse, n_jobs=self.n_jobs)
        # use the optimized columns 
        self.fire_cols, *args = self.dataset.merge_fire(self.dataset.fire_dict, damp_surface=self.dataset.fire_dict['damp_surface'], wind_damp=self.dataset.fire_dict['wind_damp'], wind_lag=self.dataset.fire_dict['wind_lag'], split_direct=self.dataset.fire_dict['split_direct'])
        
        logger.debug(f'dataset x_cols = {self.dataset.x_cols}')
        logger.debug(f'dataset x_cols_org = {self.dataset.x_cols_org}')

        self.update_poll_meta(fire_cols= self.fire_cols, fire_dict=self.dataset.fire_dict)

        self.n_jobs = n_jobs_temp
        if and_save:
            self.save_meta()

    def op_fire_zone(self, step=50, and_save=True):
        """Slowly reduce maximum fire distance use the zone_list that give the best model performance. Perform after optimizing fire_dict. 

        This is done before adding lag columns.

        
        Args: 
            step: distance in km to reduce the maximum fire distance 
            and_save: if True, update and save model meta file 

        Raises:
            AssertionError: if the self.dataset.x_cols attribute doesn't exist. 
            AssertionError: if the self.dataset.fire_dict attribute doesn't exist. 
            AssertionError: if the self.model attribute doesn't exist. 

        """
        logger = logging.getLogger(__name__)
         
        print('======== trim fire zone_list ========')

        # check attributes
        if not hasattr(self.dataset, 'x_cols'):
            msg = 'dataset object need x_cols attribute. Call self.op2_rm_cols() or load one'
            logger.exception(msg)
            raise AssertionError(msg)
        
        if not hasattr(self.dataset, 'fire_dict'):
            msg  = 'dataset object need fire_dict attribute. Call self.op_fire() or load one'
            logger.exception(msg)
            raise AssertionError(msg)
             
        # check attributes
        if not hasattr(self, 'model'):
            msg = 'model not load, use self.op_rf(), or load the model'
            logger.exception(msg)
            raise AssertionError(msg)

        logger.info( f'x_cols  {self.dataset.x_cols}')
        # check old score 
        self.dataset.split_data(split_ratio=self.split_lists[1])
        xtrn, ytrn, self.dataset.x_cols, weights = self.dataset.get_data_matrix(
            use_index=self.dataset.split_list[0], x_cols=self.dataset.x_cols)
        xval, yval, _, sample_weight = self.dataset.get_data_matrix(
            use_index=self.dataset.split_list[1], x_cols=self.dataset.x_cols)
        self.model.fit(xtrn, ytrn, weights)
        old_zone_list = self.dataset.zone_list.copy() 
        old_score = cal_scores(yval, self.model.predict(xval), header_str='val_', sample_weight=sample_weight)['val_mean_squared_error']
        logger.info(f'old score before triming fire zone {old_score} using zone list {old_zone_list}')

        for i in range(10):
        #for i in range(3):
             
            self.dataset.trim_fire_zone(step=step)
            # use the optimized columns 
            self.fire_cols, *args = self.dataset.merge_fire(self.dataset.fire_dict, damp_surface=self.dataset.fire_dict['damp_surface'], wind_damp=self.dataset.fire_dict['wind_damp'], wind_lag=self.dataset.fire_dict['wind_lag'], split_direct=self.dataset.fire_dict['split_direct'])
            # update x_cols 
            self.dataset.x_cols_org = [col for col in self.dataset.x_cols  if 'fire' not in col]  + self.fire_cols
            self.dataset.x_cols = self.dataset.x_cols_org
            
            self.dataset.split_data(split_ratio=self.split_lists[1])
            xtrn, ytrn, self.dataset.x_cols, weights = self.dataset.get_data_matrix(
                use_index=self.dataset.split_list[0], x_cols=self.dataset.x_cols)
            xval, yval, _, sample_weight = self.dataset.get_data_matrix(
                use_index=self.dataset.split_list[1], x_cols=self.dataset.x_cols)

            # check new score 
            self.model.fit(xtrn, ytrn, weights)
            score_dict = cal_scores(yval, self.model.predict(xval), header_str='val_', sample_weight=sample_weight)
            #print('score for ', self.dataset.zone_list, 'is', score_dict)
            new_score = score_dict['val_mean_squared_error']
            if new_score < old_score:
                logger.info(f'trimed fire zone to {self.dataset.zone_list}')
                # keep the new zone  and try the next iteration
                old_score = new_score
                old_zone_list = self.dataset.zone_list.copy()
              
            
        # assign the best zone_list as attribute 
        self.dataset.zone_list = old_zone_list
        # use the optimized columns 
        self.fire_cols, *args = self.dataset.merge_fire(self.dataset.fire_dict, damp_surface=self.dataset.fire_dict['damp_surface'], wind_damp=self.dataset.fire_dict['wind_damp'], wind_lag=self.dataset.fire_dict['wind_lag'])
        # update x_cols 
        self.dataset.x_cols = [col for col in self.dataset.x_cols  if 'fire' not in col]  + self.fire_cols
        
        self.dataset.split_data(split_ratio=self.split_lists[1])
        xtrn, ytrn, self.dataset.x_cols_org, weights = self.dataset.get_data_matrix(
            use_index=self.dataset.split_list[0], x_cols=self.dataset.x_cols)
        xval, yval, _, sample_weight = self.dataset.get_data_matrix(
            use_index=self.dataset.split_list[1], x_cols=self.dataset.x_cols)

        # check new score 
        self.model.fit(xtrn, ytrn, weights)
        self.score_dict = cal_scores(yval, self.model.predict(xval), header_str='val_', sample_weight=sample_weight)

        msg = f'final zone list {self.dataset.zone_list} give score {self.score_dict}'
        logger.info(msg)
        print(msg)
        logger.debug(f'dataset x_cols = {self.dataset.x_cols}')
        logger.debug(f'dataset x_cols_org = {self.dataset.x_cols_org}')
        self.update_poll_meta(fire_cols= self.fire_cols, fire_dict=self.dataset.fire_dict, zone_list=self.dataset.zone_list)
        if and_save:
            self.save_meta()

         
    def op4_lag(self, and_save=True):
        """optimization 4: improve model performance by adding lag columns and remove unncessary lag columns

        Args:
            and_save(optional): if True, update and save model meta file 

        Raises:
            AssertionError: if the self.dataset.x_cols_org attribute doesn't exist. 
            AssertionError: if the self.dataset.fire_dict attribute doesn't exist. 
            AssertionError: if the self.model attribute doesn't exist. 

        """
        logger = logging.getLogger(__name__)

        n_jobs_temp = self.n_jobs
        if (0 <= datetime.now().hour <= 5) or (21 <= datetime.now().hour <= 23):
            # use all cpu during the night
            self.n_jobs = -1

        # check attributes
        if not hasattr(self.dataset, 'x_cols_org'):
            raise AssertionError('dataset object need x_cols_org attribute. Call self.op2_rm_cols() or load one')
        
        if not hasattr(self.dataset, 'fire_dict'):
            raise AssertionError('dataset object need fire_dict attribute. Call self.op_fire() or load one')
             
        # check attributes
        if not hasattr(self, 'model'):
            raise AssertionError('model not load, use self.op_rf(), or load the model')

        print('===== improve model performance by adding lag columns =====')
         
        logger.info( f'x_cols_org before op lag_dict {self.dataset.x_cols_org}')
        # save dataframe without lag 
        self.dataset.data_org = self.dataset.data[[self.pollutant] + self.dataset.x_cols_org]

        if (self.dataset.with_interact ) & (self.dataset.fire_dict['split_direct']):
            lag_range = [1, 20]
        elif  (self.dataset.with_interact ) & (not self.dataset.fire_dict['split_direct']):
            lag_range = [1, 30]
        else:
            lag_range = [1, 100]
         
        # look for the best lag
        self.dataset.lag_dict, gp_result = op_lag(
            self.dataset, self.model, split_ratio=self.split_lists[1], lag_range=lag_range, n_jobs=self.n_jobs)
        #dataset.lag_dict = {'n_max': 2, 'step': 5}
        self.dataset.build_lag(
            lag_range=np.arange(
                1,
                self.dataset.lag_dict['n_max'],
                self.dataset.lag_dict['step']),
            roll=self.dataset.lag_dict['roll'])
         
        self.dataset.x_cols = self.dataset.data.columns.drop(self.pollutant)
        print(f'lag_dict is {self.dataset.lag_dict}')
        logger.info(f'lag_dict is {self.dataset.lag_dict}')
        logger.debug( f'x_cols after op lag_dict {self.dataset.x_cols}')
        # see the scores after the lag columns are added
        self.dataset.split_data(split_ratio=self.split_lists[1])
        xtrn, ytrn, self.dataset.x_cols, weights = self.dataset.get_data_matrix(
            use_index=self.dataset.split_list[0], x_cols=self.dataset.x_cols)
        xval, yval, _, sample_weight= self.dataset.get_data_matrix(
            use_index=self.dataset.split_list[1], x_cols=self.dataset.x_cols)
        xtest, ytest, *args = self.dataset.get_data_matrix(
            use_index=self.dataset.split_list[2], x_cols=self.dataset.x_cols)
        logger.info(f'xtrn has shape  {xtrn.shape}')
        self.model.fit(xtrn, ytrn, weights)
        self.score_dict   = cal_scores(yval, self.model.predict(xval), header_str='val_', sample_weight=sample_weight)
        msg = f'op4 score {self.score_dict}'
        print(msg)
        logger.info(msg)
        score_dict = cal_scores(ytest,self.model.predict(xtest),header_str='test_')
        logger.info(f'op4 test score {score_dict}')

        self.n_jobs = n_jobs_temp

        print('================= remove unncessary lag columns =================')
        importances = self.model.feature_importances_
        feat_imp = pd.DataFrame(
            importances,
            index=self.dataset.x_cols,
            columns=['importance'])
        feat_imp = feat_imp.sort_values(
            'importance', ascending=False).reset_index()
        

        # optimize 1 drop unuse cols
        to_drop = feat_imp['index'].to_list()
        if self.dataset.city_name== 'Chiang Mai':
            no_drop = ['Humidity(%)', 'Temperature(C)', 'Wind_Speed(kmph)'] + [a for a in self.dataset.x_cols_org if 'fire' in a]
            for s in no_drop:
                to_drop.remove(s)
        to_drop.reverse()
        self.model, self.dataset.x_cols = reduce_cols(
            dataset=self.dataset, x_cols=self.dataset.x_cols, to_drop=to_drop, model=self.model, trn_i=0, val_i=1)
        
        logger.info( f'x_cols after remove columns {self.dataset.x_cols}')
        self.update_poll_meta()
        if and_save:
            # save feature of importance 
            feat_imp.to_csv(self.dataset.model_folder+f'{self.poll_name}_op5_featimp.py', index=False )
            self.save_meta()

    def op6_rf(self, and_save=True):
        """optimization 6: optimize for the best rf again
        
        Args:
            and_save(optional): if True, update and save model meta file 

        """
        logger = logging.getLogger(__name__)

        n_jobs_temp = self.n_jobs
        if (0 <= datetime.now().hour <= 5) or (21 <= datetime.now().hour <= 23):
            # use all cpu during the night
            self.n_jobs = -1

        self.dataset.split_data(split_ratio=self.split_lists[1])

        logger.info(f'x_cols {self.dataset.x_cols}')
        xtrn, ytrn, self.dataset.x_cols, weights = self.dataset.get_data_matrix(
            use_index=self.dataset.split_list[0], x_cols=self.dataset.x_cols)
        xval, yval, _, sample_weight = self.dataset.get_data_matrix(
            use_index=self.dataset.split_list[1], x_cols=self.dataset.x_cols)
        xtest, ytest, *args = self.dataset.get_data_matrix(
            use_index=self.dataset.split_list[2], x_cols=self.dataset.x_cols)

        self.model = do_rf_search(xtrn, ytrn,cv_split='other', sample_weight=weights, n_jobs=self.n_jobs)
        self.n_jobs = n_jobs_temp
        self.score_dict = cal_scores(
        yval,
        self.model.predict(xval),
        header_str='val_', sample_weight=sample_weight)

        msg = f'val score after op6 {self.score_dict}'
        print(msg)
        logger.info(msg)
        score_dict = cal_scores(ytest, self.model.predict(xtest), header_str='testop6_')
        msg = f'test score after op6  {score_dict}'
        print(msg)
        logger.info(msg)

        if and_save:
            self.save_all()

        return score_dict

    def final_fit(self):
        """Merge train and validation data to perform the final fit

        """

        logger = logging.getLogger(__name__)
        # final split
        self.dataset.split_data(split_ratio=self.split_lists[2])
        xtrn, ytrn, self.dataset.x_cols, weights = self.dataset.get_data_matrix(
            use_index=self.dataset.split_list[0], x_cols=self.dataset.x_cols)
        xtest, ytest, _, test_weights = self.dataset.get_data_matrix(
             use_index=self.dataset.split_list[1], x_cols=self.dataset.x_cols)    


        self.model.fit(xtrn, ytrn, weights)
        ytest_pred = self.model.predict(xtest)

        self.score_dict = cal_scores(ytest, ytest_pred, header_str='test_', sample_weight=test_weights)
        msg = f'final score for test set {self.score_dict}'
        print(msg)
        logger.info(msg)
        # calculate the daily prediction error 
        ytest_pred_df = pd.DataFrame(ytest, index=self.dataset.split_list[1], columns=['actual'])
        ytest_pred_df['pred'] = ytest_pred 
        ytest_pred_df = ytest_pred_df.resample('d').mean().dropna()
        avg_score_dict = cal_scores(ytest_pred_df['actual'].values, ytest_pred_df['pred'].values, header_str='avg_trn_')
        msg = f'daily avg score for test set {avg_score_dict}'
        print(msg)
        logger.info(msg)

        self.update_poll_meta(rf_avg_score=avg_score_dict)

        if self.dataset.log_poll:
            score_dict_log = self.score_dict 
            ytest = np.exp(ytest)
            ytest_pred = np.exp(ytest_pred)
            self.score_dict = cal_scores(ytest, ytest_pred, header_str='test_', sample_weight=test_weights)
            msg = f'final score for test set after removing log {self.score_dict}'
            print(msg)
            logger.info(msg)

            avg_score_dict_log = avg_score_dict

            # calculate the daily prediction error 
            ytest_pred_df = pd.DataFrame(ytest, index=self.dataset.split_list[1], columns=['actual'])
            ytest_pred_df['pred'] = ytest_pred 
            ytest_pred_df = ytest_pred_df.resample('d').mean().dropna()
            avg_score_dict = cal_scores(ytest_pred_df['actual'].values, ytest_pred_df['pred'].values, header_str='avg_trn_')
            msg = f'daily avg score for test set after removing log {avg_score_dict}'
            print(msg)
            logger.info(msg)

            self.poll_meta['rf_avg_score'] = avg_score_dict
            self.poll_meta['score_dict_log'] = score_dict_log
            self.poll_meta['avg_score_dict_log'] = avg_score_dict_log


    def search_tpot(self):
        """Search for TPOT model, explore the best pipeline, and print out the best score. This step is done after obtaining best fire parameter and lag dict.

        """

        # feature engineering settting
        self.dataset.lag_dict = self.poll_meta['lag_dict']
        self.dataset.x_cols_org = self.poll_meta['x_cols_org']
        
        # build data matrix 
        self.fire_cols, *args = self.dataset.merge_fire(self.dataset.fire_dict, damp_surface=self.dataset.fire_dict['damp_surface'], wind_damp=self.dataset.fire_dict['wind_damp'], wind_lag=self.dataset.fire_dict['wind_lag'], split_direct=self.dataset.fire_dict['split_direct'])
        self.dataset.build_lag(
        lag_range=np.arange(
            1,
            self.dataset.lag_dict['n_max'],
            self.dataset.lag_dict['step']),
        roll=self.dataset.lag_dict['roll'])
        self.dataset.x_cols = self.dataset.data.columns.drop(self.pollutant)
        self.dataset.split_data(split_ratio=self.split_lists[2])

        xtrn, ytrn, self.dataset.x_cols, weights = self.dataset.get_data_matrix(
            use_index=self.dataset.split_list[0], x_cols=self.dataset.x_cols)
        #xval, yval, _, sample_weight = self.dataset.get_data_matrix(
        #    use_index=self.dataset.split_list[1], x_cols=self.dataset.x_cols)
        xtest, ytest, _, sample_weight = self.dataset.get_data_matrix(
            use_index=self.dataset.split_list[1], x_cols=self.dataset.x_cols)

        #ask TPOT to hunt for the best model
        tpot = TPOTRegressor(generations=5, population_size=50, verbosity=2)
        tpot.fit(xtrn, ytrn, sample_weight=weights)

        score_dict = cal_scores(ytest, tpot.predict(xtest), header_str='test_', sample_weight=sample_weight)
        print(f'Tpot test score is {score_dict}')
        tpot.export(self.dataset.model_folder+f'{self.pollutant}_tpot_pipeline.py')

        pickle.dump(tpot, open( self.dataset.model_folder + f'{self.poll_name}_tpot_model.pkl', 'wb'))


    def save_feat_imp(self, filename=None, title=''):
        """Build feature of importance plots and save the plot as png file
        
        Args:
            filename: if not None, save the plot as a filename  

        """
        # build feature of importance using build in rf
        try:
            importances = self.model.feature_importances_
            feat_imp = pd.DataFrame(
                importances,
                index=self.dataset.x_cols,
                columns=['importance'])
            feat_imp = feat_imp.sort_values(
                'importance', ascending=False).reset_index()
            #show_fea_imp(feat_imp,filename=dataset.report_folder + f'{poll_name}_rf_fea_op2.png', title='rf feature of importance(default)')
        except:
            # custom feature of importance
            xtrn, ytrn, dataset.x_cols, weights = self.dataset.get_data_matrix(
                use_index=self.dataset.split_list[0], x_cols=self.dataset.x_cols)
            feat_imp = feat_importance(self.model, xtrn, ytrn, self.dataset.x_cols, n_iter=50)

        # obtain feature of importance without lag
        feat_imp['index'] = feat_imp['index'].str.split('_lag_', expand=True)[0]
        feat_imp = feat_imp.groupby('index').sum()
        feat_imp = feat_imp.sort_values(
            'importance', ascending=False).reset_index()
        feat_imp.to_csv(self.dataset.model_folder+f'{self.poll_name}_final_featimp_with_interact.py', index=False )

        # split the interaction terms if exist 
        temp = feat_imp['index'].str.split('_n_', expand=True)
        df1 = pd.concat([temp[0], feat_imp['importance']], axis=1)
        df1.columns = ['index' , 'importance']
        
        if 1 in temp.columns:
            df2 = pd.concat([temp[1], feat_imp['importance']], axis=1).dropna()
            df2.columns = ['index' , 'importance']
            feat_imp = pd.concat([df1, df2], ignore_index=True)
            
        feat_imp = feat_imp.groupby('index').sum()
        feat_imp = feat_imp.sort_values(
                    'importance', ascending=False).reset_index()

        feat_imp.to_csv(self.dataset.model_folder+f'{self.poll_name}_final_featimp.py', index=False )
        show_fea_imp(feat_imp, filename=filename, title=title)

    def get_default_meta(self, **kwargs):
        """Setup pollution meta dictionary and add as self.poll_meta attribute 

        Args:
            keywords arguments for overiding default poll_meta

        """
        logger = logging.getLogger(__name__)
        logger.info('use default meta')
        print('use default meta')
        poll_meta = {"rolling_win": 1, "cat_hour": True, "cat_month":True, "with_interact":False, "split_direct":False, "fill_missing": True, "group_hour": 2, "split_lists": [[0.4, 0.3, 0.3], [0.45, 0.25, 0.3], [0.7, 0.3]]}
        poll_meta.update(kwargs)
        poll_meta['fire_dict'] = {'w_speed': 7, 'shift': -5, 'roll': 44, 'damp_surface': 2, 'wind_damp': False, 'wind_lag': False, 'split_direct': False}
        poll_meta['lag_dict'] ={"n_max": 1, "step": 12, "roll": True}

        self.poll_meta = poll_meta

    def update_poll_meta(self, **kwargs):
        """Update self.poll_meta attribute 

        """
        try:
            self.poll_meta.update({'x_cols_org': self.dataset.x_cols_org,
            'x_cols': self.dataset.x_cols,
            'zone_list': self.dataset.zone_list,
            'fire_cols': self.fire_cols,
            'fire_dict': self.dataset.fire_dict,
            'lag_dict': self.dataset.lag_dict,
            'rf_score': self.score_dict,
            'rf_params': self.model.get_params()
            })
        except:
            pass

        self.poll_meta.update(kwargs)

    def save_meta(self):
        """Save self.poll_meta 

        """
        # load model meta to setup parameters
        poll_meta = load_meta(self.dataset.model_folder + f'{self.poll_name}_model_meta.json')
        if 'fire_dict' in self.poll_meta.keys():
            self.poll_meta['fire_dict']['wind_damp'] = int(bool(self.poll_meta['fire_dict']['wind_damp']))
            self.poll_meta['fire_dict']['wind_lag'] = int(bool(self.poll_meta['fire_dict']['wind_lag']))
            self.poll_meta['fire_dict']['split_direct'] = int(bool(self.poll_meta['fire_dict']['split_direct']))
        
        save_meta(self.dataset.model_folder + f'{self.poll_name}_model_meta.json', self.poll_meta)

    def save_model(self):
        """Save trained model 

        """

        pickle.dump(self.model, open( self.dataset.model_folder +f'{self.poll_name}_rf_model.pkl', 'wb'))

    def save_all(self):
        """Save model meta, model file and dataset data 

        """
        self.update_poll_meta()
        self.save_meta()
        self.dataset.save_()
        self.save_model()
        

    def load_model(self):
        """Try to load the model if exist 
        
        """

        try: 
            # load model
            self.model = pickle.load(
                open(self.dataset.model_folder +f'{self.poll_name}_rf_model.pkl', 'rb'))
        except:
            pass


def train_city_s1(city:str, pollutant= 'PM2.5', n_jobs=-2, default_meta=False, 
        search_wind_damp=False, choose_cat_hour=False, choose_cat_month=True, 
        add_weight=True, op_fire_twice=False, search_tpot=False, 
        main_data_folder: str = '../data/',
        model_folder='../models/', report_folder='../reports/'):
    """Training pipeline from process raw data, hyperparameter tune, and save model.

    Args:
        city: city name
        pollutant(optional): pollutant name
        n_jobs(optional): number of CPUs to use during optimization
        default_meta(optional): if True, override meta setting with the default value 
        search_wind_damp(optional): if True, search in four options of the fire features.
        add_weight(optional): if True, use non-uniform weight when fitting and evaluating the model.
        choose_cat_hour(optional): if True, see if  adding/not adding hour as catergorical variable is better
        choose_cat_month(optional): if True, see if adding/not adding month as catergorical variable is better 
        op_fire_twice(optiohnal): if True, optimize fire data after optimizing lag 
        search_tpot(optional): If True, also search for other model using TPOT
        main_data_folder(optional): main data folder for initializing Dataset object [default:'../data/]
        model_folder(optional): model folder  for initializing Dataset object [default:'../models/']
        report_folder(optional): folder to save figure for initializing Dataset object [default:'../reports/']


    Returns:
        dataset: dataset object
        model: model object
        poll_meta(dict): parameter dictionary 

    """
    # start logging 
    set_logging(level=10)
    logger = logging.getLogger(__name__)
    # initialize a trainer object
    trainer = Trainer(city=city, pollutant=pollutant)
    trainer.n_jobs = n_jobs

    if default_meta:
        trainer.get_default_meta()

    if ~ add_weight:
        trainer.dataset.add_weight = 0
    #if 'x_cols_org' in trainer.poll_meta.keys():
    #    trainer.dataset.x_cols = trainer.dataset.x_cols_org = trainer.poll_meta['x_cols_org']
         
    # look for the best rf model 
    trainer.op_rf(fire_dict=trainer.dataset.fire_dict)
    if choose_cat_hour:
        trainer.choose_cat_hour()

    if choose_cat_month:
        trainer.choose_cat_month()
    # remove columns
    trainer.op2_rm_cols()
    logger.info(f'current columns {trainer.dataset.x_cols_org}')
    # op fire
    trainer.op_fire(x_cols=trainer.dataset.x_cols_org, search_wind_damp=search_wind_damp)
    trainer.op_fire_zone(step=50)
    
    # see if adding lag improve things 
    if trainer.dataset.with_interact:
        # use smaller lag range 
        trainer.op4_lag(lag_range=[1, 20])
    else:
        trainer.op4_lag()

    if op_fire_twice:
        trainer.op_fire(x_cols=trainer.dataset.x_cols, with_lag=True, search_wind_damp=search_wind_damp)
    # serach rf model again
    trainer.op6_rf()
    trainer.final_fit()
    # save plot
    trainer.save_feat_imp(filename=trainer.dataset.report_folder  +f'{trainer.poll_name}_rf_fea_op2_nolag.png', title='rf feature of importance')
    trainer.save_all()

    if search_tpot:
        trainer.search_tpot()

    # turn of logging 
    logging.shutdown()

    return trainer.dataset, trainer.model, trainer


def train_hyper_search(city:str, pollutant= 'PM2.5', n_jobs=-2, default_meta=False, add_weight=True, search_list=['split_direct', 'with_interact', 'with_traffic'], main_data_folder: str = '../data/', model_folder='../models/', report_folder='../reports/' ):
    """Grid search training hyperparmeter. Record the setting and model performance for each feature choice combination.

    
    Args:
        city: city name
        pollutant(optional): pollutant name
        n_jobs(optional): number of CPUs to use during optimization
        default_meta(optional): if True, override meta setting with the default value 
        add_weight(optional): if True, use non-uniform weight when fitting and evaluating the model.
        search_list(optional): feature to search 
        main_data_folder(optional): main data folder for initializing Dataset object [default:'../data/]
        model_folder(optional): model folder  for initializing Dataset object [default:'../models/']
        report_folder(optional): folder to save figure for initializing Dataset object [default:'../reports/']

    """
    set_logging(level=10)
    logger = logging.getLogger(__name__)
    # obtain the combination of parameter 
    param_list = [*product(*tuple([[0,1]]*len(search_list)))]
    logger.info(f'len of search {len(param_list)}')

    
    
    poll_name = pollutant.replace('.', '')
    city_name = city.lower().replace(' ', '_')
    # load explored parameters 
    search_filename =  model_folder + f'{city_name}/' + f'{ poll_name}_search.csv'
    try: 
        
        explored_df = pd.read_csv(search_filename)
        
        explored = list(zip(explored_df['split_direct'], explored_df['with_interact'], explored_df['with_traffic']))
        print('found exisiting search file')
        print(f'explored parameter {explored}')
    except:
        explored = []

    # remove explored item
    param_list = [ item for item in param_list if item not in explored]

    #result_df = pd.DataFrame()
    for params in tqdm(param_list):
        print(f'parameter {params}')
        # build a dictionary 
        result_dict = {}
        for k, v in zip(search_list, params):
            result_dict[k] = v

        logger.info(f'search parameter {result_dict}')

        try:
            with_traffic = result_dict['with_traffic']
        except:
            with_traffic = 1

        # initialize a trainer object
        trainer = Trainer(city=city, pollutant=pollutant)
        trainer.n_jobs = n_jobs

        if default_meta:
            trainer.get_default_meta()

        if ~ add_weight:
            trainer.dataset.add_weight = 0
    

        if not with_traffic:
            try: 
                # do not want traffic data 
                print('before dropping traffic', trainer.dataset.data_no_fire.shape)
                trainer.dataset.data_no_fire = trainer.dataset.data_no_fire.drop('traffic_index', axis=1)
                print('after dropping traffic', trainer.dataset.data_no_fire.shape)
            except:
                pass
            
        trainer.dataset.with_interact = result_dict['with_interact']
        trainer.dataset.fire_dict['split_direct'] = result_dict['split_direct']

        # look for the best rf model 
        trainer.op_rf(fire_dict=trainer.dataset.fire_dict, and_save=False)
        trainer.choose_cat_hour(and_save=False)
        # remove columns
        trainer.op2_rm_cols(and_save=False)
        trainer.op_fire(x_cols=trainer.dataset.x_cols_org, search_wind_damp=True, and_save=False)
        trainer.op_fire_zone(step=50, and_save=False)

        # see if adding lag improve things 
        trainer.op4_lag(and_save=False)


        test_score_dict = trainer.op6_rf(and_save=False)
        # add score dict into the result
        result_dict.update(trainer.score_dict)
        print(f'validation score {trainer.score_dict}')
        result_dict.update(test_score_dict)

        trainer.final_fit()
        # add score dict into the result
        result_dict.update(trainer.score_dict)
        #print(f'final test score {trainer.score_dict}')

        # add parameters 
        key_list = ['cat_hour']
        result_dict.update( { k: trainer.poll_meta[k] for k in key_list })
        result_dict['len_cols'] = len(trainer.dataset.x_cols)


        logger.info(f'search parameter gives result {result_dict}')
       

        result_df =  pd.DataFrame(result_dict, index=[0]) 

        if os.path.exists(search_filename):
            result_df.to_csv(search_filename, header=False, mode='a', index=False)
        else:
            result_df.to_csv( search_filename, index=False )

    result_df = result_df.sort_values('test_mean_squared_error')
    best_params_dict = result_df.loc[0, search_list].to_dict()

    # save the best parameter into poll_meta 
    for k in best_params_dict.keys():
        trainer.update_poll_meta(k = best_params_dict[k])
    trainer.save_meta()

    return result_df