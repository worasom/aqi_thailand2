# -*- coding: utf-8 -*-
from ..imports import *
from ..gen_functions import *
from ..features.dataset import Dataset
from ..visualization.vis_model import *

def do_knn_search(x_trn:np.array, y_trn:np.array, cv_split:str='time', n_splits:int=5,param_dict:dict=None, x_tree=True):
    """Perform randomize parameter search for KNN regressor return the best estimator 
    
    Args: 
        x_trn: 2D array of x data 
        y_trn: 2D np.array of y data
        cv_split(optional): if "time". Use TimeSeriesSplit, which don't shuffle the data.
        n_splits(optional): number of cross validation split [default:5]
        params_dict(optional): search parameter dictionary [default:None]
        x_tree(optional): if True, use ExtraTreesRegressor instead of RandomForestRegressor

    Returns: best estimator 
    """
    m = KNeighborsRegressor()

    param_dict = { 'n_neighbors': range(5,100)}

    if cv_split =='time':
        cv = TimeSeriesSplit(n_splits=n_splits)
        
    else:
        cv = n_splits
    #hyper parameter tuning
    search = RandomizedSearchCV(m, param_distributions=param_dict,
                        n_iter=100,n_jobs=-2, cv=cv, random_state=40)

    search.fit(x_trn,y_trn)
    
    print(search.best_params_, search.best_score_)
    
    return search.best_estimator_

def do_ln_search(x_trn:np.array, y_trn:np.array, cv_split:str='time', n_splits:int=5,param_dict:dict=None, ln_type='elastic', n_jobs=-2):
    """Perform randomize parameter search for linear regressir return the best estimator 

    Args: 
        x_trn: 2D array of x data 
        y_trn: 2D np.array of y data
        cv_split(optional): if "time". Use TimeSeriesSplit, which don't shuffle the data.
        n_splits(optional): number of cross validation split [default:5]
        params_dict(optional): search parameter dictionary [default:None]
        ln_type(optional): type of linear regressor can be 'elastic', or lasso [default:'elastic']
        n_jobs(optional): number of CPU use [default:-1]

    Returns: best estimator 

    """

    if ln_type == 'elastic':
        m = ElasticNet( )
    elif ln_type == 'lasso':
        m = Lasso()
    elif ln_type == 'ridge':
        m = Ridge()

    if param_dict==None:
        if ln_type == 'elastic':
            param_dict  = { "alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
                            "l1_ratio": np.arange(0.0, 1.0, 0.1)}

    if cv_split =='time':
        cv = TimeSeriesSplit(n_splits=n_splits)
    else:
        cv = n_splits
    
    #hyper parameter tuning
    search = RandomizedSearchCV(m, param_distributions=param_dict,
                        n_iter=100,n_jobs=n_jobs, cv=cv, random_state=40)     

    search.fit(x_trn,y_trn)
    
    print(search.best_params_, search.best_score_)
    
    return search.best_estimator_   

def op_lag_fire(dataset, model, split_ratio, lag_range=[2, 168], step_range=[1,25],wind_range:list=[2,20],shift_range:list=[-72,72],roll_range:list=[24, 240]):
    """Search for the best lag and fire parameters using skopt optimization 
    
    Args: 
        dataset: dataset object 
        model: model object
        split_ratio: list of split ratio
        lag_range(optional): min and max value of wind speed 
        step_range(optional): min and max value of shift parameter
        
    Return: fire_dict fire dictionary 
    
    """
    # build search space 
    n_max = Integer(low=lag_range[0], high=lag_range[1], name='n_max')
    step = Integer(low=step_range[0], high=step_range[1], name='step')
    
    # build search space 
    wind_speed = Integer(low=wind_range[0], high=wind_range[1], name='wind_speed')
    shift = Integer(low=shift_range[0], high=shift_range[1], name='shift')
    roll = Integer(low=roll_range[0],high=roll_range[1], name='roll')
    
    #roll = Categorical([True, False], name='roll')
    dimensions = [n_max, step, wind_speed, shift, roll]
    
    
    # setup the function for skopt
    @use_named_args(dimensions)
    def fit_with( n_max, step, wind_speed, shift, roll):
        # function to return the score (smaller better)
        fire_dict = { 'w_speed': wind_speed, 
                      'shift': shift,
                      'roll': roll}
        _, *args = dataset.merge_fire(fire_dict)
        dataset.data_org = dataset.data[ [dataset.monitor] + dataset.x_cols_org]
        
        # function to return the score (smaller better)
        dataset.build_lag(lag_range=np.arange(1, n_max, step), roll=True)
        dataset.x_cols = dataset.data.columns.drop(dataset.monitor)
        
        dataset.split_data(split_ratio=split_ratio)
        xtrn, ytrn, x_cols = dataset.get_data_matrix(use_index=dataset.split_list[0], x_cols=dataset.x_cols)
        xval, yval, _ = dataset.get_data_matrix(use_index=dataset.split_list[1], x_cols=dataset.x_cols)
        model.fit(xtrn,ytrn)
        y_pred = model.predict(xval)
        
        return mean_squared_error(yval,y_pred)
    
    gp_result = gp_minimize(func=fit_with,dimensions=dimensions,n_jobs=-2,random_state=30)
    n_max, step, wind_speed, shift, roll = gp_result.x
    lag_dict = {'n_max':int(n_max),
                'step':int(step),
                'roll': True}
    fire_dict = {'w_speed': int(wind_speed), 'shift': int(shift), 'roll': int(roll)}
    score = gp_result.fun
    print('new mean squared error', score, 'using', lag_dict and fire_dict )
    
    return lag_dict, fire_dict, gp_result

def get_nn_model(input_shape:int, output_shape:int, num_layer:int, nn_size:int, act_fun:str, drop:float, lr:float,momentum:float):
    """Get fully connected NN model according to the specified parameters
    
    Args:
        input_shape: input dimension (according to the len of the columns)
        output_shape: output dimension
        num_layer: number of hidden layer can be 0 
        nn_size: number of neuron in one layer
        act_fun: name of activation function
        drop: dropout parameter
        lr: optimizer learning rate
        momentum: momentum parameter
        
    """
    # set optimizser
    adam = Adam(learning_rate=lr)
    # create model 
    model = Sequential()
    # Input - Layer
    model.add(Dense(nn_size, activation=act_fun, input_dim=input_shape))
    
    if num_layer>0:
        for i in range(num_layer):
            name = f'layer_dense{i+1}'
            model.add(Dense(nn_size,
                 activation=act_fun,
                        name=name))
            model.add(BatchNormalization(momentum=momentum))
            
    model.add(Dropout(drop))
    # Output- Layer
    model.add(Dense(output_shape, activation = 'linear')) 
    model.compile(loss='mse', optimizer=adam)
    return model    

def do_nn_search(xtrn:np.array, ytrn:np.array, xval, yval,n_jobs=-2):
    """Perform skopt optimization search for the best NN architecture.

    Args:
        xtrn: normalized xtrn
        ytrn: normalized ytrn 
        xval: normalized xval
        ytrn: normalized yval 
    """

    # nn search parameters
    num_layers = Integer(low=1, high=5, name='num_layer')
    nn_sizes = Integer(low=8, high=1024, name='nn_size')
    act_funs = Categorical(categories=['relu','softplus'],
                                 name='act_fun')
    drops = Real(low=0, high=0.2,
                             name='drop')
    lrs = Real(low=1e-4, high=1e-2, prior='log-uniform',
                             name='lr')
    momentums = Real(low=0.7, high=0.99, 
                             name='momentum')

    dimensions = [num_layers,nn_sizes,act_funs,drops,lrs,momentums]

    @use_named_args(dimensions)
    def fit_with(num_layer, nn_size, act_fun, drop, lr, momentum):
        # function to return the score (smaller better)
        model = get_nn_model(input_shape=xtrn.shape[1], output_shape=ytrn.shape[1], num_layer=num_layer,
                          nn_size=nn_size, act_fun=act_fun, drop=drop, lr=lr, momentum=momentum)
        # set early stoping
        esm = EarlyStopping(patience=8, verbose=0, restore_best_weights=True)
        # train model
        history = model.fit(xtrn, ytrn, validation_split=0.2,
                            verbose=0, epochs=200, callbacks=[esm])

        y_pred = model.predict(xval)
        K.clear_session()

        return mean_squared_error(yval, y_pred)


    gp_result = gp_minimize(
        func=fit_with, dimensions=dimensions, n_jobs=n_jobs, random_state=30)
    
    num_layer, nn_size, act_fun, drop, lr, momentum = gp_result.x
    nn_params_dict = {'num_layer': int(num_layer),
                       'nn_size': int(nn_size), 
                       'act_fun': str(act_fun), 
                       'drop': round(float(drop), 3), 
                       'lr': round(float(lr), 3), 
                       'momentum': round(float(momentum), 3)}
    
    print('nn parameter is ', nn_params_dict)
    score = gp_result.fun
    print(score)

    return nn_params_dict, gp_result

def train_city_s2(city:str='Chiang Mai', pollutant:str='PM2.5', build=False, model=None, fire_dict=None, x_cols_org=[], lag_dict=None, x_cols=[]):
    """Training pipeline from process raw data, hyperparameter tune, and save model.

    #. If build True, build the raw data from files 
    #. Process draw data into data using default values
    #. Optimization 1: optimize for the best NN model 
    
    #. Build pollution meta and save

    Args:
    city: city name
    pollutant(optional): pollutant name
    build(optional): if True, also build the data
    model(optional): 
    x_cols_org(optional):
    lag_dict(optional):

    Returns: 
        dataset
        model

    """

    print('====== NN op1 load RF model ====== ')
    data, model, fire_cols = load_model1(city='Chiang Mai', pollutant='PM2.5', split_list=[0.45, 0.25, 0.3])

    if (len(x_cols_org) != 0) and (lag_dict !=None):
        data.x_cols_org = poll_meta['x_cols_org']
        print('\n x_cols_org', data.x_cols_org)
        data.data_org = data.data[ [data.monitor] + data.x_cols_org]
        data.build_lag(lag_range=np.arange(1, data.lag_dict['n_max'], data.lag_dict['step']), roll=data.lag_dict['roll'])
        data.x_cols = data.data.columns.drop(data.monitor)
         

    if len(x_cols)==0:
        data.x_cols = data.data.columns.drop(data.monitor)
        print('x_cols', data.x_cols )
    else:
        data.x_cols = x_cols 


    data.split_data(split_ratio=[0.45, 0.25, 0.3])
    xtrn, ytrn, data.x_cols = data.get_data_matrix(use_index=data.split_list[0], x_cols=data.x_cols)
    ytrn = ytrn.reshape(-1,1)
    xval, yval, _ = data.get_data_matrix(use_index=data.split_list[1], x_cols=data.x_cols)
    yval = yval.reshape(-1,1)

    x_scaler = MinMaxScaler()
    xtrn = x_scaler.fit_transform(xtrn)
    xval = x_scaler.transform(xval)
    y_scaler = MinMaxScaler()
    ytrn = y_scaler.fit_transform(ytrn)
    
    nn_dict, gp_result = do_nn_search(xtrn, ytrn, xval, yval, n_jobs=-2)
    
    model = get_nn_model(input_shape=xtrn.shape[1], output_shape=ytrn.shape[1], num_layer=nn_dict['num_layer'],nn_size=nn_dict['nn_size'],act_fun=nn_dict['act_fun'], drop=nn_dict['drop'],lr=nn_dict['lr'],momentum=nn_dict['momentum'])
    esm = EarlyStopping(patience=8,verbose=0,restore_best_weights=True)
    history = model.fit(xtrn, ytrn, validation_split=0.2,verbose=0,epochs=1000,callbacks=[esm])
    ypred = model.predict(xval)
    ypred = y_scaler.inverse_transform(ypred)
    print('validation score', cal_scores(yval, ypred, header_str='val_')) 



    return data, nn_dict, model, x_scaler, y_scaler
 
def train_city_s1_v2(city:str='Chiang Mai', pollutant:str='PM2.5', build=False, model=None, fire_dict=None, x_cols_org=[], lag_dict=None,x_cols=[]):
    """Training pipeline from process raw data, hyperparameter tune, and save model.

    #. If build True, build the raw data from files 
    #. Process draw data into data using default values
    #. Optimization 1: optimize for the best randomforest model 
    #. Optimization 2: remove unncessary columns 
    #. Optimization 3: find the best fire feature 
    #. Optimization 4: optimize for lag columns
    #. Optimization 5: drop unncessary lag columns
    #. Optimization 6: optimize for the best RF again  
    #. Build pollution meta and save

    Args:
        city: city name
        pollutant(optional): pollutant name
        build(optional): if True, also build the data
        model(optional):
        op_fire(optional):
        x_cols_org(optional):
        lag_dict(optional):
        x_cols(optional):

    Returns: 
        dataset
        model
        poll_meta(dict) 

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
    if fire_dict==None:
        # use default fire feature
        fire_cols, *args = data.merge_fire()
    else:
        data.fire_dict = fire_dict 
        fire_cols, *args = data.merge_fire(data.fire_dict)
    data.monitor = data.pollutant = pollutant
    

    #. Optimization 1: optimize for the best randomforest model 

    if (model==None) or ( len(x_cols_org) == 0):
        # split the data into 4 set
        print('=================optimize 1: find the best RF model=================')
        data.split_data(split_ratio=[0.45, 0.25, 0.3])
        xtrn, ytrn, x_cols = data.get_data_matrix(use_index=data.split_list[0] )
        xval, yval, _ = data.get_data_matrix(use_index=data.split_list[1])
        data.x_cols = x_cols

        model = do_rf_search(xtrn,ytrn, cv_split='other')
        score_dict = cal_scores(yval, model.predict(xval), header_str ='val_')
        print('optimize 1 score', score_dict)    

        print('=================optimize 2: remove unncessary columns=================')
        importances = model.feature_importances_
        feat_imp = pd.DataFrame(importances, index=x_cols, columns=['importance']) 
        feat_imp = feat_imp.sort_values('importance',ascending=False).reset_index()
        show_fea_imp(feat_imp, filename=data.report_folder + f'{poll_name}_fea_imp_op1.png', title='rf feature of importance(raw)')
        
        # columns to consider droping are columns with low importance
        to_drop = feat_imp['index']
        to_drop = [a for a in to_drop if 'fire' not in a]
        for s in ['Humidity(%)','Temperature(C)','Wind Speed(kmph)']:
            to_drop.remove(s)
        to_drop.reverse()
        model, x_cols_org = reduce_cols(dataset=data,x_cols=data.x_cols,to_drop=to_drop,model=model,trn_i=0, val_i=1)
    
    
    if fire_dict==None:
        print('================= optimization 3: find the best fire feature ===================')
        data.fire_dict, gp_result  = sk_op_fire(data, model, trn_index=data.split_list[0], val_index=data.split_list[1])
        fire_cols, *args = data.merge_fire(data.fire_dict)

    if lag_dict==None:
        print('================= optimization 4&7: improve model performance by adding lag columns =================')
        # prepare no-lag columns 
        data.x_cols_org = x_cols_org  
        data.data_org = data.data[ [data.monitor] + data.x_cols_org]
        print('model parameters', model.get_params())
        # look for the best lag 
        #data.lag_dict, gp_result = op_lag(data, model, split_ratio=[0.45, 0.25, 0.3])
        data.lag_dict, data.fire_dict, gp_result = op_lag_fire(data, model, split_ratio=[0.45, 0.25, 0.3])
        _, *args = data.merge_fire(data.fire_dict)
        data.data_org = data.data[ [data.monitor] + data.x_cols_org]
        #data.lag_dict = {'n_max': 2, 'step': 5}
        data.build_lag(lag_range=np.arange(1, data.lag_dict['n_max'], data.lag_dict['step']), roll=data.lag_dict['roll'])
        print('data.column with lag', data.data.columns)
        data.x_cols = data.data.columns.drop(data.monitor)

        print('x_cols', data.x_cols)
        data.split_data(split_ratio=[0.45, 0.25, 0.3])
        xtrn, ytrn, data.x_cols = data.get_data_matrix(use_index=data.split_list[0], x_cols=data.x_cols)
        xval, yval, _ = data.get_data_matrix(use_index=data.split_list[1], x_cols=data.x_cols)
        print('xtrn has shape', xtrn.shape)
        model.fit(xtrn,ytrn)
        score_dict = cal_scores(yval, model.predict(xval), header_str = 'val_')
        print('op4 score', score_dict)

        print('================= optimization 5: remove unncessary lag columns =================')
        importances = model.feature_importances_
        feat_imp = pd.DataFrame(importances, index=data.x_cols, columns=['importance']) 
        feat_imp = feat_imp.sort_values('importance',ascending=False).reset_index()

        # optimize 1 drop unuse cols 
        to_drop = feat_imp['index'].to_list()
        no_drop = ['Humidity(%)','Temperature(C)','Wind Speed(kmph)']  + [a for a in data.x_cols_org if 'fire' in a]
        for s in no_drop:
            to_drop.remove(s)
        to_drop.reverse()
        model, data.x_cols = reduce_cols(dataset=data,x_cols=data.x_cols,to_drop=to_drop,model=model,trn_i=0, val_i=1)
         
    else:
        data.lag_dict = lag_dict
        data.x_cols_org = x_cols_org
        data.data_org = data.data[ [data.monitor] + data.x_cols_org]
        data.build_lag(lag_range=np.arange(1, data.lag_dict['n_max'], data.lag_dict['step']), roll=True)
        data.x_cols = x_cols
        
 
    print('================= optimization 8: optimize for the best rf again =================')
    data.split_data(split_ratio=[0.45, 0.25, 0.3])
     
    print('x_cols in op7', data.x_cols)
    xtrn, ytrn, data.x_cols = data.get_data_matrix(use_index=data.split_list[0],x_cols=data.x_cols)
    xval, yval, _ = data.get_data_matrix(use_index=data.split_list[1],x_cols=data.x_cols)
    
    model = do_rf_search(xtrn, ytrn,cv_split='other')
    score_dict = cal_scores(yval, model.predict(xval), header_str ='val_')
    print(score_dict)

    # final split 
    data.split_data(split_ratio=[0.7, 0.3])
    xtrn, ytrn, data.x_cols = data.get_data_matrix(use_index=data.split_list[0],x_cols=data.x_cols)
    xtest, ytest, _ = data.get_data_matrix(use_index=data.split_list[1],x_cols=data.x_cols)
    model.fit(xtrn,ytrn)
    score_dict = cal_scores(ytest, model.predict(xtest), header_str ='test_')
    print('final score for test set', score_dict)
    

    pickle.dump(model, open(data.model_folder +f'{poll_name}_rf_model.pkl', 'wb'))

    # build feature of importance using build in rf
    try: 
        importances = model.feature_importances_
        feat_imp = pd.DataFrame(importances, index=data.x_cols, columns=['importance']) 
        feat_imp = feat_imp.sort_values('importance',ascending=False).reset_index()
        #show_fea_imp(feat_imp,filename=data.report_folder + f'{poll_name}_rf_fea_op2.png', title='rf feature of importance(default)')
    except:
        # custom feature of importance
        feat_imp = feat_importance(model,xtrn,ytrn,data.x_cols,n_iter=50)
        #show_fea_imp(feat_imp,filename=data.report_folder + f'{poll_name}_rf_fea_op2.png', title='rf feature of importance(shuffle)')

    # obtain feature of importance without lag 
    feat_imp['index'] = feat_imp['index'].str.split('_lag_', expand=True)[0]
    feat_imp = feat_imp.groupby('index').sum()
    feat_imp = feat_imp.sort_values('importance',ascending=False).reset_index()
    show_fea_imp(feat_imp,filename=data.report_folder + f'{poll_name}_rf_fea_op2_nolag.png', title='rf feature of importance(final)')

    poll_meta =  { 'x_cols_org': data.x_cols_org,
    'x_cols': data.x_cols,
    'fire_cols': fire_cols,
    'fire_dict': data.fire_dict,
    'lag_dict': data.lag_dict,
    'rf_score': score_dict,
    'rf_params': model.get_params()}

    model_meta = load_meta(data.model_folder + 'model_meta.json')
    model_meta[pollutant] = poll_meta
    save_meta(data.model_folder + 'model_meta.json', model_meta)

    data.save_()
     
    return data, model, poll_meta