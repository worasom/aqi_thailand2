# -*- coding: utf-8 -*-
import os
import sys

if __package__: 

    
    from ..src.imports import *
    from ..src.gen_functions import *
    #from ..src.features.dataset import Dataset
    from ..src.models.train_model import *
     
else:
    # run as a script, use absolute import
    _i = os.path.abspath('..')
   
    if _i not in sys.path:
        sys.path.insert(0, _i)
    # -*- coding: utf-8 -*-
    from src.imports import *
    from src.gen_functions import *
    #from src.features.dataset import Dataset
    from src.models.train_model import *
     
def train_prov(prov, use_impute=False, default_meta=False):
    # if use_impute:
        
    #     dataset, model, trainer = train_city_s1(city=prov, pollutant= 'PM2.5', instr='MODIS', add_weight=True, 
    #         to_op_fire_zone=True, op_lag=True, choose_cat_hour=False, 
    #         choose_cat_month=False, use_impute=use_impute, default_meta=default_meta, n_jobs=3,
    #         model_folder='../models_w_impute/', report_folder='../reports_w_impute/')
         
    # else:
        
    dataset, model, trainer = train_city_s1(city=prov, pollutant= 'PM2.5', instr='MODIS', add_weight=True, 
                                            to_op_fire_zone=True, op_lag=True, choose_cat_hour=False, 
                                            choose_cat_month=False, use_impute=use_impute, default_meta=default_meta, n_jobs=3,
                                            model_folder='../models/', report_folder='../reports/')
         
def main():

    prov_list = [
    #'Rayong', 
    # 'Khon Kaen',
    # 'Songkhla',
    # 'Samut Sakhon',
    # 'Nan',
    #'Sa Kaeo',
    #### 'Loei',
    #'Ratchaburi',
    
    #'Chon Buri', 
    #'Saraburi',
    #'Bangkok',
    #'Chiang Mai'
    ]

    prov_list = [
    #     'Phuket',
    # 'Pathum Thani',
    # 'Samut Prakan',
    # 'Nonthaburi',
    # 'Phra Nakhon Si Ayutthaya',
    # 'Lampang',
    # 'Narathiwat',
    # 'Yala',
    # 'Mae Hong Son',
    # 'Chiang Rai',
    # 'Lamphun',
    # 'Phrae',
   
    #'Phayao',
    # 'Nakhon Sawan',
    # 'Surat Thani',
    # 'Chachoengsao',
    # 'Prachin Buri',
    # 'Tak',
    # 'Kanchanaburi',
    # 'Satun',
    # 'Nakhon Pathom',
    # 'Sukhothai',
    'Si Sa Ket',
    'Amnat Charoen',
    'Mukdahan',
    'Bueng Kan',
    'Nong Bua Lam Phu',
    'Yasothon',
    'Nakhon Nayok',
    'Sakon Nakhon',
    'Ubon Ratchathani',
    'Nong Khai',
    'Samut Songkhram',
    'Chai Nat',
    'Phetchaburi',
    'Ang Thong',
    'Nakhon Si Thammarat',
    'Nakhon Phanom',
    'Phitsanulok',
    'Trat',
    'Suphan Buri',
    'Nakhon Ratchasima']

    # for prov in prov_list:
    #     dataset = Dataset(prov)
    #     dataset.build_all_data(build_fire=True)
    #     try:
    #         dataset = Dataset(prov)
    #         dataset.build_all_data(build_fire=True)
    #     except:
    #         print(' faild to build ', prov)

    #dataset = Dataset('Nan')
    #dataset.build_all_data(build_fire=True)

    prov_list = ['Mukdahan',
    'Bueng Kan',
    'Nakhon Nayok',
    'Sakon Nakhon',
    'Ubon Ratchathani',
    'Nong Khai',
    'Samut Songkhram',
    'Nakhon Si Thammarat',
    'Nakhon Phanom',
    'Phitsanulok',
    'Trat',
    'Suphan Buri', 
    'Nakhon Ratchasima'
    ]
    

    bad_prov_list = ['Si Sa Ket',
    'Amnat Charoen',
    'Nong Bua Lam Phu',
    'Yasothon',
    'Chai Nat',
    'Phetchaburi',
    'Ang Thong']

    for prov in prov_list:
        print('Train Data for ', prov)
        dataset = Dataset(prov)
        dataset.build_all_data(build_fire=True)
        try:
            train_prov(prov=prov, use_impute=1, default_meta=True)
        except:
            bad_prov_list.append(prov)

    print('bad province ', bad_prov_list)


if __name__ == '__main__':

    main()