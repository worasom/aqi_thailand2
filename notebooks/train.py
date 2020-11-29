# -*- coding: utf-8 -*-
import os
import sys
import logging

if __package__: 

    from ..src.models.train_model import train_hyper_search

else:
    # run as a script, use absolute import
    _i = os.path.abspath('..')
   
    if _i not in sys.path:
        sys.path.insert(0, _i)
    from src.models.train_model import train_city_s1

if __name__ == '__main__':

    city = sys.argv[1]
    pollutant = sys.argv[2]
    if len(sys.argv) >= 4:
        n_jobs = int(sys.argv[3])
    else:
        n_jobs = -1
    
    data_folder = os.path.abspath('../data/') + '/'
    
    dataset, model, trainer = train_city_s1(city=city, pollutant= pollutant, n_jobs=n_jobs,  search_wind_damp=True, choose_cat_hour=True)