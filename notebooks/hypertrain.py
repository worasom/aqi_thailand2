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
    from src.models.train_model import train_hyper_search

    
if __name__ == '__main__':

    #main(main_folder='../../data/', cdc_data=True, build_json=True)

    city = sys.argv[1]
    pollutant = sys.argv[2]
    if len(sys.argv) >= 4:
        n_jobs = int(sys.argv[3])
    else:
        n_jobs = -1

    data_folder = os.path.abspath('../data/') + '/'

    train_hyper_search(city=city, pollutant=pollutant, n_jobs =n_jobs)

