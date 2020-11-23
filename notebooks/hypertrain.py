# -*- coding: utf-8 -*-
 
import os
import sys
import logging

if __package__: 

    from ..src.models.train_model import train_hyper_search

else:
    # run as a script, use absolute import
    _i = os.path.dirname(os.path.dirname(os.path.abspath('..')))
    if _i not in sys.path:
        sys.path.insert(0, _i)
    from src.models.train_model import train_hyper_search

    
if __name__ == '__main__':

    #main(main_folder='../../data/', cdc_data=True, build_json=True)
    print(sys.argv)

    data_folder = os.path.abspath('../data/') + '/'

    print(data_folder)

