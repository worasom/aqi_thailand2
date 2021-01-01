# -*- coding: utf-8 -*-
 
import os
import sys
import logging

if __package__: 

    from ..src.features.dataset import Dataset

else:
    # run as a script, use absolute import
    _i = os.path.abspath('..')
   
    if _i not in sys.path:
        sys.path.insert(0, _i)
    from src.features.dataset import Dataset


if __name__ == '__main__':

    #main(main_folder='../../data/', cdc_data=True, build_json=True)
    print(sys.argv)

    city = sys.argv[1]
     
    
    dataset = Dataset(city_name=city)
    dataset.build_fire()
    dataset.build_all_data(build_fire=False, build_holiday=True)