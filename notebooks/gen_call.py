# -*- coding: utf-8 -*-
import os
import sys
import logging

if __package__: 

    from ..src.features.landuse import label_landuse_fire
    from ..src.features.dataset import Dataset

else:
    # run as a script, use absolute import
    _i = os.path.abspath('..')
   
    if _i not in sys.path:
        sys.path.insert(0, _i)
    from src.features.landuse import label_landuse_fire
    from src.features.dataset import Dataset


if __name__ == '__main__':

    function_name =  sys.argv[1]

    print(function_name)

    if function_name == 'label_landuse_fire':

        city = sys.argv[2]
        label_landuse_fire(city=city)
     
    elif function_name == 'builddata':

        city = sys.argv[2]
        if len(sys.argv)>=4:
            instr = sys.argv[3]
        else:
            instr = 'MODIS'
        
        print(f'Building dataset of {city}')
        dataset = Dataset(city_name=city)
        dataset.build_fire(instr=instr)
        dataset.build_all_data(build_fire=False, build_holiday=True)
