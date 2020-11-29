# -*- coding: utf-8 -*-
import os
import sys
import logging

if __package__: 

    from ..src.features.landuse import label_landuse_fire

else:
    # run as a script, use absolute import
    _i = os.path.abspath('..')
   
    if _i not in sys.path:
        sys.path.insert(0, _i)
    from src.features.landuse import label_landuse_fire


if __name__ == '__main__':

    function_name =  sys.argv[1]

    print(function_name)

    if function_name == 'label_landuse_fire':

        city = sys.argv[2]
        label_landuse_fire(city=city)
     