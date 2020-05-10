# -*- coding: utf-8 -*-
import sys
from pathlib import Path
import re
import os
from tqdm import tqdm
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
import math


import requests
import wget
from bs4 import BeautifulSoup
from selenium import webdriver


import time
from datetime import datetime, date,  timedelta


from bokeh.io import output_file, output_notebook, show, reset_output,export_png
from bokeh.models import (
   GMapPlot, GMapOptions, ColumnDataSource, Circle, LogColorMapper, BasicTicker, ColorBar,
    DataRange1d, PanTool, WheelZoomTool, BoxSelectTool, CategoricalColorMapper, Slider, DateRangeSlider,
    DateSlider, SingleIntervalTicker, LinearAxis,Legend, LegendItem
)
from bokeh.models.mappers import ColorMapper, LinearColorMapper
from bokeh.palettes import Viridis5
from bokeh.plotting import figure, show, output_file 
from bokeh.layouts import widgetbox,row, column, gridplot
from mpl_toolkits.basemap import Basemap
from bokeh.tile_providers import get_provider, Vendors

