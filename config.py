import yaml
import json
import os
import numpy as np


LAG_DICT = {'unit_sales': [1,2,5,7,14,21,28,35],
            'onpromotion': [14, 60]}

SLIDING_DICT = {'unit_sales': [3, 7, 14, 30, 60]}

# initialise dirs
RAW_DATA_DIR = 'datasets/'
FEATURE_DIR = 'feature_sets/'


if not os.path.exists(RAW_DATA_DIR):
    os.system('mkdir '+RAW_DATA_DIR)

# PEAK_PERIOD = np.reshape(np.array(CONFIG['peak_period']),[-1,2]).tolist()
