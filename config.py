import yaml
import json
import os
import numpy as np

def load_config(filepath):
    with open(filepath) as f:
        config = yaml.load(f)

    print('-------------------------------------------------------------')
    print('configurations: ')
    print(json.dumps(config['config'], indent=2))
    print('-------------------------------------------------------------')

    return config['config']


CONFIG = load_config('config.yaml')

STOREID_LIST = tuple(CONFIG['storeId'])

INFLUENCE_SHEET = CONFIG['influence_sheet']
INFLUENCE_MIN = CONFIG['influence_min']
INFLUENCE_MAX = CONFIG['influence_max']


PREDICT_START_DATE = CONFIG['predict_start_date']
PREDICT_LENGTH = CONFIG['predict_length']
LAG_LENGTH = CONFIG['lag_length']

LAG_COLS = CONFIG['lag_cols']

SLIDING_WINDOW_LENGTH = CONFIG['sliding_window_length']
SLIDING_COLS = CONFIG['sliding_cols']

MODEL_LIST = CONFIG['model_list']


# initialise dirs
PREIDCT_DATA_DIR = 'datasets/'
PREDICT_DATASET_DIR = 'datasets/{}/'.format(PREDICT_START_DATE)
PREDICT_TMP_DIR = 'datasets/{}/train_test/'.format(PREDICT_START_DATE)
PREDICT_MODEL_DIR = 'datasets/{}/models/'.format(PREDICT_START_DATE)

if not os.path.exists(PREIDCT_DATA_DIR):
    os.system('mkdir '+PREIDCT_DATA_DIR)
if not os.path.exists(PREDICT_DATASET_DIR):
    os.system('mkdir ' + PREDICT_DATASET_DIR)
if not os.path.exists(PREDICT_TMP_DIR):
    os.system('mkdir ' + PREDICT_TMP_DIR)
if not os.path.exists(PREDICT_MODEL_DIR):
    os.system('mkdir ' + PREDICT_MODEL_DIR)

PEAK_PERIOD = np.reshape(np.array(CONFIG['peak_period']),[-1,2]).tolist()
