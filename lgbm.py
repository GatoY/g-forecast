
#%%
# -------------- #

import pandas as pd
import os
import datetime
import math

import load_data
from util_log import logger
# from featureEng import (
# )
# from models import grid_search
from config import (
    RAW_DATA_DIR,
    LAG_DICT,
    SLIDING_DICT
)

from lightgbm import LGBMRegressor


#%%
# valid period. '2017-07-31' to '2017-08-16'
SPLIT_DATE = datetime.date(2017,7, 31)

def add_weekday(df):
    df['weekday'] = df['date'].dt.weekday
    return df

def add_month(df):
    df['month'] = df['date'].dt.add_month
    return df





def item_info(df):
    items_df = pd.read_csv(RAW_DATA_DIR+'items.csv')
    df = df.merge(items_df, on='item_nbr')
    return df

def store_info(df):
    stores_df = pd.read_csv(RAW_DATA_DIR+'stores.csv')
    df = df.merge(items_df, on='store_nbr')
    return df

def transaction_info(df):
    transactions_df = pd.read_csv(RAW_DATA_DIR+'transactions.csv')
    df = df.merge(transactions_df, on = 'store_nbr')
    return df

def oil_info(df):
    oil_df = pd.read_csv(RAW_DATA_DIR+'oil.csv',parse_dates=['date'])
    df = df.merge(oil_df, on = 'date')



#%%
def lagging(df,
            col,
            lag_len):
    """ Generate lagging features.
    Args:
        df: a DataFrame, contains all columns in cols_lag
        len_lag: Int, the length of lagging features
        cols_lag: List of String, the columns needs to generate lagging features 

    Returns:
        df: a DataFrame with lagging features, for example 'Sales_lag_1'.
    """

    for lag in lag_len:
        df[col + '_lag_' + str(lag)] = df[col].shift(periods=lag)
        
    return df


#%%
logger.debug('Get raw data')
train_df, test_df = load_data.main()


#%%

def filling_missing_value(df,
                          store_frame,
                          start_date=None,
                          end_date=None):
    """ filling missing value.
    Args:
        df: a DataFrame, contains all data with cols: 'Sales'
        store_frame: a DataFrame, contains only one store data with cols: 'Sales'

    Returns:
        df: a DataFrame. store_frame data filled
    """

    mask_length = 28
    if start_date == None and end_date == None:
        date_range = pd.date_range(
            store_frame.dateTime.min(), store_frame.dateTime.max())
    else:
        date_range = pd.date_range(
            start_date, end_date)

    storeId = store_frame.storeId.unique()[0]
    for date in date_range:
        t_d = store_frame[store_frame['dateTime'] == date]

    return df
train_df = train_df[train_df['store_nbr'] == 1]

train_df = train_df[train_df['item_nbr']==103520]

#%%
def lagging(df,
            col,
            lag_len):
    """ Generate lagging features.
    Args:
        df: a DataFrame, contains all columns in cols_lag
        len_lag: Int, the length of lagging features
        cols_lag: List of String, the columns needs to generate lagging features 

    Returns:
        df: a DataFrame with lagging features, for example 'Sales_lag_1'.
    """

    for lag in lag_len:
        df[col + '_lag_' + str(lag)] = df[col].shift(periods=lag)
        
    return df


#%%
for col in LAG_DICT:
    lag_len = LAG_DICT[col]
    train_df = lagging(train_df, col, lag_len)


#%%
def sliding_window(df,
                   sliding_len,
                   col
                   ):
    """ Generate sliding window features.
    Args:
        df: a DataFrame, contains all columns in cols_lag
        len_sliding_window: Int, the length of sliding window, >2 
        cols_sliding: List of String, the columns needs to generate sliding window features 

    Returns:
        df: a DataFrame with sliding window features, for example 'unit_sales_mean_2'.
    """

    def roll(data, roll_size, col):
        roll = data[col].shift(1).rolling(roll_size)
        data[col + '_mean_' + str(roll_size)] = roll.mean()
        data[col + '_std_' + str(roll_size)] = roll.std()
#         data[col + '_max_' + str(roll_size)] = roll.max()
#         data[col + '_min_' + str(roll_size)] = roll.min()
        return data

    for roll_size in sliding_len:
        df = roll(df, roll_size, col)
    df = df.dropna()
    return df


#%%
for col in SLIDING_DICT:
    sliding_len = SLIDING_DICT[col]
    train_df = sliding_window(train_df,
                             sliding_len,
                             col) 


#%%
train_df = train_df.dropna()


#%%
def train_valid_split(df):
    train_df = df[df['date']<SPLIT_DATE]
    valid_df = df[df['date']>=SPLIT_DATE]
    return train_df, valid_df


#%%
train, valid = train_valid_split(train_df)


#%%
def feature_label(df):

    features = df.drop(columns = ['date', 'store_nbr', 'item_nbr', 'unit_sales']).values
    labels = df['unit_sales'].values
    return features, labels


#%%
train_X, train_y = feature_label(train)
valid_X, valid_y = feature_label(valid)


#%%
model =LGBMRegressor(
                colsample_bytree = 0.8,
                learning_rate= 0.08,  # [0.08, 0.1]
                max_depth=6,
                num_leaves= 30,
                min_child_weight= 5,
                n_estimators= 200,  # [300, 500],
#                 nthread = 4,
                seed= 1337,
                silent= 1,
                subsample= 0.8
                )


model.fit(train_X, train_y)

print('score {}'.format(model.score(valid_X, valid_y)))