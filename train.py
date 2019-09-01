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
    FEATUER_DIR,
    LAG_COLS
)

import run_model

# valid period. '2017-07-31' to '2017-08-16'
SPLIT_DATE = datetime.date(2017,7, 31)

def add_weekday(df):
    df['weekday'] = df['date'].dt.weekday
    return df

def add_month(df):
    df['month'] = df['date'].dt.add_month
    return df


def train_valid_split(df):
    train_df = df[df['date']<SPLIT_DATE]
    valid_df = df[df['date']>=SPLIT_DATE]
    return train_df, valid_df

def lagging(df,
            lag_len=50,
            lag_cols=['unit_sales']
            ):
    """ Generate lagging features.
    Args:
        df: a DataFrame, contains all columns in cols_lag
        len_lag: Int, the length of lagging features
        cols_lag: List of String, the columns needs to generate lagging features 

    Returns:
        df: a DataFrame with lagging features, for example 'Sales_lag_1'.
    """

    # lag
    for lag in range(1, lag_len):
        _df_shift = df[lag_cols].shift(periods=lag)
        for col in lag_cols:
            df[col + '_lag_' + str(lag)] = _df_shift[col]
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


def sliding_window(df,
                   sliding_window_len=50,
                   sliding_cols=['unit_sales']
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
        data[col + '_max_' + str(roll_size)] = roll.max()
        data[col + '_min_' + str(roll_size)] = roll.min()
        return data


    for roll_size in range(2, sliding_window_len): 
        for col in sliding_cols:
            df = roll(df, roll_size, col)

    df = df.dropna()
    return df



def main(
    model_used=None,
    feature_used=None):
    

    logger.debug('Get raw data')
    train_df, test_df = load_data.main()
    
    # store_list = train_df.store_nbr.unique()[2:7]
    # train_df = train_df[train_df['store_nbr'].isin(store_list)]
    train_df = train_df.dropna()

    train_df = train_df[train_df['date']>=datetime.date(2017, 7, 1)]
    logger.debug('Train valid split')
    train_df, valid_df = train_valid_split(train_df)

    logger.debug('Add weekday')
    train_df = add_weekday(train_df)
    valid_df = add_weekday(valid_df)

    logger.debug('Train model')
    run_model.main(
        model_used,
        train_df,
        valid_df
        )

    # storeId_list = train_df['store_nbr'].unique()
    # # logger.debug('Filling missing value, lag, sliding window')
    # for storeId in storeId_list:
    #     store_frame = train_df[train_df['store_nbr'] == storeId]
    #     item_list = store_frame.item_nbr.unique()
    #     logger.debug(storeId)
    #     count = 0

    #     for itemId in item_list:
    #         item_frame = store_frame[store_frame['item_nbr']==itemId]
    #         item_frame = item_frame.sort_values(by=['date'])
    #         item_frame = item_frame.reset_index(drop=True)
    #         item_frame = lagging(
    #             item_frame,
    #             lag_len=LAG_LENGTH, 
    #             lag_cols=LAG_COLS)

    #         if count == 0:
    #             df = item_frame
    #         else:
    #             df = pd.concat([df, item_frame])
        
    #         count += 1

    #     df.to_csv(FEATUER_DIR+'{}_lag_sales.csv'.format(storeId), index=False)
    #     del df

        
        
    # logger.debug('filter peak periods')
    # # filter peak periods
    # df = filter_peak_period(df,
    #                         PREDICT_START_DATE,
    #                         PEAK_PERIOD)

    # logger.debug('add influence')
    # # add influence
    # df = add_influence(df,
    #                    influence_file=INFLUENCE_SHEET,
    #                    influence_threshold_min=INFLUENCE_MIN,
    #                    influence_threshold_max=INFLUENCE_MAX
    #                    )

    # # add weather influence
    # df = add_weather_influence(df,
    #                            influence_path=WEATHER_DATA_PATH,
    #                            influence_threshold_min=WEATHER_INFLUENCE_MIN,
    #                            influence_threshold_max = WEATHER_INFLUENCE_MAX)

    # logger.debug('sort and clean')
    # # sort and clean
    # df = df.dropna()
    # df = df.sort_values(by=['storeId', 'dateTime'])
    # df = df.drop_duplicates(subset=['storeId', 'dateTime'], keep='first')
    # df = df.reset_index(drop=True)
    # df.dateTime = pd.to_datetime(df.dateTime)
    # # record intermedia file
    # df.to_csv(PREDICT_DATASET_DIR + 'datasets.csv', index=False)


    

    # logger.debug('grid search training')

    # grid_search(
    #     VALID_START_DATE,
    #     PREDICT_LENGTH,
    #     PREDICT_TMP_DIR,
    #     PREDICT_MODEL_DIR,
    #     train_valid_dict,
    #     model_list=MODEL_LIST,
    #     verbose=2
    # )


if __name__ == '__main__':
    main()
