# -------------- #

import pymysql
import pandas as pd
import os
import datetime
import math

import load_data
from util_log import logger
from featureEng import (
    lagging,
    sliding_window,
    filling_missing_value,
    add_influence,
    add_weather_influence,
    filter_peak_period,
    gen_datasets,
    train_valid_split
)
from models import grid_search
from config import (
    STOREID_LIST,

    INFLUENCE_SHEET,
    INFLUENCE_MIN,
    INFLUENCE_MAX,

    PREDICT_START_DATE,
    PREDICT_LENGTH,

    LAG_LENGTH,
    LAG_COLS,
    SLIDING_WINDOW_LENGTH,
    SLIDING_COLS,
    MODEL_LIST,

    PREIDCT_DATA_DIR,
    PREDICT_DATASET_DIR,
    PREDICT_TMP_DIR,
    PREDICT_MODEL_DIR,

    PEAK_PERIOD
)


def main():
    # Valid period,  [VALID_START_DATE, VALID_END_DATE)
    VALID_END_DATE = datetime.datetime.strptime(PREDICT_START_DATE, '%Y-%m-%d')
    VALID_START_DATE = VALID_END_DATE-datetime.timedelta(days=PREDICT_LENGTH)

    logger.debug('Get foot traffic data')
    # fetch raw data
    sql_data = load_data(VALID_END_DATE)
    if sql_data['dateTime'].max() != VALID_END_DATE:
        logger.debug('data integrity check failed!')
    sql_data.to_csv(PREDICT_DATASET_DIR + 'raw_data.csv', index=False)

    storeId_list = sql_data.storeId.unique()
    logger.debug('Filling missing value, lag, sliding window')
    for storeId, count in zip(storeId_list, range(len(storeId_list))):

        store_frame = sql_data[sql_data['storeId'] == storeId]
        store_frame = store_frame.sort_values(by=['dateTime'])
        store_frame = store_frame.reset_index(drop=True)

        # filling missing value
        store_frame = filling_missing_value(
            store_frame,
            store_frame[store_frame['storeId'] == storeId])
        store_frame = store_frame.sort_values(by=['storeId', 'dateTime'])
        store_frame = store_frame.reset_index(drop=True)

        # lag features
        store_frame = lagging(
            store_frame, lag_len=LAG_LENGTH, lag_cols=LAG_COLS)

        # sliding features
        store_frame = sliding_window(store_frame,
                                     sliding_window_len=SLIDING_WINDOW_LENGTH,
                                     sliding_cols=SLIDING_COLS
                                     )

        if count == 0:
            df = store_frame
        else:
            df = pd.concat([df, store_frame])

    logger.debug('filter peak periods')
    # filter peak periods
    df = filter_peak_period(df,
                            PREDICT_START_DATE,
                            PEAK_PERIOD)

    logger.debug('add influence')
    # add influence
    df = add_influence(df,
                       influence_file=INFLUENCE_SHEET,
                       influence_threshold_min=INFLUENCE_MIN,
                       influence_threshold_max=INFLUENCE_MAX
                       )


    # add weather influence
    df = add_weather_influence(df,
                               influence_path=WEATHER_DATA_PATH,
                               influence_threshold_min=WEATHER_INFLUENCE_MIN,
                               influence_threshold_max = WEATHER_INFLUENCE_MAX)


    logger.debug('sort and clean')
    # sort and clean
    df = df.dropna()
    df = df.sort_values(by=['storeId', 'dateTime'])
    df = df.drop_duplicates(subset=['storeId', 'dateTime'], keep='first')
    df = df.reset_index(drop=True)
    df.dateTime = pd.to_datetime(df.dateTime)
    # record intermedia file
    df.to_csv(PREDICT_DATASET_DIR + 'datasets.csv', index=False)

    logger.debug('gen train datasets')
    # gen train datasets
    train_frame_dict = gen_datasets(
        df,
        PREDICT_DATASET_DIR,
        PREDICT_LENGTH,
        changing_cols=['month', 'weekday', 'Inside', 'influence'])

    logger.debug('train valid split')
    # train valid split
    train_valid_dict = train_valid_split(
        train_frame_dict,
        VALID_START_DATE,
        PREDICT_TMP_DIR,
        PREDICT_LENGTH)

    logger.debug('grid search training')
    # grid search
    grid_search(
        VALID_START_DATE,
        PREDICT_LENGTH,
        PREDICT_TMP_DIR,
        PREDICT_MODEL_DIR,
        train_valid_dict,
        model_list=MODEL_LIST,
        verbose=2
    )


if __name__ == '__main__':
    main()
