# -------------- #
# v2.0

import pandas as pd
import numpy as np
import datetime
import pickle
import math
import pymysql

from sklearn.preprocessing import OneHotEncoder
import lime
import lime.lime_tabular

from util_log import logger
explainer = lime.lime_tabular.LimeTabularExplainer(train, feature_names=iris.feature_names, class_names=iris.target_names, discretize_continuous=True)

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
)

from featureEng import (
    lagging,
    sliding_window,
    filling_missing_value,
    add_influence,
    gen_datasets,
    train_valid_split
)


def fetch_data(start_date, end_date, STOREID_LIST):
    logger.debug('Start date: %s and end date: %s' % (start_date, end_date))
    logger.debug('Get foot traffic data')
    cnx = pymysql.connect(user=MYSQL_USER, password=MYSQL_PASSWORD,
                          host=MYSQL_HOST,
                          database=MYSQL_DB)
    query = ("""
    SELECT 
    Stats.dateTime AS dateTime,
    MONTH(Stats.dateTime) AS month,
    WEEKDAY(Stats.dateTime) AS weekday,
    Stats.outsideOpportunity AS Outside,
    Stats.totalVisitors AS Inside,
    Stats.sales AS Sales,
    Stats.storeId
    FROM
    Kepler.Stats
        INNER JOIN
    Kepler.Store ON Stats.storeId = Store.storeId
        INNER JOIN
    Kepler.StoresLocation ON Store.storeId = StoresLocation.storeId
    WHERE
    Stats.Duration = 2
        AND Stats.totalVisitors != 0
        AND Sales > 0
        AND dateTime > '%s'
        AND dateTime <= '%s' AND Store.storeId in %s
            """ % (str(start_date), str(end_date), STOREID_LIST))

    sql_data = pd.read_sql(query, cnx)
    return sql_data


def gen_future_sets(data,
                    end_date,
                    PREDICT_LENGTH,
                    influence_file='specialDaysInfluence v1.1.xlsx',
                    influence_threshold_min=0,
                    influence_threshold_max=0
                    ):
    # Get special days data
    logger.debug('Get special days data')
    influence = pd.read_excel(influence_file)[['date', 'influence']]
    influence['date'] = pd.to_datetime(influence['date'])
    influence = influence[(influence.influence >= influence_threshold_max) | (
        influence.influence <= influence_threshold_min)]

    # generate changing columns in the future
    future_sets = pd.DataFrame(
        columns=['month', 'weekday', 'influence', 'dateTime'])

    for date_time in pd.date_range(end_date,
                                   end_date + datetime.timedelta(days=PREDICT_LENGTH-1)):

        influence_value = influence.loc[influence['date'] == date_time, 'influence'].values[
            0] if date_time in influence.date.unique() else 0
        date_time = pd.DatetimeIndex([date_time])[0]
        weekday = date_time.weekday()
        month = date_time.month
        future_sets.loc[len(future_sets)] = [
            month, weekday, influence_value, date_time]
    future_sets.month = future_sets.month.astype(int)
    future_sets.weekday = future_sets.weekday.astype(int)
    future_sets.influence = future_sets.influence.astype(float)
    
    # generate all features in the future
    data.dateTime = pd.to_datetime(data.dateTime)
    count = 0
    for storeId in data.storeId.unique():
        t = data[data['storeId'] == storeId]
        t = t.merge(future_sets, on=[
                    'month', 'weekday', 'dateTime'], how='right')
        t = t.fillna(method='ffill')

        
        if count == 0:
            future_frame = t
        else:
            future_frame = pd.concat([future_frame, t])
        count += 1
    future_frame = future_frame.reset_index(drop=True)

    return future_frame


def get_feature(d):
    month_weekday_data = OneHotEncoder(categories=[range(1, 13), range(7)]).fit_transform(
        d[['month', 'weekday']]).toarray()
    lagging_data = d.drop(
        columns=['month', 'weekday', 'storeId', 'Inside', 'dateTime', 'Outside', 'Sales']).values
    features = np.hstack((month_weekday_data, lagging_data))
    return features


def main():

    end_date = datetime.datetime.strptime(PREDICT_START_DATE, '%Y-%m-%d')
    start_date = end_date - \
        datetime.timedelta(days=max(LAG_LENGTH, SLIDING_WINDOW_LENGTH))

    # Raw_data
    logger.debug('fetch data')
    sql_data = fetch_data(
        start_date-datetime.timedelta(days=30), end_date, STOREID_LIST)

    logger.debug('generate future frame')
    logger.debug('Filling missing value, lag, sliding window')
    storeId_list = sql_data.storeId.unique()
    for storeId, count in zip(storeId_list, range(len(storeId_list))):

        store_frame = sql_data[sql_data['storeId'] == storeId]
        store_frame = store_frame.sort_values(by=['dateTime'])
        store_frame = store_frame.reset_index(drop=True)

        # filling missing value
        store_frame = filling_missing_value(store_frame,
                                            store_frame[store_frame['storeId']
                                                        == storeId],
                                            start_date,
                                            end_date)
        store_frame = store_frame.sort_values(by=['storeId', 'dateTime'])

        store_frame = store_frame[store_frame['dateTime'] > start_date]
        store_frame = store_frame.reset_index(drop=True)

        # lag features
        store_frame = lagging(store_frame,
                              lag_len=LAG_LENGTH,
                              lag_cols=LAG_COLS)

        # sliding features
        store_frame = sliding_window(store_frame,
                                     sliding_window_len=SLIDING_WINDOW_LENGTH,
                                     sliding_cols=SLIDING_COLS
                                     )

        if count == 0:
            df = store_frame
        else:
            df = pd.concat([df, store_frame])

    future_frame = gen_future_sets(df,
                                   end_date,
                                   PREDICT_LENGTH,
                                   influence_file=INFLUENCE_SHEET,
                                   influence_threshold_min=INFLUENCE_MIN,
                                   influence_threshold_max=INFLUENCE_MAX
                                   ).reset_index(drop=True)
    # print(future_frame)
    future_frame.to_csv(PREDICT_DATASET_DIR +
                        'predict_datasets.csv', index=False)

    logger.debug('load pre-trained models')
    # load pre trained models
    model_list = []
    for i in range(PREDICT_LENGTH):

        model_list.append(pickle.load(
            open(PREDICT_MODEL_DIR + 'best_model_' + str(i) + '.pkl', 'rb')))
    print(PREDICT_MODEL_DIR)
    logger.debug('predict')
    # predict
    store_list = future_frame.storeId.unique()
    for i in range(PREDICT_LENGTH):
        date_time = end_date + datetime.timedelta(days=i)
        logger.debug('predict %s' % date_time)
        for storeId in store_list:
            t = future_frame[(future_frame['storeId'] == storeId) & (
                future_frame['dateTime'] == date_time)]
            features = get_feature(t)
            inside = model_list[i].predict(features)[0]
            future_frame.loc[(future_frame['storeId'] == storeId) & (
                future_frame['dateTime'] == date_time), 'Inside'] = inside

    future_frame[['storeId', 'dateTime', 'Inside']].to_csv(
        PREDICT_DATASET_DIR + 'predict_results.csv', index=False)


if __name__ == '__main__':
    main()
