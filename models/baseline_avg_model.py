import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

import mlflow
import mlflow.sklearn


# %%
class AvgModel(object):
    def __init__(
                self, 
                label,
                train_days):
        self.label = label 
        self.train_days = train_days
        return

    def averaging_by_weekday(self, df_store, column):
        """
        
        """
        result = {}
        for k in range(7):
            chunk = df_store[df_store['weekday'] == k]
            if self.train_days > 0:
                start_date = chunk['dateTime'].iloc[-1] - \
                    pd.Timedelta(days=self.train_days)
                chunk = chunk[chunk['dateTime'] >= start_date]
                # print(chunk)
            result[k] = chunk[column].mean()
        return result

    def fit(self, df):
        self.model = {}
        store_list = df['storeId'].unique()
        for store_id in store_list:
            df_store = df[df['storeId'] == store_id]
            self.model[store_id] = self.averaging_by_weekday(
                df_store, self.label)
        return

    def predict(self, df):
        def create_fn(model):
            return lambda row: model[row['storeId']][row['weekday']]
        pred = df.apply(create_fn(self.model), axis=1)
        return pred


def run(train_set, test_set, max_train_days, run_name='weekly_avgerage_model'):
    mlflow.set_experiment('')
    with mlflow.start_run(run_name=run_name):
        avg_model = AvgModel(pred_label='Inside',
                             max_train_days=max_train_days)
        avg_model.fit(train_set)
        inside_pred = avg_model.predict(test_set)

        df_result = test_set.copy()
        df_result['Inside_pred'] = inside_pred.values
        df_result.to_csv('./tmp_data/result.csv', index=False)

        df_pred = df_result[['storeId', 'dateTime', 'Inside', 'Inside_pred']]
        col_r2 = r2_score_by_store(df_pred).iloc[:, 1:].mean(axis=0)
        col_error_rate = error_rate_by_store(df_pred).iloc[:, 1:].mean(axis=0)

        for k in col_r2.index:
            mlflow.log_metric(k, col_r2[k])

        for k in col_error_rate.index:
            mlflow.log_metric(k, col_error_rate[k])

        print(col_r2)
        print(col_error_rate)

        mlflow.log_param('model', run_name)
        mlflow.log_param('train_days', max_train_days)
        mlflow.log_artifact('./tmp_data/result.csv')
    return


def main():
    df_stats = pd.read_csv('exp_data/1_raw_data/df_stats.csv',
                           parse_dates=['dateTime'])
    # df_stats = pd.read_csv('exp_data/1_raw_data/df_stats.csv')
    df_stores = pd.read_csv('exp_data/1_raw_data/df_stores.csv')

    df = df_stats.merge(df_stores, on='storeId')

    df = df[~df['dateTime'].between('2017-12-10', '2018-02-10')]
    df = df[~df['dateTime'].between('2018-12-10', '2019-02-10')]
    run_name = 'weekly_avgerage_model_remove_outliers'
    # run_name = 'weekly_avgerage_model'

    train_set, test_set = split_dataset(df, 28)

    for k in range(5, 150, 5):
        run(train_set, test_set, k, run_name)
    return


def show_results():
    filepath = ''
    df = pd.read_csv(filepath, parse_dates=['dateTime'])
    # df = pd.read_csv(filepath)
    col_r2 = r2_score_by_store(df[['storeId', 'dateTime', 'Inside', 'Inside_pred']])
    r2_score_by_store(
        df[['storeId', 'dateTime', 'Inside', 'Inside_pred']]).describe()
    error_rate_by_store(df[['storeId', 'dateTime', 'Inside', 'Inside_pred']])
    error_rate_by_store(
        df[['storeId', 'dateTime', 'Inside', 'Inside_pred']]).describe()

    col_r2 = col_r2.iloc[:, 1:].mean(axis=0)
    col_r2

    [k for k in col_r2.index]
    return


if __name__ == '__main__':
    main()
