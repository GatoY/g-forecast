import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing, Holt
from statsmodels.tsa.arima_model import ARIMA

import mlflow
import mlflow.sklearn


# https://medium.com/datadriveninvestor/how-to-build-exponential-smoothing-models-using-python-simple-exponential-smoothing-holt-and-da371189e1a1
# https://www.statsmodels.org/dev/examples/notebooks/generated/exponential_smoothing.html
# https://machinelearningmastery.com/time-series-forecasting-methods-in-python-cheat-sheet/


class SESModel(object):
    def __init__(self, pred_label='Inside', max_train_days=-1):
        self.pred_label = pred_label
        self.max_train_days = max_train_days
        self.model = {}
        return

    def averaging_by_week(self, df_store, column):
        """
        TODO: add multi columns
        """
        result = {}
        for k in range(7):
            chunk = df_store[df_store['weekday'] == k]
            if self.max_train_days > 0:
                start_date = chunk['dateTime'].iloc[-1] - \
                    pd.Timedelta(days=self.max_train_days)
                chunk = chunk[chunk['dateTime'] >= start_date]
                # print(chunk)
            result[k] = chunk[column].mean()
        return result

    def fit(self, df):
        stores = df['storeId'].unique()
        for store_id in stores:
            df_store = df[df['storeId'] == store_id]
            self.model[store_id] = self.averaging_by_week(
                df_store, self.pred_label)
        return

    def predict(self, df):
        def create_fn(model):
            return lambda row: model[row['storeId']][row['weekday']]
        pred = df.apply(create_fn(self.model), axis=1)
        return pred


def run(train_set, test_set, max_train_days):
    mlflow.set_experiment('')
    with mlflow.start_run():
        avg_model = SESModel(pred_label='Inside',
                             max_train_days=max_train_days)
        avg_model.fit(train_set)
        inside_pred = avg_model.predict(test_set)
        r2 = r2_score(test_set['Inside'], inside_pred)
        print('r2: {}'.format(r2))

        mlflow.log_param('train_days', max_train_days)
        mlflow.log_metric('r2', r2)
    return


def test(train_set, test_set):
    # %%
    # r2s = []
    y_true = []
    y_pred = []
    for store_id in train_set['storeId'].unique():
        train = train_set[train_set['storeId'] == store_id].iloc[:, :6]
        test = test_set[test_set['storeId'] == store_id].iloc[:, :6]

        # train = train[train['weekday'] == 0]
        # test = test[test['weekday'] == 0]

        train = pd.Series(train['Inside'].values, train['dateTime'].values)
        test = pd.Series(test['Inside'].values, test['dateTime'].values)

        # del fcast1
        # fcast1 = SimpleExpSmoothing(train).fit()
        # fcast1 = ExponentialSmoothing(train, seasonal='add', trend='add', seasonal_periods=7).fit()
        # fcast1 = Holt(train).fit()
        # pred = fcast1.forecast(test.shape[0])
        fcast1 = ARIMA(train, order=(1, 1, 1)).fit()
        pred = fcast1.predict(train.shape[0], train.shape[0] + test.shape[0] - 1, typ='levels')
        # pred = fcast1.predict(1, 10, typ='levels')
        # fcast1.predict(train.shape[0], train.shape[0], typ='levels')

        y_true.append(test.values)
        y_pred.append(pred.values)
        # print(test)
        # print(y_pred)
        # r2 = r2_score(test.values[:10], y_pred.values[:10])


    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    print(y_true.shape)
    print(y_pred.shape)
# %%
    df_result = pd.DataFrame({
        'storeId': test_set['storeId'].values,
        'Inside': y_true,
        'Inside_pred': y_pred,
    })

    # df_result = pd.read_csv('/Users/yinchuandong/PycharmProjects/ka/experiments/mlruns/1/3393e54394004e7497030f299258a955/artifacts/result.csv')
    r2_list = []
    for store_id in df_result['storeId'].unique():
        df_store = df_result[df_result['storeId'] == store_id]
        r2 = r2_score(df_store['Inside'], df_store['Inside_pred'])
        r2_list.append((store_id, r2))


    df_result2 = pd.DataFrame(r2_list, columns=['storeId', 'r2'])

    r2_score(df_result['Inside'], df_result['Inside_pred'])
    df_result2
    df_result2.describe()
# %%

    return


def main():
    df_stats = pd.read_csv('exp_data/1_raw_data/df_stats.csv',
                           parse_dates=['dateTime'])
    df_stores = pd.read_csv('exp_data/1_raw_data/df_stores.csv')

    df = df_stats.merge(df_stores, on='storeId')
    train_set, test_set = split_dataset(df, 28)

    test(train_set, test_set)
    return


if __name__ == '__main__':
    main()
