import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from fbprophet import Prophet


import mlflow
import mlflow.sklearn

# %%



def test(train_set, test_set):
    # %%
    # r2s = []
    y_true = []
    y_pred = []
    for store_id in train_set['storeId'].unique():
        # store_id = 721
        train = train_set[train_set['storeId'] == store_id].iloc[:, :6]
        test = test_set[test_set['storeId'] == store_id].iloc[:, :6]

        # train = train[train['weekday'] == 0]
        # test = test[test['weekday'] == 0]

        train = pd.DataFrame({'ds': train['dateTime'], 'y': train['Inside']})
        test = pd.DataFrame({'ds': test['dateTime'], 'y': test['Inside']})

        fcast1 = Prophet(weekly_seasonality=True, daily_seasonality=False, yearly_seasonality=True)
        fcast1.fit(train)
        future = fcast1.make_future_dataframe(periods=test.shape[0])

        pred = fcast1.predict(future)

        # pred.tail()
        # r2_score(train['y'], pred['yhat'][:-27])
        # r2_score(test['y'], pred['yhat'][-27:])
        y_true.append(test['y'].values)
        y_pred.append(pred['yhat'][-test.shape[0]:].values)
        # break
        # print(test)
        # print(y_pred)
        # r2 = r2_score(test.values[:10], y_pred.values[:10])

    'storeId' in df.columns
    # assert ('storeId' in df.columns) and ('dateTime' in df.columns)
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

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
