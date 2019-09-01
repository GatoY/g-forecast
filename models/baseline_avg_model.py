import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

import mlflow
import mlflow.sklearn

# from config import (
#     SPLIT_DATE
# )

# %%
class AvgModel(object):
    def __init__(
                self, 
                label,
                train_days):
        self.label = label 
        self.train_days = train_days
        return

    def averaging_by_weekday(
                            self, 
                            df_item, 
                            column):
        """
        
        """
        result = {}
        for k in range(7):
            chunk = df_item[df_item['weekday'] == k]
            if self.train_days > 0:
                start_date = chunk['date'].iloc[-1] - \
                    pd.Timedelta(days=self.train_days)
                chunk = chunk[chunk['date'] >= start_date]
                # print(chunk)
            result[k] = chunk[column].mean()
        return result

    def fit(self, df):
        self.model = {}
        self.model_item = {}
        store_list = df['store_nbr'].unique()
        for store_id in store_list:
            df_store = df[df['store_nbr'] == store_id]
            item_list = df_store.item_nbr.unique()
            self.model[store_id] = {}
            for item_id in item_list:
                df_item = df_store[df_store['item_nbr'] == item_id]
                result =  self.averaging_by_weekday(df_item, self.label)
                self.model[store_id][item_id] = result
                if item_id not in self.model_item:
                    pass
                else:
                    o_r = self.model_item[item_id]
                    for k in range(7):
                        result[k] = (result[k]+o_r[k])/2
                self.model_item[item_id] = result
        return

    def predict(self, valid_df):
        def create_fn(row):
            try:
                return self.model[row['store_nbr']][row['item_nbr']][row['weekday']]
            except:
                return self.model_item[row['item_nbr']][row['weekday']]

        pred = valid_df.apply(create_fn, axis=1)
        print(pred)
        return pred




if __name__ == '__main__':
    main()
