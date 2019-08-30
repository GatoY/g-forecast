# -------------- #

import pandas as pd
import numpy as np
import datetime
import pickle
import time
from util_log import logger

from sklearn.metrics import r2_score
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from params_search import ParamsGridSearch

import warnings

warnings.filterwarnings("ignore")


def load_models():
    return {
        'CatBoost': {
            'model_fn': CatBoostRegressor(verbose=False),
            'params': {
                'iterations': [50],
                'depth': [6],
                'learning_rate': [0.5],
                'loss_function': ['RMSE']
            }
        },



        'LGBMRegressor': {
            'model_fn': LGBMRegressor(),
            'params': {
                'colsample_bytree': [0.8],
                'learning_rate': [0.08],  # [0.08, 0.1]
                'max_depth': [7],
                'num_leaves': [60],
                'min_child_weight': [5],
                'n_estimators': [300],  # [300, 500],
                # 'nthread': [4],
                'seed': [1337],
                'silent': [1],
                'subsample': [0.8]
                }
        },

        'random_forest': {
            'model_fn': RandomForestRegressor(n_jobs=1),
            'params': {
                # 'n_jobs': -1,
                'n_estimators': [50, 200, 300],
                # 'warm_start': True,
                'max_features': ['auto'],
                'max_depth': [5, 7],
                # 'min_samples_leaf': 2,
            }
        },

        'XGBoost': {
            'model_fn': XGBRegressor(),
            'params': {
                # when use hyperthread, xgboost may become slower
                'nthread': [4],
                'objective': ['reg:linear'],
                'learning_rate': [.03, 0.05],  # so called `eta` value
                'max_depth': [6, 7],
                'min_child_weight': [4],
                'silent': [1],
                'subsample': [0.7],
                'colsample_bytree': [0.7],
                'n_estimators': [300, 500]},
        }
    }
    
# def lgb(train_X, train_y, valid_X, valid_y, model_dir, i):
#     np.save(str(i)+'_train_X.npy', train_X)
#     np.save(str(i)+'_train_y.npy', train_y)
#     np.save(str(i)+'_valid_X.npy', valid_X)
#     np.save(str(i)+'_valid_y.npy', valid_y)

#     m = LGBMRegressor(**{'colsample_bytree': 0.8,
#             'learning_rate': 0.08,  
#             'max_depth': 7,
#             'num_leaves': 60,
#             'min_child_weight': 5,
#             'n_estimators': 300,  
#             'seed': 1337,
#             'silent': 1,
#             'subsample': 0.8})
#     m.fit(train_X, train_y)
#     logger.debug(r2_score(valid_y, m.predict(valid_X)))
#     pickle.dump(m, open(
#                 model_dir + 'best_model_' + str(i) + '.pkl', 'wb'))     



def grid_search(start_date,
                predict_length,
                train_test_dir,
                model_dir,
                train_valid_dict,
                model_list=['LGBMRegressor',
                            'XGBoost',
                            'CatBoost'],
                verbose=0
                ):

    model_results = pd.DataFrame(
        columns=['model_name', 'predict_days', 'r2_score'])

    logger.debug('load models')
    models = load_models()


    for i in range(predict_length):
        logger.debug('train %sth model' % i)

        train_X = train_valid_dict[i]['train_X']
        train_y = train_valid_dict[i]['train_y']
        valid_X = train_valid_dict[i]['valid_X']
        valid_y = train_valid_dict[i]['valid_y']

        # lgb(train_X, train_y, valid_X, valid_y, model_dir, i)
        # continue

        best_model = None
        best_score = float('-inf')
        for model_name in model_list:
            m = models[model_name]['model_fn']
            params = models[model_name]['params']
            m_grid = ParamsGridSearch(
                model_name,
                m, 
                params,
                verbose=verbose)
            m_grid.fit(train_X, train_y, valid_X, valid_y)
            results = m_grid.results()
            results['name'] = model_name
            results.to_csv(model_dir + model_name + '_' +
                           str(i) + '.csv', index=False)

            score = m_grid.best_score_
            if score > best_score:
                best_score = score
                best_model = m_grid.best_estimator_
            # print([model_name, i, score, str((e - s).seconds) + 's'])
            model_results.loc[len(model_results)] = [model_name, i, score]

        pickle.dump(best_model, open(
            model_dir + 'best_model_' + str(i) + '.pkl', 'wb'))        
    model_results.to_csv(model_dir + 'model_result.csv', index=False)
    # print('it spends %ss' % int(end - start))
