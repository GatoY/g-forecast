import copy
from sklearn.metrics import r2_score
from sklearn.model_selection import ParameterGrid
import pandas as pd

import mlflow
import mlflow.sklearn

class ParamsGridSearch():
    def __init__(self, 
                model_name, 
                estimator,
                param_grid, 
                verbose=0):
        """
        Params:

        Output:
        """

        self.model_name = model_name
        self.estimator = estimator
        self.param_grid = param_grid
        self.verbose = verbose

        self.best_estimator_ = None
        self.best_score_ = float('-inf')
        self.best_params = None

        self.results_ = pd.DataFrame(columns=['Params', 'Score'])

    def fit(self, X, y, valid_X, valid_y):
        """
        X : array-like, shape = [n_samples, n_features]
        Training vector, where n_samples is the number of samples and n_features is the number of features.

        y : array-like, shape = [n_samples] or [n_samples, n_output], optional
        Target relative to X for classification or regression; None for unsupervised learning.
        """
        
        with mlflow.start_run():
            for param in list(ParameterGrid(self.param_grid)):
                estimator = clone(self.estimator)
                estimator.set_params(**param)
                estimator.fit(X, y)
                score = r2_score(valid_y, estimator.predict(valid_X))
                self.results_.loc[len(self.results_)] = [param, score]

                if score > self.best_score_:
                    self.best_score_ = score
                    self.best_estimator_ = estimator
                    self.best_params = param
                if self.verbose > 0:
                    print(param)
                    print('score %s' % score)

                mlflow.log_param('model', self.model_name)
                mlflow.log_params(param)
                mlflow.log_metric('r2_score', score)
                mlflow.sklearn.log_model(estimator, "model")


    def results(self):
        return self.results_

    def predict(self, X):
        return self.best_estimator_.predict(X)

    def score(self, X, y):
        return r2_score(y, self.best_estimator_.predict(X))


def clone(estimator, safe=True):
    """Constructs a new estimator with the same parameters.

    Clone does a deep copy of the model in an estimator
    without actually copying attached data. It yields a new estimator
    with the same parameters that has not been fit on any data.

    Parameters
    ----------
    estimator : estimator object, or list, tuple or set of objects
        The estimator or group of estimators to be cloned

    safe : boolean, optional
        If safe is false, clone will fall back to a deep copy on objects
        that are not estimators.

    """

    def iteritems(d, **kw):
        """Return an iterator over the (key, value) pairs of a dictionary."""
        return iter(getattr(d, 'items')(**kw))

    estimator_type = type(estimator)
    # XXX: not handling dictionaries
    if estimator_type in (list, tuple, set, frozenset):
        return estimator_type([clone(e, safe=safe) for e in estimator])
    elif not hasattr(estimator, 'get_params'):
        if not safe:
            return copy.deepcopy(estimator)
        else:
            raise TypeError("Cannot clone object '%s' (type %s): "
                            "it does not seem to be a scikit-learn estimator "
                            "as it does not implement a 'get_params' methods."
                            % (repr(estimator), type(estimator)))
    klass = estimator.__class__
    new_object_params = estimator.get_params(deep=False)
    for name, param in iteritems(new_object_params):
        new_object_params[name] = clone(param, safe=False)
    new_object = klass(**new_object_params)
    params_set = new_object.get_params(deep=False)

    # quick sanity check of the parameters of the clone
    for name in new_object_params:
        param1 = new_object_params[name]
        param2 = params_set[name]
        if param1 is not param2:
            raise RuntimeError('Cannot clone object %s, as the constructor '
                               'either does not set or modifies parameter %s' %
                               (estimator, name))
    return new_object
