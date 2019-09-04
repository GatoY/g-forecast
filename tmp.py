import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

import mlflow
mlflow.log_params
x = np.random.normal(size=100)
sns.distplot(x)