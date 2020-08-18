'''
File: regression.py
Project: metrics
File Created: Monday, 17th August 2020 3:14:47 pm
Author: Sparsh Dutta (sparsh.dtt@gmail.com)
-----
Last Modified: Tuesday, 18th August 2020 3:23:05 pm
Modified By: Sparsh Dutta (sparsh.dtt@gmail.com>)
-----
Copyright 2020 Sparsh Dutta
'''
from sklearn import metrics as skmetrics

import numpy as np

__all__ = ['RegressionMetrics']

class RegressionMetrics:
    def __init__(self, metric):
        """
            Constructor class for regression metrics

        """
        self.metrics = {
                "mae" : self._mae,
                "msle": self._msle,
                "mse": self._mse,
                "rmsle": self._rmsle,
                "rmse": self._rmse,
                "r2": self._r2,
        }
        self.metric = metric
        
        if self.metric not in self.metrics:
            raise Exception("Invalid metric passed")

    def __call__(self, y_true, y_pred):
        return self.metrics[self.metric](y_true, y_pred)

    @staticmethod
    def _mae(y_true, y_pred):
        return skmetrics.mean_absolute_error(y_true=y_true, y_pred=y_pred)

    @staticmethod
    def _msle(y_true, y_pred):
        return skmetrics.mean_squared_log_error(y_true=y_true, y_pred=y_pred)

    @staticmethod
    def _mse(y_true, y_pred):
        return skmetrics.mean_squared_error(y_true=y_true, y_pred=y_pred)

    def _rmsle(self, y_true, y_pred):
        return np.sqrt(self._msle(y_true=y_true, y_pred=y_pred))

    def _rmse(self, y_true, y_pred):
        return np.sqrt(self._mse(y_true=y_true, y_pred=y_pred))

    @staticmethod
    def _r2(y_true, y_pred):
        return skmetrics.r2_score(y_true=y_true, y_pred=y_pred)