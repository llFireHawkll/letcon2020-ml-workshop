'''
File: classification.py
Project: metrics
File Created: Monday, 17th August 2020 5:26:33 pm
Author: Sparsh Dutta (sparsh.dtt@gmail.com)
-----
Last Modified: Wednesday, 19th August 2020 12:03:01 am
Modified By: Sparsh Dutta (sparsh.dtt@gmail.com>)
-----
Copyright 2020 Sparsh Dutta
'''
from sklearn import metrics as skmetrics

import numpy as np

__all__ = ['ClassificationMetrics']

class ClassificationMetrics:
    def __init__(self, metric):
        """
        init class for classification metrics
        """
        self.metrics = {
            "accuracy": self._accuracy,
            "auc": self._auc,
            "f1": self._f1,
            "kappa": self._kappa,
            "logloss": self._log_loss,
            "precision": self._precision,
            "quadratic_kappa": self._quadratic_weighted_kappa,
            "recall": self._recall,
        }

        self.metric = metric
        
        if self.metric not in self.metrics:
            raise Exception("Invalid metric passed")

    def __call__(self, y_true, y_pred, y_proba=None):
        if self.metric == "auc":
            if y_proba is not None:
                return self.metrics[self.metric](y_true, y_proba[:, 1])
            else:
                return np.nan
        elif self.metric == "logloss":
            if y_proba is not None:
                return self.metrics[self.metric](y_true, y_proba[:, 1])
            else:
                return np.nan
        elif self.metric == "multiclass_logloss":
            if y_proba is not None:
                return self.metrics[self.metric](y_true, y_proba)
            else:
                return np.nan
        else:
            return self.metrics[self.metric](y_true, y_pred)

    @staticmethod
    def _accuracy(y_true, y_pred):
        return skmetrics.accuracy_score(y_true=y_true, y_pred=y_pred)

    def _auc(self, y_true, y_pred):
        return skmetrics.roc_auc_score(y_true=y_true, y_score=y_pred)

    @staticmethod
    def _f1(y_true, y_pred):
        return skmetrics.f1_score(y_true=y_true, y_pred=y_pred)

    @staticmethod
    def _log_loss(y_true, y_pred):
        return skmetrics.log_loss(y_true=y_true, y_pred=y_pred)

    @staticmethod
    def _kappa(y1, y2, weights):
        return skmetrics.cohen_kappa_score(y1=y1, y2=y2, weights=weights)

    @staticmethod
    def _precision(y_true, y_pred):
        return skmetrics.precision_score(y_true=y_true, y_pred=y_pred)

    @staticmethod
    def _quadratic_weighted_kappa(y1, y2):
        return skmetrics.cohen_kappa_score(y1, y2, weights="quadratic")

    @staticmethod
    def _recall(y_true, y_pred):
        return skmetrics.recall_score(y_true=y_true, y_pred=y_pred)