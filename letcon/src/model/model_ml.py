'''
File: model_ml.py
Project: model
File Created: Monday, 17th August 2020 4:17:57 pm
Author: Sparsh Dutta (sparsh.dtt@gmail.com)
-----
Last Modified: Tuesday, 18th August 2020 3:24:14 pm
Modified By: Sparsh Dutta (sparsh.dtt@gmail.com>)
-----
Copyright 2020 Sparsh Dutta
'''
from letcon.src.metrics.classification import ClassificationMetrics
from letcon.src.metrics.regression import RegressionMetrics
from letcon.src.optimizer.optuna import OptunaOptimizer

import os
import copy
import shap
import numpy as np
import pandas as pd
import xgboost as xgb
import catboost as cgb
import matplotlib.pyplot as plt
 
__all__ = ['Model', 'PredictOnNewData']

class Model:
    def __init__(self, config):   
        # Initializing class attributes
        self._task = config['task']
        self._seed = config['random_state']

        # Initializing data attributes
        (self.xtrain, self.xvalid,
         self.ytrain, self.yvalid, 
         self.xtest) = config['data']

        # Intializing num_class attribute if task is classification
        if self._task == 'classification':
            self.num_class = self.ytrain.nunique()

        # Intializing model attributes
        self.model_type = config['model_type']
        self.model_name = config['model_name']
        self.model_initial_params = config['model_initial_params']

        self.scoring_function_name = config['scoring_function']
        if self._task == 'regression':
            self.scoring_function = RegressionMetrics(metric=self.scoring_function_name)
        elif self._task == 'classification':
            self.scoring_function = ClassificationMetrics(metric=self.scoring_function_name)

        # 
        self.define_model_inputs(config=config['model_inputs'])

        # Intializing model hyperparameter training attributes
        self.enable_tuning = config['hyperparmeter_tuning']['enable_tuning']

        if self.enable_tuning:
            self.define_optimizer_settings(config=config['hyperparmeter_tuning'])

        # Initializing shap attributes
        self.enable_shap = config['shap_analysis']['enable_shap']

        if self.enable_shap:
            self.define_shap_settings(config=config['shap_analysis'])

    def define_model_inputs(self, config):
        if self.model_type == 'simple':
            if self.model_name == 'xgboost':
                pass
            elif self.model_name == 'catboost':
                self.cat_vars = config['cat_vars']

    def define_optimizer_settings(self, config):
        self.optimizer_type = config['optimizer']

        if self.optimizer_type == 'optuna':
            self.optimizer_params = config['optimizer_params']
            
    def define_shap_settings(self, config):
        self.shap_explainer = config['use_explainer']

    def vanilla_xgboost(self, params):
        xgb_train = xgb.DMatrix(self.xtrain, self.ytrain)
        xgb_valid = xgb.DMatrix(self.xvalid, self.yvalid)

        if self._task == 'regression':
            model = xgb.train(params=params,
                              dtrain=xgb_train,
                              evals=[(xgb_train, 'train'), (xgb_valid, 'valid')],
                              verbose_eval=False,
                              num_boost_round=500,
                              early_stopping_rounds=10)

        elif self._task == 'classification':
            params['objective'] = 'multi:softmax'
            params['num_class'] = self.num_class
            
            model = xgb.train(params=params,
                              dtrain=xgb_train,
                              evals=[(xgb_train, 'train'), (xgb_valid, 'valid')],
                              verbose_eval=False,
                              num_boost_round=500,
                              early_stopping_rounds=10)
        else:
            raise ValueError('task got unknown argument. Expected from [regression, classification]')

        return model 

    def vanilla_catboost(self, params):
        if self._task == 'regression':
            model = cgb.CatBoostRegressor(**params)
        elif self._task == 'classification':
            model = cgb.CatBoostClassifier(**params)
        else:
            raise ValueError('task got unknown argument. Expected from [regression, classification]')
        
        # Fitting Model
        model.fit(self.xtrain,self.ytrain, cat_features=self.cat_vars)

        return model

    def create_model(self):
        if self.model_type == 'simple':
            if self.model_name == 'xgboost':
                if self.enable_tuning:
                    if self.optimizer_type == 'optuna':
                            optuna_config = {'seed' : self._seed,
                                             'model' : self.vanilla_xgboost,
                                             'params' : self.optimizer_params,
                                             'score_function' : self.do_model_inference,
                                            }
                            self.model_best_params = OptunaOptimizer(config=optuna_config).best_params
                            self.trained_model = self.vanilla_xgboost(params=self.model_best_params)
                    else:
                        raise NotImplementedError
                else:
                    self.model_best_params = self.model_initial_params
                    self.trained_model = self.vanilla_xgboost(params=self.model_best_params)
            elif self.model_name == 'catboost':
                self.model_best_params = self.model_initial_params
                self.trained_model = self.vanilla_catboost(params=self.model_best_params)
        else:
            raise ValueError('task got unknown argument. Expected from [regression, classification]')
    
    def do_model_inference(self, 
                           model,
                           return_score=False):
        if self.model_type == 'simple':
            if self.model_name == 'xgboost':
                self.ytrain_pred = model.predict(xgb.DMatrix(self.xtrain))
                self.yvalid_pred = model.predict(xgb.DMatrix(self.xvalid))
                self.ytest_pred = model.predict(xgb.DMatrix(self.xtest))
            
            elif self.model_name == 'catboost':
                self.ytrain_pred = model.predict(self.xtrain)
                self.yvalid_pred = model.predict(self.xvalid)
                self.ytest_pred = model.predict(self.xtest)                    
        else:
            raise ValueError('task got unknown argument. Expected from [regression, classification]')
    
        if return_score:
            train_score = self.scoring_function(y_true=self.ytrain, y_pred=self.ytrain_pred)
            valid_score = self.scoring_function(y_true=self.yvalid, y_pred=self.yvalid_pred)

            return (train_score, valid_score)

    def do_model_evaluation(self):
        if self.model_type == 'simple':
            if self.model_name in ['xgboost', 'catboost']:
                print('Train {0} Score: {1}'.format(self.scoring_function_name,
                                                        self.scoring_function(y_true=self.ytrain, 
                                                                              y_pred=self.ytrain_pred)))
                                                                    
                print('Validation {0} Score: {1}'.format(self.scoring_function_name,
                                                            self.scoring_function(y_true=self.yvalid, 
                                                                                y_pred=self.yvalid_pred)))
        else:
            raise ValueError('task got unknown argument. Expected from [regression, classification]')  

    def do_shap_analysis(self):
        if self.model_type == 'simple':
            self.shap_explainer = shap.TreeExplainer(self.trained_model)
            
            if self.model_name == 'xgboost':
                self.shap_values = self.shap_explainer.shap_values(xgb.DMatrix(self.xtrain))   
            elif self.model_name == 'catboost':
                self.shap_values = self.shap_explainer.shap_values(self.xtrain) 
            
            plt.figure(figsize=(15,7))
            shap.summary_plot(self.shap_values, self.xtrain, max_display=30)

            plt.figure(figsize=(15,7))
            shap.summary_plot(self.shap_values, self.xtrain, max_display=30, plot_type='bar')
        else:
            raise NotImplementedError
    
    def get_model_artifacts(self):

        model_artifacts_dict = dict()

        model_artifacts_dict['task'] = self._task
        model_artifacts_dict['model'] = self.trained_model
        model_artifacts_dict['model_name'] = self.model_name
        model_artifacts_dict['model_type'] = self.model_type
        model_artifacts_dict['shap_explainer'] = self.shap_explainer
        model_artifacts_dict['scoring_function'] = self.scoring_function
        model_artifacts_dict['model_best_params'] = self.model_best_params

        return model_artifacts_dict

class PredictOnNewData:
    def __init__(self, config):
        self._task = config['task']
        self.model = config['model']
        self.model_name = config['model_name']
        self.model_type = config['model_type']
        self.shap_explainer = config['shap_explainer']
        self.scoring_function = config['scoring_function']
        self.model_best_params = config['model_best_params']
        
    def predict(self, x_data):
        if self.model_type == 'simple':
            if self.model_name == 'xgboost':
                prediction = self.model.predict(xgb.DMatrix(x_data))
            elif self.model_name == 'catboost':
                prediction = self.model.predict(x_data)

        return prediction
