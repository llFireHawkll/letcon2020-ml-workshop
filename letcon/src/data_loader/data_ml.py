'''
File: data_ml.py
Project: data_loader
File Created: Monday, 17th August 2020 4:18:40 pm
Author: Sparsh Dutta (sparsh.dtt@gmail.com)
-----
Last Modified: Tuesday, 18th August 2020 3:22:10 pm
Modified By: Sparsh Dutta (sparsh.dtt@gmail.com>)
-----
Copyright 2020 Sparsh Dutta
'''
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from letcon.src.logger.logger import logger

import os
import copy
import numpy as np
import pandas as pd

__all__ = ['DataLoader', 'ProcessPredictionData']

class DataLoader:
    def __init__(self, config):
        """
        Args:
            config ([dict]): [contains data config architecture]
        """
        # Initializing logger for dataloaderml class
        logger_function = logger(name='DataLoaderML')

        # Initializing class attributes
        self._task = config['task']
        self._seed = config['random_state']

        # Initializating data attributes
        self.train_data = config['train_data']
        self.test_data = config['test_data']

        # Initializing the variables
        self.unique_id = config['unique_id']
        self.time_id = config['time_id']
        self.x_vars = config['x_vars']
        self.cat_vars = config['cat_vars']
        self.y_var = config['y_var']

        self.x_modelling_vars = copy.deepcopy(list(set(self.x_vars) -\
                                                   set(self.cat_vars)))
        
        # Intializing train_test_split attributes
        self.stratify = config['stratify']
        self.test_size = config['test_size']
        self.use_full_dataset = config['use_full_dataset']

        # Initializing data processing attributes
        self.encoding_style = config['encoding_style']
        self.impute_missing = config['impute_missing']
        self.capping_vars = config['capping_vars']

        self.encoding_dict = {}
        
        # run data processing
        self.process_data()

    def get_missing_variables(self):
        self.train_missing_vars = list()
        self.test_missing_vars = list()

        for variable in self.x_vars:
            if self.train_data[variable].isnull().sum() != 0:
                self.train_missing_vars.append(variable)

            if self.test_data[variable].isnull().sum() != 0:
                self.test_missing_vars.append(variable)

    def run_standard_checks(self):
        # Running data check 
        if self.train_data is None:
            raise ValueError('train_data cannot be None. Provide a dataframe.')

        if self.test_data is None:
            print('test_data is none. Setting test_data equal to train_data')
            self.test_data = copy.deepcopy(self.train_data)

        # Running variable check
        assert len(self.x_vars) != 0
        assert (self.y_var != None) or (self.y_var != '')
        assert list(set(self.cat_vars) - set(self.cat_vars)) == list()

        # Running train_test attribute check
        assert type(self.use_full_dataset) == bool

        # Running data processing attribute check
        assert (type(self.encoding_style) == str) or (type(self.encoding_style) == dict)
        assert (type(self.impute_missing) == str) or (type(self.impute_missing) == dict)
        assert type(self.capping_vars) == dict

        if type(self.encoding_style) == dict:
            assert len(self.encoding_style) == len(self.cat_vars)

        # Find missing variables
        self.get_missing_variables()

    @staticmethod
    def do_imputation(data,
                      column='',
                      imputation_method='median'):
        if data[column].dtype == 'object':
            value = data[column].mode()[0]
        else:
            if imputation_method == 'mean':
                value = data[column].mean()
            elif imputation_method == 'median':
                value = data[column].median()
            elif imputation_method == 'mode':
                value = data[column].mode()[0]
            elif imputation_method == 'max':
                value = data[column].max()
            elif imputation_method == 'min':
                value = data[column].min()
            elif imputation_method == 'zero':
                value = 0.0
            else:
                raise NotImplementedError
            
        data[column].fillna(value, inplace=True)

        return data

    @staticmethod
    def do_capping(data,
                   column='',
                   percentiles_range=[]):
        """[Cap the column based on the percentile quantile range]

        Args:
            data ([pandas dataframe]): [reference of dataframe]
            column (str, column_name): [column to be capped]. Defaults to ''.
            percentiles_range (list, percentile quantile range): [percentile quantile range]. Defaults to [].

        Returns:
            [pandas dataframe]: [reference of the dataframe with capped column]
        """
        # calculating the percentile value or lower/upper limit to cap
        percentiles = data[column].quantile(percentiles_range).values
        
        # capping the variable -> lower limit
        data[column] = np.where(data[column] <= percentiles[0], percentiles[0], data[column])

        # capping the variable -> upper limit
        data[column] = np.where(data[column] >= percentiles[1], percentiles[1], data[column])

        return data

    @staticmethod
    def do_encoding(data,
                    column='',
                    encoding_dict={},
                    encoding_method='label_encoder'):
        
        if encoding_method == 'label_encoder':
            encoding_dict[column] = LabelEncoder()
            data[column] = encoding_dict[column].fit_transform(data[column])

        elif encoding_method == 'one_hot_encoder':
            raise NotImplementedError

        return data, encoding_dict

    def do_preprocessing(self):
        
        # First we do imputation
        if type(self.impute_missing) == str:
            imputation_method = self.impute_missing

            for variable in self.train_missing_vars:
                self.train_data = self.do_imputation(data=self.train_data,
                                                     column=variable,
                                                     imputation_method=imputation_method)

            for variable in self.test_missing_vars:
                self.test_data = self.do_imputation(data=self.test_data,
                                                    column=variable,
                                                    imputation_method=imputation_method)                                
        else:
            for variable in self.impute_missing.keys():
                imputation_method = self.impute_missing[variable]
                
                self.train_data = self.do_imputation(data=self.train_data,
                                                     column=variable,
                                                     imputation_method=imputation_method)

                self.test_data = self.do_imputation(data=self.test_data,
                                                    column=variable,
                                                    imputation_method=imputation_method) 

        # Second we do capping of variables
        for variable in self.capping_vars.keys():
            percentile_range = self.capping_vars[variable]
            
            if variable == self.y_var:
                self.train_data = self.do_capping(data=self.train_data,
                                                  column=variable,
                                                  percentiles_range=percentile_range)
            else:
                self.train_data = self.do_capping(data=self.train_data,
                                                  column=variable,
                                                  percentiles_range=percentile_range)
                
                self.test_data = self.do_capping(data=self.test_data,
                                                 column=variable,
                                                 percentiles_range=percentile_range)
        
        # Third we do encoding of categorical variables
        if type(self.encoding_style) == str:
            for variable in self.cat_vars:
                encoding_method = self.encoding_style[variable]

                if encoding_method == 'label_encoder':
                    self.x_modelling_vars = self.x_modelling_vars + [variable]
                elif encoding_method == 'one_hot_encoder':
                    raise NotImplementedError


                self.train_data, self.encoding_dict = self.do_encoding(data=self.train_data,
                                                                       column=variable,
                                                                       encoding_dict=self.encoding_dict,
                                                                       encoding_method=encoding_method)
                
                self.test_data, self.encoding_dict = self.do_encoding(data=self.test_data,
                                                                      column=variable,
                                                                      encoding_dict=self.encoding_dict,
                                                                      encoding_method=encoding_method)
        else:
            for variable in self.encoding_style.keys():
                encoding_method = self.encoding_style[variable]
                
                if encoding_method == 'label_encoder':
                    self.x_modelling_vars = self.x_modelling_vars + [variable]
                elif encoding_method == 'one_hot_encoder':
                    raise NotImplementedError

                self.train_data, self.encoding_dict = self.do_encoding(data=self.train_data,
                                                                       column=variable,
                                                                       encoding_dict=self.encoding_dict,
                                                                       encoding_method=encoding_method)
                
                self.test_data, self.encoding_dict = self.do_encoding(data=self.test_data,
                                                                      column=variable,
                                                                      encoding_dict=self.encoding_dict,
                                                                      encoding_method=encoding_method)
                            
    def get_train_val_split(self):
        if self.use_full_dataset:
            (self.xtrain, 
            self.xvalid, 
            self.ytrain, 
            self.yvalid) = (self.train_data[self.x_modelling_vars], 
                            self.train_data[self.x_modelling_vars],
                            self.train_data[self.y_var], 
                            self.train_data[self.y_var])
        else:
            if self.stratify == None:
                (self.xtrain, 
                 self.xvalid, 
                 self.ytrain, 
                 self.yvalid) = train_test_split(self.train_data[self.x_modelling_vars],
                                                 self.train_data[self.y_var],
                                                 test_size=self.test_size,
                                                 random_state=self._seed
                                                )
            else:
                (self.xtrain, 
                 self.xvalid, 
                 self.ytrain, 
                 self.yvalid) = train_test_split(self.train_data[self.x_modelling_vars],
                                                 self.train_data[self.y_var],
                                                 stratify=self.train_data[self.stratify],
                                                 test_size=self.test_size,
                                                 random_state=self._seed
                                                )
        # Creating attribute for test data
        self.xtest = copy.deepcopy(self.test_data[self.x_modelling_vars])

    def process_data(self):
        """
            Encapsulating all data processing steps

        """
        # Step-1 Run data checks
        self.run_standard_checks()
        # Step-2 Run imputation and encoding
        self.do_preprocessing()
        # Step-3 Generate train and validation split dataframe
        # with test data aswell for prediction
        self.get_train_val_split()
    
    def print_data_stats(self):
        print('Train Size: {0}, Valid Size: {1}, Test Size: {2}'.format(self.xtrain.shape,
                                                                        self.xvalid.shape,
                                                                        self.xtest.shape))

    def get_data(self):
        return (self.xtrain,self.xvalid,self.ytrain,self.yvalid,self.xtest)

    def get_data_artifacts(self):
        data_artifacts_dict = dict()

        data_artifacts_dict['task'] = self._task
        data_artifacts_dict['x_modelling_vars'] = self.x_modelling_vars

        return data_artifacts_dict

    def __str__(self):
        return 'Train and Test Size are {0}, {1}\nStratified on {2} with Stratification Ratio:{3}'.format(self.xtrain.shape,
                                                                                                          self.xtest.shape,
                                                                                                          self.stratify,
                                                                                                          self.test_size)

    def __repr__(self):
        return {'Train Size': self.xtrain.shape,
                'Test Size' : self.xtest.shape,
                'Stratified on' : self.stratify,
                'Stratification Ratio' : self.test_size}

class ProcessPredictionData:
    def __init__(self, config):
        
        self._task = config['task']
        self.x_modelling_vars = config['x_modelling_vars']
    
    def do_preprocessing(self, x_data):
        data_to_predict = np.array([x_data[feature] for feature in self.x_modelling_vars]).reshape(1, -1)
        data_to_predict = pd.DataFrame(data_to_predict, columns=self.x_modelling_vars)
        return data_to_predict      