'''
File: utils.py
Project: utils
File Created: Tuesday, 18th August 2020 12:26:55 am
Author: Sparsh Dutta (sparsh.dtt@gmail.com)
-----
Last Modified: Tuesday, 18th August 2020 7:53:27 pm
Modified By: Sparsh Dutta (sparsh.dtt@gmail.com>)
-----
Copyright 2020 Sparsh Dutta
'''
from letcon.src.config import ARTIFACT_OUTPUT_PATH

import pickle
import numpy as np
import pandas as pd

## Object Related Utilities
def save_artifacts(data_object, model_object):
    pickle.dump(data_object, open(ARTIFACT_OUTPUT_PATH + '/' + 'data_object.pickle', 'wb'))
    pickle.dump(model_object, open(ARTIFACT_OUTPUT_PATH + '/' + 'model_object.pickle', 'wb'))

def load_artifacts(data_filename, model_filename):
    """[Load saved data and model artifacts]

    Args:
        data_filename ([str]): [name of data object used while saving]
        model_filename ([str]): [name of model object used while saving]

    Returns:
        [tuple]: [tuple of data and trained model object]
    """
    data_pipeline = pickle.load(open(ARTIFACT_OUTPUT_PATH + '/' + data_filename + '.pickle', 'rb'))
    trained_model = pickle.load(open(ARTIFACT_OUTPUT_PATH + '/' + model_filename + '.pickle', 'rb'))
    
    return data_pipeline, trained_model

## Data Related Utilities
def get_data_statistics(data):
    print('DataFrame Contains {0} Rows & {1} Columns\n'.format(data.shape[0], 
                                                               data.shape[1]))

    print('Sample DataFrame: \n{0}\n'.format(data.head().to_string()))

def get_missing_values_stats(data):
        """ Returns the % of missing values in dataframe
        """
        zero_val_sum_for_each_column = (data == 0.00).astype(int).sum(axis=0)
        missing_val_sum_for_each_column = data.isnull().sum()
        percent_missing_val_for_each_column = (100 * missing_val_sum_for_each_column) / data.shape[0]

        missing_val_dataframe = pd.concat([zero_val_sum_for_each_column, 
                                           missing_val_sum_for_each_column, 
                                           percent_missing_val_for_each_column], axis=1)
                                           
        missing_val_dataframe = missing_val_dataframe.rename(columns = {0 : "Zero Values", 
                                                                        1 : "Missing Values", 
                                                                        2 : "% of Total Values"})
        
        missing_val_dataframe["Total Zero Missing Values"] = missing_val_dataframe["Zero Values"] + \
                                                             missing_val_dataframe["Missing Values"]

        missing_val_dataframe["% Total Zero Missing Values"] = (100 * missing_val_dataframe["Total Zero Missing Values"]) \
                                                                 / data.shape[0]

        missing_val_dataframe["Data Type"] = data.dtypes

        # Sorting the dataframe based on %missing column
        missing_val_dataframe = missing_val_dataframe[missing_val_dataframe.iloc[:,1] != 0]\
                                .sort_values(by=["% of Total Values"], 
                                             ascending=False)\
                                .round(1)
        
        output_string = 'DataFrame has {0} columns and {1} rows.\nThere are {2} columns that have missing values\n'

        print(output_string.format(str(data.shape[1]), 
                                   str(data.shape[0]),
                                   str(missing_val_dataframe.shape[0])))
        
        print('{0}\n'.format(missing_val_dataframe.to_string()))

